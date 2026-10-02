"""Validate a ``RunConfig`` against the installed upstream APIs (trainer backend, verifiers).

Schema validation (our own sections) already happened to produce the ``RunConfig`` instance
(``config.loader.load_config`` raises ``ConfigError`` on schema failure before this module
ever runs). What is left is the opaque passthrough dicts: ``trainer.args``, ``environment``,
``rollout.sampling``. Those are only checked against upstream libraries that are actually
importable in this environment (e.g. ``unsloth`` is Linux-only) — an import failure is a
warning, not an error, so ``mg-rl validate`` stays useful on a Mac CPU dev box.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from moongazer_rl.config.schema import RunConfig
from moongazer_rl.policy.base import resolve_sampling_temperature


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def __bool__(self) -> bool:
        return self.ok

    def render(self) -> str:
        lines: list[str] = []
        if self.ok and not self.warnings:
            lines.append("OK: no errors or warnings")
        for e in self.errors:
            lines.append(f"ERROR: {e}")
        for w in self.warnings:
            lines.append(f"WARNING: {w}")
        return "\n".join(lines)


def _validate_trainer_args(config: RunConfig, report: ValidationReport) -> None:
    from moongazer_rl.trainer.registry import resolve_backend

    try:
        backend_cls = resolve_backend(config.trainer.backend)
    except KeyError as exc:
        report.errors.append(f"trainer.backend: {exc}")
        return
    except ImportError as exc:
        report.warnings.append(
            f"backend {config.trainer.backend!r} not importable here: {exc}; "
            "trainer.args unvalidated"
        )
        return

    try:
        backend = backend_cls(config)
        set_planned = getattr(backend, "set_planned_steps", None)
        if callable(set_planned):  # LR-schedule horizon must be known before validation
            set_planned(config.rollout.max_steps)
        problems = backend.validate_args(config.trainer.args)
    except ImportError as exc:
        report.warnings.append(
            f"backend {config.trainer.backend!r} not importable here: {exc}; "
            "trainer.args unvalidated"
        )
        return
    except Exception as exc:  # noqa: BLE001 - a misbehaving backend must not crash validate
        report.errors.append(
            f"trainer.args: backend {config.trainer.backend!r} raised during validation: {exc!r}"
        )
        return

    for problem in problems:
        report.errors.append(f"trainer.args: {problem}")


def _validate_environment(config: RunConfig, report: ValidationReport) -> None:
    try:
        import verifiers.v1 as vf1
    except ImportError as exc:
        report.warnings.append(f"environment plugin not installed: {exc}")
        return

    try:
        vf1.EnvConfig.model_validate(config.environment.to_env_config_dict())
    except (ImportError, ModuleNotFoundError) as exc:
        # EnvConfig's validator imports the taskset/harness plugin by id; missing plugin
        # surfaces here, not at the top-level `import verifiers.v1`.
        report.warnings.append(f"environment plugin not installed: {exc}")
    except Exception as exc:  # noqa: BLE001 - report, don't crash
        report.errors.append(f"environment: {exc}")


def _validate_sampling(config: RunConfig, report: ValidationReport) -> None:
    try:
        import verifiers.v1 as vf1
    except ImportError as exc:
        report.warnings.append(f"verifiers not importable for rollout.sampling validation: {exc}")
        return

    try:
        vf1.Sampling.model_validate(config.rollout.sampling)
    except Exception as exc:  # noqa: BLE001 - report, don't crash
        report.errors.append(f"rollout.sampling: {exc}")


POLICY_BACKENDS = ("auto", "vllm-server")


def _validate_policy(config: RunConfig, report: ValidationReport) -> None:
    backend = config.policy.backend
    if backend not in POLICY_BACKENDS:
        report.errors.append(f"policy.backend: unknown {backend!r}; available: {POLICY_BACKENDS}")
        return
    if backend == "auto":
        if config.policy.args:
            report.errors.append("policy.args: only meaningful with policy.backend: vllm-server")
        return
    if config.trainer.backend != "unsloth-native":
        report.errors.append(
            f"policy.backend: vllm-server is supported by trainer.backend unsloth-native only "
            f"(got {config.trainer.backend!r})"
        )
    from moongazer_rl.policy.vllm_server import VllmServerPolicy

    args = dict(config.policy.args)
    has_base_url = bool(args.get("base_url"))
    has_base_urls = "base_urls" in args
    if has_base_url == has_base_urls:
        if not any("exactly one of base_url or base_urls" in error for error in report.errors):
            report.errors.append(
                "policy.args must set exactly one of base_url or base_urls for "
                "policy.backend: vllm-server"
            )
        return
    if has_base_urls:
        base_urls = args.get("base_urls")
        if not isinstance(base_urls, list) or not all(
            isinstance(url, str) and url for url in base_urls
        ):
            report.errors.append("policy.args.base_urls must be a list of non-empty URLs")
            return
        if len(base_urls) < 2:
            report.errors.append("policy.args.base_urls must contain at least 2 URLs")
            return
        if len({url.rstrip("/") for url in base_urls}) != len(base_urls):
            report.errors.append("policy.args.base_urls must not contain duplicates")
            return
    args.setdefault("model", config.model.name)
    try:
        if has_base_urls:
            health_interval_s = args.pop("health_interval_s", 10.0)
            if not isinstance(health_interval_s, int | float):
                raise TypeError("health_interval_s must be a number")
            if health_interval_s < 0:
                raise ValueError("health_interval_s must be >= 0")
            urls = args.pop("base_urls")
            args["base_url"] = urls[0]
        VllmServerPolicy(**args)  # constructor-only: no network, checks shared kwargs and modes
    except TypeError as exc:
        report.errors.append(f"policy.args: {exc}")
    except ValueError as exc:
        report.errors.append(f"policy.args: {exc}")


def _validate_async(config: RunConfig, report: ValidationReport) -> None:
    """Reject async combinations that cannot preserve versioned sampling support."""
    async_config = config.async_rl
    if async_config is None:
        return
    if config.policy.backend != "vllm-server":
        report.errors.append(
            "async: policy.backend must be 'vllm-server'; the async path needs an external "
            "sampler with LoRA hot-load"
        )
    else:
        args = config.policy.args
        if bool(args.get("base_url")) == ("base_urls" in args):
            message = "async: policy.args must set exactly one of base_url or base_urls"
            if message not in report.errors:
                report.errors.append(message)
    if config.eval.every_steps is not None:
        report.warnings.append(
            "async: inline eval is not run on the async path; use `mg-rl eval`"
        )
    if async_config.eval_every_steps is not None and config.eval.checkpoint is not None:
        report.warnings.append(
            "async: eval.checkpoint is ignored by versioned async evaluation; the evaluator "
            "leases the currently published policy version"
        )

    orchestrator_args = {
        "keep_adapters",
        "export_adapter",
        "weight_version",
        "run_gen",
        "unload_grace_s",
        "publish_sink",
    }
    supplied_orchestrator_args = sorted(orchestrator_args & config.policy.args.keys())
    if supplied_orchestrator_args:
        report.errors.append(
            "async: policy.args must not set orchestrator-owned arguments: "
            f"{supplied_orchestrator_args}"
        )

    sampling = config.rollout.sampling
    if resolve_sampling_temperature(sampling) <= 0:
        report.errors.append("async: rollout.sampling temperature must be greater than 0")
    extra_body = sampling.get("extra_body")
    bodies = [("rollout.sampling", sampling)]
    if isinstance(extra_body, dict):
        bodies.append(("rollout.sampling.extra_body", extra_body))
    for prefix, body in bodies:
        if body.get("top_k") not in (None, 0, -1):
            report.errors.append(
                f"{prefix}.top_k is not supported on the async path because it cuts off "
                "behavior-policy support"
            )
        if body.get("top_p") not in (None, 1.0):
            report.errors.append(
                f"{prefix}.top_p must be absent or 1.0 on the async path because it cuts off "
                "behavior-policy support"
            )
        for key in ("min_p", "logit_bias", "bad_words", "allowed_token_ids"):
            if body.get(key):
                report.errors.append(
                    f"{prefix}.{key} is not supported on the async path because it cuts off "
                    "behavior-policy support"
                )
    if async_config.queue_capacity < config.rollout.groups_per_step:
        report.warnings.append(
            "async.queue_capacity is smaller than rollout.groups_per_step; the learner must "
            "wait across queue refills"
        )


def validate_config(config: RunConfig) -> ValidationReport:
    """Validate the passthrough sections of an already schema-valid ``RunConfig``."""
    report = ValidationReport()
    max_seq_length = config.model.max_seq_length
    max_tokens = config.rollout.sampling.get("max_tokens")
    if (
        max_seq_length is not None
        and isinstance(max_tokens, int)
        and max_seq_length <= max_tokens
        and config.trainer.sequence_overflow == "error"
    ):
        report.errors.append(
            f"model.max_seq_length ({max_seq_length}) must exceed rollout.sampling.max_tokens "
            f"({max_tokens}) plus the prompt length; rows longer than max_seq_length are "
            "refused (trainer.sequence_overflow)"
        )
    _validate_async(config, report)
    _validate_policy(config, report)
    _validate_trainer_args(config, report)
    _validate_environment(config, report)
    _validate_sampling(config, report)
    return report


def _annotation_types(annotation: Any) -> tuple[type, ...] | None:
    """Best-effort: concrete types acceptable for a dataclass field annotation. ``None`` when
    the annotation isn't a plain type or a simple ``X | None`` / ``Optional[X]`` union (in
    which case the caller should skip the type check rather than guess)."""
    import types
    import typing

    if isinstance(annotation, type):
        return (annotation,)
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if all(isinstance(a, type) for a in args):
            return tuple(args)
    return None


def grpo_config_field_report(args: dict[str, Any]) -> list[str]:
    """Flag ``trainer.args`` keys unknown to the installed ``trl.GRPOConfig`` (which also
    carries every inherited ``TrainingArguments`` field) or with an obvious type mismatch.

    Kept dependency-light: imports ``trl`` lazily, only when called, so importing
    ``config.validate`` never requires trl to be installed.
    """
    import dataclasses

    from trl.trainer.grpo_config import GRPOConfig  # trl re-exports this at the top level too,
    # but its stubs don't declare that, so pyright flags `trl.GRPOConfig` (reportPrivateImportUsage)

    fields = {f.name: f for f in dataclasses.fields(GRPOConfig)}
    problems: list[str] = []
    for key, value in args.items():
        f = fields.get(key)
        if f is None:
            problems.append(f"unknown GRPOConfig field: {key!r}")
            continue
        if value is None:
            continue
        acceptable = _annotation_types(f.type)
        if acceptable is None:
            continue
        # bool is an int subclass; only flag a real mismatch, not int/float/bool crossover.
        numeric = (int, float, bool)
        if isinstance(value, numeric) and any(issubclass(t, numeric) for t in acceptable):
            continue
        if not isinstance(value, acceptable):
            names = "|".join(t.__name__ for t in acceptable)
            problems.append(f"{key!r}: expected {names}, got {type(value).__name__}")
    return problems


__all__ = ["ValidationReport", "grpo_config_field_report", "validate_config"]
