"""Training / evaluation loop: config -> VerifiersRuntime -> RolloutGroup -> TrainerBackend.

The orchestrator owns sequencing, logging and checkpoints only. Verifiers owns rollouts;
the trainer backend owns model, loss and optimizer; the policy backend owns generation.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import random
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from moongazer_rl.config.loader import resolved_dict
from moongazer_rl.config.schema import RunConfig
from moongazer_rl.logging import lineage as lineage_mod
from moongazer_rl.logging.metrics import group_stats, throughput
from moongazer_rl.logging.records import append_jsonl, event_row
from moongazer_rl.logging.run_dir import RunLogger, runtime_info
from moongazer_rl.policy.base import PolicyBackend
from moongazer_rl.rollout.runtime import VerifiersRuntime
from moongazer_rl.rollout.schema import RolloutGroup
from moongazer_rl.trainer.base import TrainerBackend
from moongazer_rl.trainer.registry import resolve_backend

RendererFactory = Callable[[TrainerBackend, RunConfig], Any]


@dataclass
class TrainOutcome:
    run_dir: Path
    steps_completed: int
    last_step: int
    status: str


def _run_dir(config: RunConfig) -> Path:
    return Path(config.logging.output_dir or f"runs/{config.run.name}")


def _config_text(config: RunConfig) -> str:
    import yaml

    return yaml.safe_dump(resolved_dict(config), sort_keys=False, allow_unicode=True)


def _config_hash(config: RunConfig) -> str:
    return hashlib.sha256(_config_text(config).encode()).hexdigest()[:16]


def default_renderer_factory(backend: TrainerBackend, config: RunConfig) -> Any:
    """Build a ``renderers.Renderer`` from the trainer's tokenizer and ``model.renderer``."""
    tokenizer = getattr(backend, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError(
            f"trainer backend {backend.name!r} exposes no tokenizer; pass renderer= explicitly"
        )
    from pydantic import TypeAdapter
    from renderers import RendererConfig, create_renderer

    raw = dict(config.model.renderer or {})
    if raw.get("name") == "llm-jp-4":
        # Not a `renderers` package renderer: LLM-jp-4 speaks Harmony with its own tokenizer
        # (policy/llmjp4_renderer.py); chat_template_kwargs are merged like the default renderer's.
        from moongazer_rl.policy.llmjp4_renderer import build_llmjp4_renderer

        extra_body = dict((config.rollout.sampling or {}).get("extra_body") or {})
        raw.update(extra_body.get("chat_template_kwargs") or {})
        return build_llmjp4_renderer(tokenizer, raw)
    renderer_config = TypeAdapter(RendererConfig).validate_python(raw) if raw else None
    # Same contract as verifiers' TrainClient: `rollout.sampling.extra_body.chat_template_kwargs`
    # (e.g. enable_thinking) configures the renderer, not the engine.
    extra_body = dict((config.rollout.sampling or {}).get("extra_body") or {})
    chat_template_kwargs = extra_body.get("chat_template_kwargs") or None
    return create_renderer(tokenizer, renderer_config, chat_template_kwargs=chat_template_kwargs)


def _build_backend(config: RunConfig) -> TrainerBackend:
    backend_cls = resolve_backend(config.trainer.backend)
    backend = backend_cls(config)
    set_planned = getattr(backend, "set_planned_steps", None)
    if callable(set_planned):  # the LR-schedule horizon is part of what validate_args checks
        set_planned(config.rollout.max_steps)
    problems = backend.validate_args(dict(config.trainer.args))
    if problems:
        raise ValueError("invalid trainer.args:\n  " + "\n  ".join(problems))
    return backend


def _task_batches(
    tasks: Sequence[Any],
    *,
    groups_per_step: int,
    num_epochs: int | None,
    max_steps: int | None,
    seed: int,
) -> Iterable[tuple[int, list[Any]]]:
    """Deterministic (seeded) task order: shuffle per epoch, yield (step, tasks).
    ``num_epochs=None`` means "until max_steps"; both None is rejected."""
    if num_epochs is None and max_steps is None:
        raise ValueError("set rollout.num_epochs or rollout.max_steps")
    if not tasks:
        raise ValueError("taskset loaded no tasks")
    step = 0
    epochs: Iterable[int] = itertools.count() if num_epochs is None else range(num_epochs)
    for epoch in epochs:
        order = list(range(len(tasks)))
        random.Random(seed + epoch).shuffle(order)
        for start in range(0, len(order), groups_per_step):
            step += 1
            if max_steps is not None and step > max_steps:
                return
            yield step, [tasks[i] for i in order[start : start + groups_per_step]]


def _prune_checkpoints(directory: Path, keep_last: int | None) -> None:
    if not keep_last or not directory.exists():
        return
    steps = sorted(
        (p for p in directory.iterdir() if _is_checkpoint_dir(p)),
        key=lambda p: int(p.name.split("-", 1)[1]),
    )
    import shutil

    for old in steps[:-keep_last]:
        shutil.rmtree(old, ignore_errors=True)


_ORCH_STATE = "orchestrator_state.json"


def _write_orchestrator_state(target: Path, *, step: int, info: Any) -> None:
    """Record the batch position next to the backend's checkpoint so resume knows which
    batches were consumed by the weights it restores."""
    import json

    target.mkdir(parents=True, exist_ok=True)
    (target / _ORCH_STATE).write_text(
        json.dumps(
            {
                "step": step,
                "learner_global_step": info.global_step,
                "weight_version": info.weight_version,
            }
        )
    )


async def _checkpoint(
    backend: TrainerBackend, checkpoint_dir: Path, step: int, config: RunConfig, logger: RunLogger
) -> None:
    info = await asyncio.to_thread(_save_checkpoint_atomically, backend, checkpoint_dir, step)
    logger.checkpoint_saved(
        {
            "path": str(info.path),
            "step": step,
            "global_step": info.global_step,
            "weight_version": info.weight_version,
        }
    )
    _prune_checkpoints(checkpoint_dir, config.checkpoint.keep_last)


def _save_checkpoint_atomically(backend: TrainerBackend, checkpoint_dir: Path, step: int) -> Any:
    """Write into ``step-<n>.tmp`` and rename on completion, so a crash mid-save never leaves
    a half-written ``step-<n>`` that resume would pick as the latest checkpoint."""
    import shutil

    final = checkpoint_dir / f"step-{step}"
    tmp = checkpoint_dir / f"step-{step}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    if final.exists():
        shutil.rmtree(final)
    tmp.mkdir(parents=True, exist_ok=True)
    info = backend.save_checkpoint(tmp)
    _write_orchestrator_state(tmp, step=step, info=info)
    tmp.rename(final)
    info.path = final
    return info


def _read_orchestrator_state(checkpoint: Path) -> dict[str, Any]:
    import json

    state_path = checkpoint / _ORCH_STATE
    if state_path.exists():
        return dict(json.loads(state_path.read_text()))
    return {"step": int(checkpoint.name.split("-", 1)[1])}


def _read_orchestrator_step(checkpoint: Path) -> int:
    import json

    state_path = checkpoint / _ORCH_STATE
    if state_path.exists():
        return int(json.loads(state_path.read_text())["step"])
    return int(checkpoint.name.split("-", 1)[1])


async def _restore_for_resume(
    backend: TrainerBackend,
    run_dir: Path,
    checkpoint_dir: Path,
    *,
    logged_last_step: int,
    logger: RunLogger,
) -> int:
    """Restore the learner and return the batch position to resume after.

    The resume point is the latest checkpoint's batch position, never the log's last step:
    the log may run ahead of the checkpoint (steps trained but never saved) and those
    batches must be trained again since their updates were lost. A run that logged steps
    but has no checkpoint cannot be resumed; start a new run instead of silently skipping
    training that was never persisted."""
    latest = _latest_checkpoint(checkpoint_dir)
    if latest is None:
        if logged_last_step > 0:
            raise FileNotFoundError(
                f"{run_dir} has {logged_last_step} logged step(s) but no checkpoint under "
                f"{checkpoint_dir}; the trained weights are gone, so resume is impossible. "
                "Start a new run (and set checkpoint.save_steps)."
            )
        logger.event("resumed", from_step=0, checkpoint=None)
        return 0
    info = await asyncio.to_thread(backend.load_checkpoint, latest)
    state = _read_orchestrator_state(latest)
    resume_step = int(state["step"])
    recorded = state.get("learner_global_step")
    if recorded is not None and int(recorded) != int(info.global_step):
        # A backend that restores "its latest" rather than this directory would otherwise pair
        # newer weights with an older batch position.
        raise RuntimeError(
            f"checkpoint {latest} was written at learner step {recorded} but the backend "
            f"restored learner step {info.global_step}; refusing to resume with mismatched "
            "weights and batch position"
        )
    lineage_mod.record_resume(run_dir, from_step=resume_step, checkpoint=latest)
    if logged_last_step > resume_step:
        logger.event(
            "resume_rewind",
            logged_last_step=logged_last_step,
            checkpoint_step=resume_step,
            note="steps after the checkpoint were never persisted and will be trained again",
        )
    logger.event(
        "resumed",
        from_step=resume_step,
        checkpoint=str(latest),
        learner_global_step=info.global_step,
        weight_version=info.weight_version,
    )
    return resume_step


def _record_effective_config(run_dir: Path, config: RunConfig, logger: RunLogger) -> None:
    """On resume, make ``config.resolved.yaml`` the configuration actually in effect (overrides
    included) so the *next* resume continues from it; the previous file is kept as
    ``config.resolved.<UTC>.yaml`` for the record."""
    resolved_path = run_dir / "config.resolved.yaml"
    new_text = _config_text(config)
    if resolved_path.exists() and resolved_path.read_text(encoding="utf-8") == new_text:
        return
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    previous = run_dir / f"config.resolved.{stamp}.yaml"
    if resolved_path.exists():
        resolved_path.replace(previous)
    resolved_path.write_text(new_text, encoding="utf-8")
    logger.event("config_updated_on_resume", previous=previous.name if previous.exists() else None)


def _wandb_sink(config: RunConfig) -> Any | None:
    if not config.logging.wandb:
        return None
    from moongazer_rl.logging.wandb_sink import WandbSink

    sink = WandbSink(
        enabled=True,
        project="moongazer-rl",
        run_name=config.run.name,
        config=resolved_dict(config),
    )
    return sink if sink.enabled else None


def _flatten_step_row(row: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key in (
        "weight_version",
        "learner_global_step",
        "wall_time_s",
        "rollout_wall_time_s",
        "train_wall_time_s",
        "num_groups_used",
        "num_trajectories_used",
        "num_action_tokens",
    ):
        value = row.get(key)
        if isinstance(value, int | float):
            flat[key] = float(value)
    for prefix in ("metrics", "group_stats", "async"):
        for key, value in (row.get(prefix) or {}).items():
            if isinstance(value, int | float):
                flat[f"{prefix}/{key}"] = float(value)
    return flat


def _is_checkpoint_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("step-") and not path.name.endswith(".tmp")


def _latest_checkpoint(directory: Path) -> Path | None:
    if not directory.exists():
        return None
    steps = [p for p in directory.iterdir() if _is_checkpoint_dir(p)]
    if not steps:
        return None
    return max(steps, key=lambda p: int(p.name.split("-", 1)[1]))


async def _rollout_step(
    runtime: VerifiersRuntime,
    tasks: list[Any],
    *,
    num_generations: int,
    step: int,
) -> tuple[list[RolloutGroup], float]:
    started = time.perf_counter()
    groups = await runtime.rollout_groups(tasks, num_generations, learner_step=step)
    return groups, time.perf_counter() - started


async def _train_async(
    config: RunConfig,
    *,
    resume: bool,
    backend: TrainerBackend | None,
    renderer_factory: RendererFactory,
) -> TrainOutcome:
    run_dir = _run_dir(config)
    save_token_ids = bool(config.logging.save_token_ids)
    start_step = 0
    # Build (and validate) the backend before touching the run directory, so a bad
    # trainer.args or a missing dependency fails without leaving a half-created run behind.
    backend = backend or _build_backend(config)
    if resume:
        logger, state = RunLogger.open_for_resume(run_dir, save_token_ids=save_token_ids)
        start_step = state.last_step or 0
        _record_effective_config(run_dir, config, logger)
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"run directory {run_dir} is not empty; use resume")
        logger = RunLogger.create(
            run_dir,
            config_raw_text=_config_text(config),
            resolved=resolved_dict(config),
            runtime_info=runtime_info(),
            save_token_ids=save_token_ids,
        )
        lineage_mod.init_lineage(run_dir, config_hash=_config_hash(config), parent_run=None)

    status = "failed"
    steps_completed = 0
    last_step = start_step
    wandb = _wandb_sink(config)
    try:
        # The LR-schedule horizon must be known before setup(); only rollout.max_steps is
        # known here (epoch-based runs depend on the taskset size loaded later), so a
        # backend without a horizon keeps the LR constant rather than guessing.
        set_planned = getattr(backend, "set_planned_steps", None)
        if callable(set_planned):
            set_planned(config.rollout.max_steps)
        # Backend calls are synchronous and may spin their own event loop or load a
        # model for minutes; keep them off the orchestrator's loop like train_step.
        await asyncio.to_thread(backend.setup)
        checkpoint_dir = run_dir / "checkpoints"
        if resume:
            start_step = await _restore_for_resume(
                backend, run_dir, checkpoint_dir, logged_last_step=start_step, logger=logger
            )
            last_step = start_step
        fingerprint = getattr(backend, "tokenizer_fingerprint", None)
        if callable(fingerprint):
            logger.event("tokenizer_fingerprint", fingerprint=fingerprint())
        optimizer_info = getattr(backend, "optimizer_info", None)
        if callable(optimizer_info):
            logger.event("optimizer", **optimizer_info())

        policy: PolicyBackend = backend.policy()
        renderer = renderer_factory(backend, config)
        current_step = {"value": start_step, "eval": False}

        async def on_traces(group: RolloutGroup, records: list[dict]) -> None:
            if current_step["eval"]:
                # Evaluation rollouts never enter the training logs.
                _append_eval_records(run_dir, group, records, step=int(current_step["value"]))
                return
            logger.log_group(group, step=int(current_step["value"]), trace_records=records)

        runtime = VerifiersRuntime(
            environment=config.environment.to_env_config_dict(),
            sampling=dict(config.rollout.sampling),
            max_concurrent=config.rollout.max_concurrent,
            policy=policy,
            renderer=renderer,
            model_name=config.model.name,
            on_traces=on_traces,
        )
        last_checkpoint_step = start_step
        async with runtime:
            tasks = runtime.load_tasks()
            logger.event("tasks_loaded", num_tasks=len(tasks), taskset_id=runtime.taskset_id)
            batches = _task_batches(
                tasks,
                groups_per_step=config.rollout.groups_per_step,
                num_epochs=config.rollout.num_epochs,
                max_steps=config.rollout.max_steps,
                seed=config.run.seed,
            )
            for step, batch in batches:
                if step <= start_step:
                    continue
                current_step["value"] = step
                groups, rollout_s = await _rollout_step(
                    runtime, batch, num_generations=config.rollout.num_generations, step=step
                )
                train_started = time.perf_counter()
                result = await asyncio.to_thread(backend.train_step, groups)
                train_s = time.perf_counter() - train_started
                await policy.sync_weights(result.weight_version)
                row = logger.log_step(
                    result,
                    groups,
                    rollout_wall_time_s=rollout_s,
                    train_wall_time_s=train_s,
                    step=step,
                )
                if wandb is not None:
                    wandb.log(step, _flatten_step_row(row))
                steps_completed += 1
                last_step = step
                if config.checkpoint.save_steps and step % config.checkpoint.save_steps == 0:
                    await _checkpoint(backend, checkpoint_dir, step, config, logger)
                    last_checkpoint_step = step
                if config.eval.every_steps and step % config.eval.every_steps == 0:
                    current_step["eval"] = True
                    try:
                        stats = await _evaluate_inline(
                            config, runtime, run_dir, step=step, tasks=tasks
                        )
                    finally:
                        current_step["eval"] = False
                    logger.event("eval", step=step, **stats)
            # The last update must not die with backend.close(): save it unless the loop's
            # final step was already checkpointed (or checkpointing is disabled).
            if config.checkpoint.save_steps and last_step > last_checkpoint_step:
                await _checkpoint(backend, checkpoint_dir, last_step, config, logger)
        status = "completed"
    finally:
        try:
            backend.close()
        finally:
            logger.finish(status, steps_completed=steps_completed, last_step=last_step)
            if wandb is not None:
                wandb.finish()
    return TrainOutcome(
        run_dir=run_dir, steps_completed=steps_completed, last_step=last_step, status=status
    )


def _append_eval_records(
    run_dir: Path, group: RolloutGroup, records: list[dict], *, step: int
) -> None:
    from moongazer_rl.logging.records import rollout_row

    append_jsonl(
        run_dir / "eval_rollouts.jsonl",
        [rollout_row(group, t, step=step) for t in group.trajectories],
    )
    append_jsonl(run_dir / "eval_traces.jsonl", records)


async def _evaluate_inline(
    config: RunConfig, runtime: VerifiersRuntime, run_dir: Path, *, step: int, tasks: list[Any]
) -> dict[str, float]:
    """Evaluate the first ``eval.num_tasks`` tasks with current weights; append to eval.jsonl."""
    num = config.eval.num_tasks or len(tasks)
    started = time.perf_counter()
    groups = await runtime.rollout_groups(
        tasks[:num], config.rollout.num_generations, learner_step=step
    )
    elapsed = time.perf_counter() - started
    stats = group_stats(groups)
    stats.update(throughput(groups, elapsed))
    append_jsonl(run_dir / "eval.jsonl", [event_row("eval", step=step, **stats)])
    return stats


def train(
    config: RunConfig,
    *,
    resume: bool = False,
    backend: TrainerBackend | None = None,
    renderer_factory: RendererFactory = default_renderer_factory,
) -> Path:
    """Run training and return the run directory. ``backend``/``renderer_factory`` are injection
    points for tests; production callers pass only ``config``."""
    outcome = asyncio.run(
        _train_async(config, resume=resume, backend=backend, renderer_factory=renderer_factory)
    )
    return outcome.run_dir


async def _evaluate_async(
    config: RunConfig, *, backend: TrainerBackend | None, renderer_factory: RendererFactory
) -> dict[str, Any]:
    backend = backend or _build_backend(config)
    try:
        await asyncio.to_thread(backend.setup)
        restored = await _restore_for_eval(backend, config)
        policy = backend.policy()
        renderer = renderer_factory(backend, config)
        collected: list[tuple[RolloutGroup, list[dict]]] = []

        async def on_traces(group: RolloutGroup, records: list[dict]) -> None:
            collected.append((group, records))

        runtime = VerifiersRuntime(
            environment=config.environment.to_env_config_dict(),
            sampling=dict(config.rollout.sampling),
            max_concurrent=config.rollout.max_concurrent,
            policy=policy,
            renderer=renderer,
            model_name=config.model.name,
            on_traces=on_traces,
        )
        async with runtime:
            tasks = runtime.load_tasks()
            num = config.eval.num_tasks or len(tasks)
            started = time.perf_counter()
            groups = await runtime.rollout_groups(
                tasks[:num], config.rollout.num_generations, learner_step=0
            )
            elapsed = time.perf_counter() - started
        stats = group_stats(groups)
        stats.update(throughput(groups, elapsed))
        out_dir = _run_dir(config) / f"eval-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = [rec for _, records in collected for rec in records]
        append_jsonl(out_dir / "traces.jsonl", rows)
        from moongazer_rl.logging.records import rollout_row, write_json_atomic

        append_jsonl(
            out_dir / "rollouts.jsonl",
            [rollout_row(g, t, step=0) for g in groups for t in g.trajectories],
        )
        write_json_atomic(
            out_dir / "summary.json",
            {"stats": stats, "num_groups": len(groups), "checkpoint": restored},
        )
        return {"output_dir": str(out_dir), "checkpoint": restored, **stats}
    finally:
        backend.close()


async def _restore_for_eval(backend: TrainerBackend, config: RunConfig) -> dict[str, Any] | None:
    """Standalone eval restores the weights it is meant to evaluate (``eval.checkpoint``):
    a path, ``"latest"`` under the run's checkpoints, or ``None`` for the fresh model.
    A missing explicit path is an error; ``"latest"`` with no checkpoints evaluates the
    fresh model and says so in the returned record."""
    choice = config.eval.checkpoint
    if choice is None:
        return None
    if choice == "latest":
        latest = _latest_checkpoint(_run_dir(config) / "checkpoints")
        if latest is None:
            return {"path": None, "note": "no checkpoint found; evaluated the initial model"}
        target = latest
    else:
        target = Path(choice)
        if not target.is_dir():
            raise FileNotFoundError(f"eval.checkpoint {target} is not a directory")
    info = await asyncio.to_thread(backend.load_checkpoint, target)
    recorded = _read_orchestrator_state(target).get("learner_global_step")
    if recorded is not None and int(recorded) != int(info.global_step):
        # Same guard as resume: a backend that restores "its latest" must not report
        # those weights as the requested checkpoint's evaluation.
        raise RuntimeError(
            f"eval.checkpoint {target} was written at learner step {recorded} but the backend "
            f"restored learner step {info.global_step}; refusing to evaluate mismatched weights"
        )
    return {
        "path": str(target),
        "global_step": info.global_step,
        "weight_version": info.weight_version,
    }


def evaluate(
    config: RunConfig,
    *,
    backend: TrainerBackend | None = None,
    renderer_factory: RendererFactory = default_renderer_factory,
) -> dict[str, Any]:
    return asyncio.run(_evaluate_async(config, backend=backend, renderer_factory=renderer_factory))


__all__ = ["TrainOutcome", "default_renderer_factory", "evaluate", "train"]
