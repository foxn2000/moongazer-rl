"""Shim for ``trl==0.24.0`` ``GRPOTrainer`` and Unsloth 2026.9's patched copy of it.

The same subclass works against both bases because it only overrides methods whose *name
and signature* Unsloth preserves (Unsloth regenerates the trainer from TRL's source with
string replacements, see ``unsloth/models/rl_replacements.py``). Optional Gemma 4 memory
settings replace projection/scoring orchestration while retaining the upstream loss formula.

Overrides
---------
``_generate_single_turn``
    Returns our already-sampled ``(prompt_ids, completion_ids, sampling_logprobs, {})``.
    This is the *lowest* injection point: ``_generate`` still does the length metrics and
    ``num_items_in_batch``, and ``_generate_and_score_completions`` still does the padding,
    ``old_per_token_logps``, ``ref_per_token_logps``, rewards and advantages.
``_prepare_inputs``
    Slices the scored batch by the planner's micro-batches and trims each micro-batch to its
    own padding width (``trim_micro_batch_padding``): TRL pads to the widest row of the whole
    step, and on Gemma 4 the padded columns run through every GEMM and the non-band masks
    disable the banded sliding kernel.
``_generate``
    ``super()`` then replace ``num_items_in_batch`` with the batch's **action**-token total, so
    the DAPO normaliser matches the masked numerator.
``_generate_and_score_completions``
    ``super()`` then post-process: attach the action mask as ``tool_mask``, overwrite the
    advantages with the ones computed per canonical ``RolloutGroup``, and record the
    sampling-vs-learner statistics.
``_compute_loss``
    Only reached on the *plain TRL* path (Unsloth replaces ``compute_loss`` wholesale and
    reads ``tool_mask`` itself). Applies the action mask to the loss while keeping the model
    attention mask complete.

Post-init flags
---------------
``self.use_vllm = True`` and ``self.vllm_importance_sampling_correction = True`` are set
*after* ``__init__`` so no vLLM engine is created, but the two code paths they gate become
active: ``old_per_token_logps`` is always computed (the learner's logprobs at generation
time) and the loss is multiplied by the truncated importance ratio
``clamp(exp(old - sampling), max=vllm_importance_sampling_cap)``. That is exactly the
correction we need, because our samples come from a policy snapshot, not from the learner's
current forward pass.
"""

from __future__ import annotations

import dataclasses
import difflib
import functools
import inspect
import time
import warnings
from collections.abc import Sequence
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

    from moongazer_rl.trainer.unsloth_native.dataset import RolloutBatch
    from moongazer_rl.trainer.unsloth_native.memory import MemorySettings

from moongazer_rl.trainer.unsloth_native.objective import (
    AlgorithmSpec,
    global_denominator,
    loss_diagnostics,
    policy_gradient_loss,
    row_advantage_scale,
    zoo_loss_kwargs,
)

TRL_VERSIONS = ("0.24.0",)

# Keys the backend owns. "reject" -> a user value is an error; "warn" -> kept, with a warning.
CONTROLLED_ARGS: dict[str, tuple[str, str]] = {
    "num_generations": ("reject", "set from rollout.num_generations (the GRPO group size G)"),
    "num_iterations": ("reject", "must be 1: one optimizer step per train_step(groups)"),
    "gradient_accumulation_steps": (
        "reject",
        "must be 1: accumulate by sending more groups to one train_step instead",
    ),
    "steps_per_generation": (
        "reject",
        "must be 1: the shim supplies generations, not the dataloader",
    ),
    "generation_batch_size": (
        "reject",
        "derived from per_device_train_batch_size * num_generations",
    ),
    "use_vllm": ("reject", "generation is done by the rollout policy, never by the trainer"),
    "vllm_mode": ("reject", "generation is done by the rollout policy, never by the trainer"),
    "use_liger_loss": (
        "reject",
        "the Liger path bypasses _compute_loss, so the per-token action mask would be ignored",
    ),
    "remove_unused_columns": ("reject", "the shim bypasses the dataloader entirely"),
    "output_dir": ("reject", "set from logging.output_dir"),
    "report_to": ("reject", "the run logger owns metric reporting"),
    "seed": ("reject", "set from run.seed"),
    "max_completion_length": (
        "warn",
        "defaults to model.max_seq_length; it does not truncate our completions, it only "
        "normalises the dr_grpo loss",
    ),
    "temperature": (
        "match",
        "must equal rollout.sampling.temperature: _get_per_token_logps_and_entropies divides "
        "the learner logits by args.temperature, so a mismatch makes the importance ratio "
        "wrong even for a perfectly on-policy sample",
    ),
    "mask_truncated_completions": (
        "warn",
        "the shim re-implements it from ModelCall.finish_reason == 'length' (TRL's own test, "
        "'last token is not eos/pad', misfires for renderer stop tokens such as Gemma's <turn|>); "
        "note that Unsloth's GRPOConfig forces it on for loss_type=dapo",
    ),
}


HORIZON_FREE_SCHEDULERS = frozenset({"constant", "constant_with_warmup", "inverse_sqrt"})
"""``lr_scheduler_type`` values that do not need a total-step count. Everything else in
``transformers.SchedulerType`` (linear, cosine, polynomial, ...) decays to (near) zero at
``num_training_steps``, so it is only meaningful with a known horizon."""

DEFAULT_LR_SCHEDULER = "constant"
"""What the backend falls back to when no training horizon is known."""

METRIC_DRIVEN_SCHEDULERS = frozenset({"reduce_lr_on_plateau", "greedy"})
"""Schedulers whose ``step()`` takes a validation metric. ``Trainer`` never steps them in the
training loop — it guards with ``isinstance(..., (ReduceLROnPlateau, GreedyLR))``
(``transformers/trainer.py:1768``) and drives them from ``_maybe_log_save_evaluate``
(``:2987-2993``) instead. This backend replays the training-loop step and runs no evaluation
loop, so it can never feed them: our unconditional ``lr_scheduler.step()`` would raise
``TypeError: step() missing 1 required positional argument: 'metrics'``.

In practice ``TrainingArguments._validate_args`` (``training_args.py:1709``) already refuses
both unless ``eval_strategy`` is set, and setting it makes ``Trainer.__init__`` demand an
``eval_dataset`` we never pass — so today they fail with a confusing two-hop error raised deep
inside ``GRPOTrainer.__init__``, *after* the model has been loaded. Rejecting them in
``validate_args`` turns that into one clear message before anything is loaded, and keeps the
backend correct if transformers ever relaxes that guard."""


def scheduler_needs_horizon(lr_scheduler_type: Any) -> bool:
    """True when ``lr_scheduler_type`` decays over a fixed number of optimizer steps."""
    name = str(getattr(lr_scheduler_type, "value", lr_scheduler_type))
    return name not in HORIZON_FREE_SCHEDULERS and name not in METRIC_DRIVEN_SCHEDULERS


def validate_lr_schedule(args: dict[str, Any], *, planned_steps: int | None) -> list[str]:
    """Reject a decaying LR schedule whose horizon nobody knows.

    ``Trainer.create_scheduler(num_training_steps=N)`` bakes ``N`` into the schedule. Guessing
    it means the LR silently reaches 0 while the orchestrator keeps stepping, so a decaying
    schedule requires either ``trainer.args.max_steps`` or
    ``UnslothNativeBackend.set_planned_steps()``.
    """
    problems: list[str] = []
    requested_type = args.get("lr_scheduler_type")
    requested_name = str(getattr(requested_type, "value", requested_type))
    if requested_type is not None and requested_name in METRIC_DRIVEN_SCHEDULERS:
        problems.append(
            f"trainer.args.lr_scheduler_type={requested_type!r} is stepped with a validation "
            "metric, not with a step count. The unsloth-native backend replays Trainer's "
            "training-loop step and runs no evaluation loop, so it can never supply one "
            "(transformers also requires an eval_strategy plus an eval_dataset for it). Use a "
            f"step-count schedule: {sorted(HORIZON_FREE_SCHEDULERS)}, or a decaying one with a "
            "known horizon (max_steps / set_planned_steps)."
        )

    horizon = planned_steps if planned_steps and planned_steps > 0 else None
    if horizon is None:
        max_steps = args.get("max_steps")
        if isinstance(max_steps, int) and max_steps > 0:
            horizon = max_steps
    if horizon is not None:
        return problems

    if requested_type is not None and scheduler_needs_horizon(requested_type):
        problems.append(
            f"trainer.args.lr_scheduler_type={requested_type!r} decays over a fixed number of "
            "optimizer steps, but no horizon is known. Set trainer.args.max_steps, or have the "
            "orchestrator call backend.set_planned_steps(n) before setup(), or use one of "
            f"{sorted(HORIZON_FREE_SCHEDULERS)}."
        )
    warmup_ratio = args.get("warmup_ratio")
    if isinstance(warmup_ratio, (int, float)) and warmup_ratio > 0:
        problems.append(
            f"trainer.args.warmup_ratio={warmup_ratio!r} is a fraction of the total step count, "
            "which is unknown here. Use trainer.args.warmup_steps, or set max_steps / "
            "set_planned_steps(n)."
        )
    return problems


def config_field_names() -> set[str]:
    """Every settable ``GRPOConfig`` field (includes Unsloth's extra fields when patched)."""
    from moongazer_rl.trainer.unsloth_native.compat import (
        ensure_trl_importable,
        ensure_unsloth_first,
    )

    ensure_unsloth_first()  # unsloth before trl, always
    ensure_trl_importable()
    from trl import GRPOConfig  # pyright: ignore[reportPrivateImportUsage]

    return {f.name for f in dataclasses.fields(GRPOConfig) if f.init}


def validate_args(args: dict[str, Any], *, temperature: float | None = None) -> list[str]:
    """Human-readable problems with ``trainer.args``. No GPU, no model load.

    ``temperature`` is the rollout sampling temperature the backend will force; when given, a
    conflicting ``trainer.args.temperature`` is an error and an equal one is accepted.
    """
    problems: list[str] = []
    try:
        known = config_field_names()
    except Exception as exc:  # pragma: no cover - trl import failure is reported elsewhere
        return [f"could not import trl.GRPOConfig: {exc}"]

    unknown = sorted(set(args) - known)
    for key in unknown:
        close = difflib.get_close_matches(key, sorted(known), n=3, cutoff=0.6)
        hint = f" (did you mean: {', '.join(close)}?)" if close else ""
        problems.append(f"unknown trainer.args key {key!r}{hint}")

    for key, (policy, why) in CONTROLLED_ARGS.items():
        if key in args and policy == "reject":
            problems.append(
                f"trainer.args.{key} is controlled by the unsloth-native backend: {why}"
            )

    if temperature is not None and "temperature" in args:
        given = args["temperature"]
        if not isinstance(given, (int, float)) or abs(float(given) - float(temperature)) > 1e-9:
            problems.append(
                f"trainer.args.temperature={given!r} conflicts with "
                f"rollout.sampling.temperature={temperature!r}: "
                f"{CONTROLLED_ARGS['temperature'][1]}. Drop trainer.args.temperature (it is "
                "taken from the rollout config) or make the two equal."
            )

    loss_type = args.get("loss_type")
    if loss_type is not None and loss_type not in ("grpo", "bnpo", "dr_grpo", "dapo"):
        problems.append(
            f"trainer.args.loss_type={loss_type!r} is not one of grpo/bnpo/dr_grpo/dapo"
        )
    level = args.get("importance_sampling_level")
    if level is not None and level not in ("token", "sequence"):
        problems.append(
            f"trainer.args.importance_sampling_level={level!r} is not one of token/sequence"
        )
    scale = args.get("scale_rewards")
    if scale is not None and scale not in ("group", "batch", "none", True, False):
        problems.append(f"trainer.args.scale_rewards={scale!r} is not one of group/batch/none")
    return problems


def build_grpo_config(
    args: dict[str, Any],
    *,
    num_generations: int,
    output_dir: str,
    seed: int,
    max_completion_length: int,
    temperature: float,
    overrides: dict[str, Any] | None = None,
    spec: AlgorithmSpec | None = None,
) -> Any:
    """``GRPOConfig(**args)`` with the keys the backend owns forced to safe values.

    ``overrides`` are defaults the backend derives but the user may still set (currently only
    ``lr_scheduler_type``): they are applied *under* the user's ``args``.

    Raises ``ValueError`` listing every problem at once (unknown keys, rejected overrides).
    """
    problems = validate_args(args, temperature=temperature)
    if problems:
        raise ValueError(
            "invalid trainer.args for the unsloth-native backend:\n  - " + "\n  - ".join(problems)
        )

    from moongazer_rl.trainer.unsloth_native.compat import (
        ensure_trl_importable,
        ensure_unsloth_first,
    )

    ensure_unsloth_first()  # unsloth before trl, always
    ensure_trl_importable()
    from trl import GRPOConfig  # pyright: ignore[reportPrivateImportUsage]

    if num_generations < 2 and (
        spec is None or spec.advantage_estimator in {"group_zscore", "group_center", "rloo"}
    ):
        raise ValueError(f"rollout.num_generations must be >= 2 for GRPO (got {num_generations})")

    resolved = dict(overrides or {})
    resolved.update(args)
    for key, (policy, why) in CONTROLLED_ARGS.items():
        if policy == "warn" and key in resolved:
            warnings.warn(
                f"trainer.args.{key}={resolved[key]!r} overrides a value the unsloth-native "
                f"backend normally controls: {why}",
                UserWarning,
                stacklevel=2,
            )
    resolved.setdefault("max_completion_length", int(max_completion_length))
    # COMPAT(unsloth==2026.9): Unsloth's regenerated GRPOConfig defaults warmup_steps to 0.1
    # (a ratio, so 1 warmup step for any short horizon) where TRL's default is 0. With our
    # one-optimizer-step-per-train_step loop that made the first real update run at LR 0.
    # Keep TRL's default unless
    # the caller asks for warmup explicitly.
    if "warmup_ratio" not in resolved:
        resolved.setdefault("warmup_steps", 0)

    per_device = int(resolved.get("per_device_train_batch_size", num_generations))
    resolved.update(
        temperature=float(temperature),
        num_generations=max(2, int(num_generations)),
        num_iterations=1,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=per_device,
        # COMPAT(trl==0.24): GRPOConfig.__post_init__ (grpo_config.py:648-681) rejects
        # generation_batch_size and steps_per_generation together and demands
        # generation_batch_size % num_generations == 0. Passing the product satisfies both for
        # any per_device_train_batch_size; steps_per_generation is corrected to 1 below.
        generation_batch_size=per_device * max(2, int(num_generations)),
        use_vllm=False,
        use_liger_loss=False,
        remove_unused_columns=False,
        output_dir=str(output_dir),
        report_to=[],
        seed=int(seed),
    )
    resolved.pop("steps_per_generation", None)

    if spec is not None:
        for key, value in zoo_loss_kwargs(spec).items():
            if key == "epsilon_low":
                resolved["epsilon"] = value
            elif key in config_field_names():
                resolved[key] = value
        resolved["beta"] = spec.beta
        resolved["delta"] = spec.delta
        resolved["scale_rewards"] = (
            spec.std_scope if spec.advantage_estimator == "group_zscore" else "none"
        )
    config = GRPOConfig(**resolved)
    if spec is not None:
        constructed_high = config.epsilon_high
        expected_high = zoo_loss_kwargs(spec)["epsilon_high"]
        if constructed_high != expected_high:
            warnings.warn(
                f"GRPOConfig rewrote epsilon_high from {expected_high} to {constructed_high}; "
                "restoring the resolved algorithm value",
                UserWarning,
                stacklevel=2,
            )
        config.mask_truncated_completions = spec.mask_truncated_completions
        for key, value in zoo_loss_kwargs(spec).items():
            setattr(config, key, value)
        config.epsilon = spec.epsilon_low
        config.epsilon_high = zoo_loss_kwargs(spec)["epsilon_high"]
    # COMPAT(trl==0.24): __post_init__ derived steps_per_generation = num_generations from the
    # generation_batch_size above. _prepare_inputs (grpo_trainer.py:1002) only regenerates when
    # `_step % (steps_per_generation * num_iterations) == 0`, so it must be 1 or the second
    # train_step would silently reuse the first step's buffered batch. Remove when TRL gains a
    # per-call generation hook (rollout_func).
    config.steps_per_generation = 1
    config.num_generations = int(num_generations)
    return config


_PROMPT_ALIGNED_KEYS = ("prompt_ids", "prompt_mask")
_LEARNER_WINDOW_KEYS = ("old_per_token_logps", "ref_per_token_logps")


def trim_micro_batch_padding(
    micro: dict[str, Any], *, pad_token_id: int | None = None, eos_token_id: Any = None
) -> tuple[dict[str, Any], int]:
    """Drop the padding columns a micro-batch does not need.

    TRL's ``_generate_and_score_completions`` left-pads every prompt and right-pads every
    completion to the longest row of the *whole* generation batch (grpo_trainer.py:1379-1387),
    so a micro-batch of one 4k-token row is forwarded at the width of the step's 16k-token
    row. Padding is not free here: the padded columns take part in every GEMM and attention
    call, and the resulting non-band masks disable Unsloth's banded sliding-window kernel and
    SDPA's ``is_causal`` fast path for Gemma 4. Trimming is exact: only columns that are
    padding in every row of the micro-batch are removed, and every completion-aligned tensor
    (masks, old/sampling/ref logprobs, importance weights, ``tool_mask``) is cut at the same
    column.

    ``old_per_token_logps`` / ``ref_per_token_logps`` may arrive in Unsloth's learner window
    ``[B, C + max_left_pad]`` (row ``i``'s completion starts at column ``max_left_pad -
    left_pad_i``; unsloth_zoo derives the window from that width). They are moved to completion
    coordinates, trimmed like every other completion-aligned tensor, and re-windowed against the
    trimmed prompt mask, so the zoo sees a window consistent with the micro-batch it forwards.

    Returns the trimmed dict and the number of padding tokens removed (rows x columns).
    """
    import torch

    prompt_mask = micro.get("prompt_mask")
    completion_mask = micro.get("completion_mask")
    completion_ids = micro.get("completion_ids")
    if not (
        isinstance(prompt_mask, torch.Tensor)
        and isinstance(completion_mask, torch.Tensor)
        and prompt_mask.ndim == 2
        and completion_mask.ndim == 2
        and prompt_mask.shape[0] == completion_mask.shape[0]
    ):
        return micro, 0
    rows, prompt_width = prompt_mask.shape
    completion_width = completion_mask.shape[1]
    live_prompt = prompt_mask.to(torch.bool).any(dim=0)
    live_completion = completion_mask.to(torch.bool).any(dim=0)
    # A truncated row can have an all-zero completion_mask while its tokens are real; keep
    # every column that still carries a non-pad token (Unsloth rebuilds completion_mask from
    # pad ids, so those columns must survive). When pad == eos the ids cannot tell padding from
    # the final eos, so rely on the mask alone.
    eos = set(eos_token_id) if isinstance(eos_token_id, (list, tuple, set)) else {eos_token_id}
    if (
        pad_token_id is not None
        and pad_token_id not in eos
        and isinstance(completion_ids, torch.Tensor)
        and completion_ids.shape == completion_mask.shape
    ):
        live_completion |= (completion_ids != pad_token_id).any(dim=0)
    lead = int(live_prompt.int().argmax()) if bool(live_prompt.any()) else prompt_width - 1
    if bool(live_completion.any()):
        keep_completion = completion_width - int(live_completion.flip(0).int().argmax())
    else:
        keep_completion = 1
    if lead == 0 and keep_completion == completion_width:
        return micro, 0
    total_width = prompt_width + completion_width
    trimmed_prompt_mask = prompt_mask[:, lead:]
    new_max_left_pad = int((trimmed_prompt_mask.to(torch.long) == 0).sum(dim=1).max())
    trimmed: dict[str, Any] = {}
    for key, value in micro.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[0] == rows:
            width = value.shape[1]
            # Learner-window logprobs are never prompt+completion wide; any excess is max_left_pad.
            windowed = key in _LEARNER_WINDOW_KEYS and width > completion_width
            if windowed:
                value = _from_learner_window(value, prompt_mask, completion_width)
                width = completion_width
            if key in _PROMPT_ALIGNED_KEYS or (key.startswith("prompt") and width == prompt_width):
                value = value[:, lead:]
            elif width == total_width:
                value = value[:, lead : prompt_width + keep_completion]
            elif width == completion_width:
                value = value[:, :keep_completion]
            if windowed:
                value = align_to_learner_window(
                    value, trimmed_prompt_mask, keep_completion + new_max_left_pad
                )
        trimmed[key] = value
    removed = rows * (lead + (completion_width - keep_completion))
    return trimmed, removed


def _from_learner_window(wide: torch.Tensor, prompt_mask: torch.Tensor, completion_len: int):
    """Inverse of :func:`align_to_learner_window`: gather row ``i``'s completion columns
    ``[max_left_pad - left_pad_i, ... + completion_len)`` out of a ``[B, C + max_left_pad]``
    learner-window tensor into ``[B, C]`` completion coordinates."""
    import torch

    batch, width = wide.shape
    extra = width - completion_len
    left_pad = (prompt_mask.to(torch.long) == 0).sum(dim=1)
    offsets = (extra - left_pad).clamp(min=0)
    cols = torch.arange(completion_len, device=wide.device).unsqueeze(0) + offsets.unsqueeze(1)
    rows = torch.arange(batch, device=wide.device).unsqueeze(1).expand_as(cols)
    valid = cols < width
    narrow = torch.zeros((batch, completion_len), dtype=wide.dtype, device=wide.device)
    narrow[valid] = wide[rows[valid], cols[valid]]
    return narrow


def make_trainer_class(base_cls: type) -> type:
    """Build the shimmed trainer subclass for ``base_cls``.

    ``base_cls`` is ``trl.GRPOTrainer`` — which is Unsloth's ``UnslothGRPOTrainer`` once
    ``PatchFastRL("GRPO")`` has run (unsloth/models/rl.py:3387 rebinds ``trl.GRPOTrainer``),
    so import it *after* patching.
    """
    import torch

    class UnslothNativeGRPOTrainer(base_cls):  # type: ignore[misc,valid-type]
        """GRPOTrainer whose generation is replaced by canonical rollouts."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.memory: MemorySettings | None = None
            self._arl_batch: RolloutBatch | None = None
            self._arl_stats: dict[str, float] = {}
            self._arl_spec: AlgorithmSpec | None = None
            self._arl_loss_stats = []
            self._arl_plan = None
            self._arl_current_rows = None
            self._arl_off_policy_by_staleness = {}
            self._arl_off_policy_by_staleness_partial = False
            # COMPAT(trl==0.24, unsloth==2026.9): flipped after __init__ so no vLLM engine is
            # built (grpo_trainer.py:485-557 only inspects args.use_vllm) while the two paths
            # they gate turn on: old_per_token_logps (grpo_trainer.py:1420) and the truncated
            # importance-sampling correction (grpo_trainer.py:1437 and 1724; Unsloth reads
            # inputs["sampling_per_token_logps"] instead, unsloth_zoo/rl_replacements.py:835).
            # Remove when TRL exposes sampling logprobs without the vLLM flag.
            self.use_vllm = True
            self.vllm_importance_sampling_correction = True
            self._last_loaded_step = -1
            # COMPAT(transformers==5.5): set by Trainer._inner_training_loop (trainer.py:1705),
            # which we never enter; _compute_loss divides by it.
            self.current_gradient_accumulation_steps = 1

        def set_memory_settings(self, settings: MemorySettings | None) -> None:
            """Install after trainer construction; None restores the original bound method."""
            from moongazer_rl.trainer.unsloth_native.memory import (
                has_unsloth_loss,
                is_gemma4_peft,
                make_memory_efficient_compute_loss,
                resolve_lm_head,
            )

            if not hasattr(self, "_arl_original_compute_loss"):
                self._arl_original_compute_loss = self.compute_loss
            replacement = self._arl_original_compute_loss
            active = (
                settings is not None
                and settings.token_chunked_logprobs
                and has_unsloth_loss(self)
                and is_gemma4_peft(self.model)
            )
            if active and settings is not None:
                if settings.token_chunked_logprobs or settings.score_rows_one_at_a_time:
                    # Trimming identifies padding by ID; EOS must remain a real token.
                    pad = self.processing_class.pad_token_id
                    eos = getattr(self.processing_class, "eos_token_id", None)
                    if pad is None or pad == eos or (isinstance(eos, (list, tuple)) and pad in eos):
                        raise ValueError(
                            "memory scoring needs a dedicated pad token; disable trainer.memory "
                            "or set a dedicated pad token distinct from eos_token_id"
                        )
                    resolve_lm_head(self.model)
                if settings.token_chunked_logprobs:
                    replacement = make_memory_efficient_compute_loss(self)
            self.compute_loss = replacement
            # A later begin_step must wrap the newly selected memory/original path.
            self._arl_algorithm_base_compute_loss = replacement
            self._arl_memory_unsloth = active
            self.memory = settings

        def _get_per_token_logps_and_entropies(
            self,
            model: Any,
            input_ids: Any,
            attention_mask: Any,
            logits_to_keep: int,
            batch_size: Any = None,
            compute_entropy: bool = False,
            *args: Any,
            **kwargs: Any,
        ):
            # U/models/rl_replacements.py:1470-1471 is the efficient-loss signal;
            # :2624-2626 also passes it POSITIONALLY after compute_entropy.
            if self.memory is not None:
                if kwargs.get("compute_efficient", False) or (args and args[0] is True):
                    return None, None
                from moongazer_rl.trainer.unsloth_native.memory import (
                    is_gemma4_peft,
                    score_per_token_logps,
                )

                if (
                    getattr(self, "_arl_memory_unsloth", False)
                    and self.memory.score_rows_one_at_a_time
                    and is_gemma4_peft(model)
                ):
                    for key in (
                        "pixel_values",
                        "pixel_values_videos",
                        "input_features",
                        "pixel_attention_mask",
                        "image_sizes",
                        "num_images",
                        "image_grid_thw",
                        "token_type_ids",
                        "mm_token_type_ids",
                    ):
                        if kwargs.get(key) is not None:
                            raise NotImplementedError(
                                "memory GRPO scoring supports text-only inputs"
                            )
                    return score_per_token_logps(
                        self,
                        model,
                        input_ids,
                        attention_mask,
                        logits_to_keep,
                        chunk_size=self.memory.logprob_chunk_size,
                    ), None
            return super()._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                batch_size,
                compute_entropy,
                *args,
                **kwargs,
            )

        # -- batch plumbing --------------------------------------------------------
        def begin_step(
            self,
            batch: RolloutBatch,
            *,
            plan: list[list[int]] | None = None,
            spec: AlgorithmSpec | None = None,
            denominator: float | None = None,
            num_chunks: int | None = None,
            active_rows: Sequence[int] | None = None,
        ) -> None:
            """Score once, then slice the row plan without shuffle or tail loss.

            num_chunks remains for callers of the previous equal-chunk interface; with
            ``active_rows`` the equal chunks cover only those rows (zero-loss rows are skipped).
            """
            self.set_rollout_batch(batch)
            if spec is not None:
                self._arl_spec = spec
            spec = self._arl_spec
            self._arl_legacy_chunks = plan is None
            expected_rows: list[int] | None = None
            if plan is None:
                rows = list(active_rows) if active_rows is not None else list(range(batch.size))
                expected_rows = rows
                chunks = num_chunks or 1
                if chunks < 1 or chunks > len(rows):
                    raise ValueError("num_chunks must be between 1 and the number of active rows")
                width = max(1, len(rows) // chunks)
                plan = [rows[i * width : min((i + 1) * width, len(rows))] for i in range(chunks)]
            flat = [i for chunk in plan for i in chunk]
            if (
                not plan
                or any(not chunk for chunk in plan)
                or len(set(flat)) != len(flat)
                or any(i < 0 or i >= batch.size for i in flat)
                or (expected_rows is not None and sorted(flat) != sorted(expected_rows))
            ):
                raise ValueError(
                    "plan must contain every row requested exactly once, with nonempty chunks"
                )
            # Rows absent from the plan are scored with the batch (advantages, normaliser,
            # metrics) but never forwarded for a gradient: the backend leaves out rows whose
            # loss is identically zero (truncated rows under mask_truncated_completions).
            self._arl_plan = plan
            self._arl_denominator = (
                denominator
                if denominator is not None
                else (
                    global_denominator(spec, batch.rows)
                    if spec is not None
                    else batch.num_action_tokens
                )
            )
            self.args.steps_per_generation = len(plan)
            self.current_gradient_accumulation_steps = len(plan)
            self._step = 0
            self._buffered_inputs = None
            self._arl_loss_stats = []
            self._arl_current_rows = None
            self._arl_row_staleness = None
            self._arl_off_policy_by_staleness = {}
            self._arl_off_policy_by_staleness_partial = False
            if spec is not None:
                self.mask_truncated_completions = spec.mask_truncated_completions
                self.args.mask_truncated_completions = spec.mask_truncated_completions
                for key, value in zoo_loss_kwargs(spec).items():
                    setattr(self.args, key, value)
                self.use_vllm = not spec.skip_old_logprob_forward
                self.vllm_importance_sampling_correction = spec.behavior_correction != "none"
                # With correction disabled plain TRL's scoring gate still needs a forced old
                # pass when skip=False. This temporary scoring gate is restored afterwards.
                from moongazer_rl.trainer.unsloth_native.memory import has_unsloth_loss

                if has_unsloth_loss(self):
                    if not hasattr(self, "_arl_algorithm_base_compute_loss"):
                        self._arl_algorithm_base_compute_loss = self.compute_loss
                    self.compute_loss = _algorithm_compute_loss(
                        self, self._arl_algorithm_base_compute_loss, spec
                    )

        def _prepare_inputs(self, inputs):
            if self._arl_plan is None:
                return super()._prepare_inputs(inputs)
            if self._arl_legacy_chunks and self._step >= len(self._arl_plan):
                # Previous num_chunks callers may request another scoring-only cycle.
                self._buffered_inputs = None
                self._step = 0
            if self._buffered_inputs is None:
                started = time.perf_counter()
                with torch.no_grad(), nullcontext():
                    scored = self._generate_and_score_completions(inputs)
                self._arl_stats["time/score_forward_s"] = time.perf_counter() - started
                multimodal = {
                    "pixel_values",
                    "image_grid_thw",
                    "num_images",
                    "pixel_attention_mask",
                    "image_sizes",
                }.intersection(scored)
                if multimodal:
                    raise NotImplementedError(
                        f"micro-batch slicing supports text-only inputs: {sorted(multimodal)}"
                    )
                sliced = [
                    {
                        key: value[indices]
                        if isinstance(value, torch.Tensor) and value.ndim > 0
                        else [value[i] for i in indices]
                        if isinstance(value, list)
                        else value
                        for key, value in scored.items()
                    }
                    for indices in self._arl_plan
                ]
                # Each micro-batch is forwarded at its own width, not the step's widest row.
                tokenizer = getattr(self, "processing_class", None)
                pad_id = getattr(tokenizer, "pad_token_id", None)
                eos_id = getattr(tokenizer, "eos_token_id", None)
                removed = 0
                forwarded_padded = 0
                forward_padding_available = True
                self._buffered_inputs = []
                for micro in sliced:
                    micro, n = trim_micro_batch_padding(
                        micro, pad_token_id=pad_id, eos_token_id=eos_id
                    )
                    removed += n
                    if all(key in micro for key in ("prompt_ids", "completion_ids")):
                        forwarded_padded += micro["prompt_ids"].shape[0] * (
                            micro["prompt_ids"].shape[1] + micro["completion_ids"].shape[1]
                        )
                    else:
                        forward_padding_available = False
                    self._buffered_inputs.append(micro)
                self._arl_stats["padding_tokens_trimmed"] = float(removed)
                if forward_padding_available:
                    self._arl_stats["num_forward_padded_tokens"] = float(forwarded_padded)
            result = self._buffered_inputs[self._step]
            limit = getattr(self, "_arl_gc_auto_max_tokens", None)
            if limit is not None and all(k in result for k in ("prompt_ids", "completion_ids")):
                padded = result["prompt_ids"].shape[0] * (
                    result["prompt_ids"].shape[1] + result["completion_ids"].shape[1]
                )
                from moongazer_rl.trainer.unsloth_native.backend import (
                    set_gradient_checkpointing,
                )

                set_gradient_checkpointing(self.model, padded > limit)
                if padded <= limit:
                    self._arl_stats["gc_off_micro_batches"] = (
                        self._arl_stats.get("gc_off_micro_batches", 0) + 1
                    )
            self._arl_current_rows = tuple(self._arl_plan[self._step])
            self._step += 1
            return result

        def set_rollout_batch(self, batch: RolloutBatch) -> None:
            """Install the rollouts the next ``training_step`` must consume."""
            self._arl_batch = batch
            self._arl_stats = {}
            # COMPAT(trl==0.24): grpo_trainer.py:1489 reshapes rewards to
            # (-1, num_generations), so this must divide the row count. Our advantages replace
            # TRL's anyway (see _generate_and_score_completions); when the batch is uniform we
            # keep the real G so TRL's own advantages stay a free cross-check.
            self.num_generations = batch.rows_per_group or max(1, batch.size)

        def _require_batch(self, n_rows: int) -> RolloutBatch:
            batch = self._arl_batch
            if batch is None:
                raise RuntimeError(
                    "UnslothNativeGRPOTrainer.set_rollout_batch() must be called before every "
                    "training_step; the trainer never generates on its own."
                )
            if len(batch.rows) != n_rows:
                raise RuntimeError(
                    f"rollout batch has {len(batch.rows)} rows but TRL asked for {n_rows}"
                )
            return batch

        # -- generation ------------------------------------------------------------
        def _generate_single_turn(
            self, prompts: Any, images: Any = None, *args: Any, **kwargs: Any
        ):
            # COMPAT(trl==0.24): grpo_trainer.py:1310 returns
            # (prompt_ids, completion_ids, logprobs, forward_kwargs) as plain python lists.
            batch = self._require_batch(len(prompts))
            prompt_ids = [list(row.prompt_ids) for row in batch.rows]
            completion_ids = [list(row.completion_ids) for row in batch.rows]
            logprobs = [list(row.sampling_logprobs) for row in batch.rows]
            return prompt_ids, completion_ids, logprobs, {}

        def _generate(self, prompts: Any, images: Any = None, *args: Any, **kwargs: Any):
            (
                prompt_ids,
                completion_ids,
                num_items_in_batch,
                logprobs,
                forward_kwargs,
            ) = super()._generate(prompts, images, *args, **kwargs)
            # COMPAT(trl==0.24): _generate returns num_items_in_batch as
            # accelerator.gather(completion_lengths).sum() — every completion token
            # (grpo_trainer.py:1317-1322, :1347). It is the DAPO normaliser
            # (grpo_trainer.py:1744-1745), while the numerator is masked by the action mask, so
            # tool/user tokens would shrink the gradient of multi-turn rows. Replace it with the
            # gathered ACTION-token total, same dtype/device and same gather, so every
            # micro-batch divides by the total the numerator can actually reach.
            batch = self._require_batch(len(prompts))
            mask_truncated = self._mask_truncated_requested()
            action_counts = torch.tensor(
                [
                    0 if (mask_truncated and row.truncated) else row.num_action_tokens
                    for row in batch.rows
                ],
                device=self.accelerator.device,
                dtype=torch.long,
            )
            total_action_tokens = self.accelerator.gather(action_counts).sum().clamp(min=1)
            # COMPAT(trl==0.24): completions/clipped_ratio (grpo_trainer.py:1337-1338) calls a
            # completion truncated when its last token is not the tokenizer eos/pad. Renderer
            # stop tokens (Gemma's <turn|>, ...) are not the tokenizer eos, so that test is
            # wrong for them; report the renderer-aware finish_reason instead.
            truncated_flags = torch.tensor(
                [1.0 if row.truncated else 0.0 for row in batch.rows],
                device=self.accelerator.device,
            )
            clipped = self._metrics["train"].get("completions/clipped_ratio")
            if clipped:
                clipped[-1] = float(self.accelerator.gather(truncated_flags).mean())
            if self._arl_spec is not None:
                total_action_tokens = torch.tensor(
                    self._arl_denominator, device=self.accelerator.device
                )
                # Preserve renderer-aware truncation normalization for legacy DAPO.
                if self._arl_spec.loss_normalization == "action_token" and mask_truncated:
                    total_action_tokens = self.accelerator.gather(action_counts).sum().clamp(min=1)
            self._arl_stats["num_items_in_batch"] = float(total_action_tokens)
            self._arl_stats["num_completion_tokens"] = float(num_items_in_batch)
            return prompt_ids, completion_ids, total_action_tokens, logprobs, forward_kwargs

        def _mask_truncated_requested(self) -> bool:
            """The configured flag, also while ``_generate_and_score_completions`` has switched
            TRL's own copy off around ``super()`` (``_generate`` runs inside that window)."""
            pinned = getattr(self, "_arl_mask_truncated_requested", None)
            if pinned is not None:
                return bool(pinned)
            return bool(getattr(self, "mask_truncated_completions", False))

        def _generate_and_score_completions(self, inputs: Any) -> dict[str, Any]:
            batch = self._require_batch(len(inputs))
            # COMPAT(trl==0.24 / unsloth==2026.9): TRL implements mask_truncated_completions as
            # "last completion token not in {eos, pad}" (grpo_trainer.py:1391-1394), and
            # Unsloth's GRPOConfig forces the flag on for loss_type=dapo. With a renderer whose
            # stop token is not the tokenizer eos (Gemma 4: <turn|> vs <eos>) that test marks
            # EVERY normally finished completion as truncated and the loss is identically zero.
            # Run super() with the flag off and apply the renderer-aware TrainRow.truncated
            # flag ourselves (below, via tool_mask, which both loss paths honour).
            mask_truncated = self._mask_truncated_requested()
            self._arl_mask_truncated_requested = mask_truncated
            self.mask_truncated_completions = False
            spec = self._arl_spec
            if spec is not None:
                self.use_vllm = not spec.skip_old_logprob_forward
                self.vllm_importance_sampling_correction = True
                # The upstream gate uses args.GAS % generate_every; both must agree.
                previous_gas = self.args.gradient_accumulation_steps
                self.args.gradient_accumulation_steps = self.args.steps_per_generation
            try:
                out = super()._generate_and_score_completions(inputs)
            finally:
                self.mask_truncated_completions = mask_truncated
                self._arl_mask_truncated_requested = None
                if spec is not None:
                    self.args.gradient_accumulation_steps = previous_gas
                    self.use_vllm = spec.behavior_correction != "none"
                    self.vllm_importance_sampling_correction = spec.behavior_correction != "none"
            device = out["completion_ids"].device
            width = out["completion_ids"].shape[1]

            tool_mask = torch.zeros((len(batch.rows), width), dtype=torch.long)
            sampling_logps = torch.zeros((len(batch.rows), width), dtype=torch.float32)
            num_masked_truncated = 0
            for i, row in enumerate(batch.rows):
                n = len(row.completion_ids)
                if mask_truncated and row.truncated:
                    num_masked_truncated += 1  # tool_mask row stays all-zero: no loss
                else:
                    tool_mask[i, :n] = torch.tensor(
                        [1 if flag else 0 for flag in row.action_mask],
                        dtype=torch.long,
                    )
                sampling_logps[i, :n] = torch.tensor(
                    row.sampling_logprobs, dtype=torch.float32
                )
            tool_mask = tool_mask.to(device, non_blocking=True)
            sampling_logps = sampling_logps.to(device, non_blocking=True)
            self._arl_stats["num_truncated_rows"] = float(batch.num_truncated_rows)
            self._arl_stats["num_truncated_rows_masked"] = float(num_masked_truncated)

            # COMPAT(unsloth==2026.9): Unsloth's patched compute_loss reads inputs["tool_mask"]
            # (unsloth/models/rl_replacements.py:2580, 2662-2667) and its grpo_accumulated_loss
            # rebuilds completion_mask from pad tokens and then multiplies by tool_mask
            # (unsloth_zoo/rl_replacements.py:927-933) — so a modified completion_mask would be
            # thrown away, while tool_mask survives. Our own _compute_loss uses the same key.
            out["tool_mask"] = tool_mask
            # COMPAT(unsloth==2026.9): the patched _generate_and_score_completions exports
            # sampling_per_token_logps instead of importance_sampling_ratio (it neutralises
            # TRL's own correction, rl_replacements.py:1181-1184). Plain TRL 0.24 does not set
            # this key; setting it on both paths keeps one code path here.
            out.setdefault("sampling_per_token_logps", sampling_logps)

            advantages = torch.tensor(
                [row.advantage for row in batch.rows], dtype=torch.float32, device=device
            )
            trl_advantages = out.get("advantages")
            # TRL groups by row position, we group by canonical RolloutGroup and by trajectory,
            # so the two only have to agree when every trajectory produced exactly one row.
            if (
                trl_advantages is not None
                and batch.one_row_per_trajectory
                and (spec is None or spec.advantage_estimator in {"group_zscore", "group_center"})
                and trl_advantages.shape == advantages.shape
            ):
                self._arl_stats["advantage_max_abs_diff_vs_trl"] = float(
                    (trl_advantages.to(torch.float32) - advantages).abs().max()
                )
            # Advantages are grouped by canonical RolloutGroup, not by "num_generations
            # consecutive rows"; see dataset.build_batch.
            if spec is not None:
                advantages = torch.tensor(
                    [row.advantage * row_advantage_scale(spec, row) for row in batch.rows],
                    dtype=torch.float32,
                ).to(
                    device, non_blocking=True
                )
                if spec.policy_objective == "reinforce" or spec.ratio_reference == "behavior":
                    out["old_per_token_logps"] = None
            out["advantages"] = advantages

            old = out.get("old_per_token_logps")
            stats_mask = tool_mask * out["completion_mask"]
            stats_sampling = sampling_logps
            if old is not None and old.shape[1] != width:
                # COMPAT(unsloth==2026.9): Unsloth's _get_per_token_logps_and_entropies left-packs
                # the prompt padding and returns [B, logits_to_keep + max_left_pad]; row i's
                # completion starts at column max_left_pad - left_pad_i (unsloth_zoo
                # rl_replacements.py::create_completion_attention_mask). Its loss aligns our
                # width-C tool_mask / sampling_per_token_logps itself; the diagnostics here
                # have to do the same before subtracting.
                stats_mask = align_to_learner_window(stats_mask, out["prompt_mask"], old.shape[1])
                stats_sampling = align_to_learner_window(
                    sampling_logps, out["prompt_mask"], old.shape[1]
                )
                self._arl_stats["learner_logps_extra_width"] = float(old.shape[1] - width)
            self._arl_stats.update(
                _sampling_stats(
                    old=old,
                    sampling=stats_sampling,
                    mask=stats_mask,
                    cap=float(getattr(self.args, "vllm_importance_sampling_cap", 2.0) or 2.0),
                )
            )
            return out

        # -- loss ------------------------------------------------------------------
        def _compute_loss(self, model: Any, inputs: Any) -> Any:
            if self._arl_spec is not None:
                spec = self._arl_spec
                completion_mask = inputs["completion_mask"]
                mask = completion_mask * inputs.get("tool_mask", torch.ones_like(completion_mask))
                new, entropy = self._get_per_token_logps_and_entropies(
                    model,
                    torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1),
                    torch.cat([inputs["prompt_mask"], completion_mask], dim=1),
                    inputs["completion_ids"].shape[1],
                    compute_entropy=True,
                )
                advantages = inputs["advantages"]
                if self.top_entropy_quantile < 1.0:
                    entropy_mask = self.get_high_entropy_mask(
                        entropy, mask, 1 - self.top_entropy_quantile
                    )
                    advantages = advantages[:, None] * entropy_mask
                loss, aux = policy_gradient_loss(
                    new,
                    inputs.get("old_per_token_logps"),
                    inputs.get("sampling_per_token_logps"),
                    mask,
                    advantages,
                    spec=spec,
                    denominator=inputs["num_items_in_batch"],
                    num_chunks=self.current_gradient_accumulation_steps,
                    ref=inputs.get("ref_per_token_logps"),
                )
                if entropy is not None:
                    aux["entropy"] = (entropy * mask).detach()
                _attach_row_indices(aux, getattr(self, "_arl_current_rows", None))
                self._arl_loss_stats.append(aux)
                return loss
            tool_mask = inputs.get("tool_mask") if isinstance(inputs, dict) else None
            if tool_mask is None:
                return super()._compute_loss(model, inputs)

            # COMPAT(trl==0.24): grpo_trainer.py:1655-1660 uses completion_mask both as the
            # model attention mask and as the loss mask. Zeroing it for tool/user tokens would
            # hide them from attention and change every following logprob, so instead the loss
            # mask gets the action mask while _get_per_token_logps_and_entropies is pinned to
            # the complete attention mask for the duration of the call.
            completion_mask = inputs["completion_mask"]
            full_attention = torch.cat([inputs["prompt_mask"], completion_mask], dim=1)
            masked = dict(inputs)
            masked["completion_mask"] = completion_mask * tool_mask.to(
                device=completion_mask.device, dtype=completion_mask.dtype
            )

            original = self._get_per_token_logps_and_entropies

            def _with_full_attention(model_: Any, input_ids: Any, _attention_mask: Any, *a, **kw):
                return original(model_, input_ids, full_attention, *a, **kw)

            self._get_per_token_logps_and_entropies = _with_full_attention  # type: ignore[method-assign]
            try:
                return super()._compute_loss(model, masked)
            finally:
                del self._get_per_token_logps_and_entropies

        # -- metrics ---------------------------------------------------------------
        def pop_shim_stats(self) -> dict[str, float]:
            stats, self._arl_stats = self._arl_stats, {}
            if self._arl_loss_stats:
                stats.update(_aggregate_loss_stats(self._arl_loss_stats, self._arl_spec))
                (
                    self._arl_off_policy_by_staleness,
                    self._arl_off_policy_by_staleness_partial,
                ) = _aggregate_off_policy_by_staleness(
                    self._arl_loss_stats,
                    getattr(self, "_arl_row_staleness", None),
                    self._arl_spec,
                )
                self._arl_loss_stats = []
            if not self._arl_off_policy_by_staleness and "behavior_ratio_raw_mean" in stats:
                keys = (
                    "behavior_log_ratio_mean",
                    "behavior_ratio_raw_p50",
                    "behavior_ratio_raw_p95",
                    "behavior_ratio_raw_p99",
                    "behavior_ratio_raw_max",
                    "importance_ratio_mean",
                    "importance_ratio_capped_fraction",
                    "behavior_ess_relative",
                    "sampling_vs_learner_logprob_diff_mean",
                )
                self._arl_off_policy_by_staleness = {
                    "all": {
                        "num_action_tokens": int(stats.get("num_loss_action_tokens", 0)),
                        **{key: stats[key] for key in keys if key in stats},
                    }
                }
                self._arl_off_policy_by_staleness_partial = True
            return stats

    UnslothNativeGRPOTrainer.__qualname__ = "UnslothNativeGRPOTrainer"
    return UnslothNativeGRPOTrainer


def micro_batch_chunks(n_rows: int, per_device_train_batch_size: int) -> int:
    """How many equal micro-batches to back-propagate ``n_rows`` in.

    The smallest divisor ``K`` of ``n_rows`` whose chunk size fits
    ``per_device_train_batch_size``. Equal chunks are required because TRL's
    ``split_tensor_dict`` (trl/trainer/utils.py:1806-1817) floors ``n_rows // num_chunks`` and
    would silently drop a ragged tail; they are also what makes the ``grpo`` loss
    (a mean over sequences) identical to the un-chunked one. A prime ``n_rows`` larger than the
    micro-batch falls back to ``K = n_rows`` (chunks of one row), which is safe, just slower.
    """
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    limit = max(1, int(per_device_train_batch_size))
    for chunks in range(1, n_rows + 1):
        if n_rows % chunks == 0 and n_rows // chunks <= limit:
            return chunks
    return n_rows  # pragma: no cover - unreachable, chunks == n_rows always matches


def align_to_learner_window(
    narrow: torch.Tensor, prompt_mask: torch.Tensor, width: int
) -> torch.Tensor:
    """Place a ``[B, C]`` completion-aligned tensor into Unsloth's ``[B, width]`` learner window.

    ``width == C + max_left_pad``; row ``i`` (whose left-padded prompt has ``left_pad_i`` pad
    tokens) starts at column ``max_left_pad - left_pad_i``. Identity when ``width == C``.
    Mirrors ``unsloth_zoo.rl_replacements.align_logprobs_with_mask`` /
    ``create_completion_attention_mask`` without importing unsloth (CPU-testable)."""
    import torch

    batch, completion_len = narrow.shape
    if width == completion_len:
        return narrow
    if width < completion_len:
        raise ValueError(
            f"learner window ({width}) is narrower than the completion width ({completion_len})"
        )
    extra = width - completion_len
    left_pad = (prompt_mask.to(torch.long) == 0).sum(dim=1)
    offsets = (extra - left_pad).clamp(min=0)
    cols = torch.arange(completion_len, device=narrow.device).unsqueeze(0) + offsets.unsqueeze(1)
    rows = torch.arange(batch, device=narrow.device).unsqueeze(1).expand_as(cols)
    valid = cols < width
    wide = torch.zeros((batch, width), dtype=narrow.dtype, device=narrow.device)
    wide[rows[valid], cols[valid]] = narrow[valid]
    return wide


def _sampling_stats(
    *,
    old: torch.Tensor | None,
    sampling: torch.Tensor,
    mask: torch.Tensor,
    cap: float,
) -> dict[str, float]:
    """Truncated-importance-sampling diagnostics over the action tokens only."""
    import torch

    if old is None:
        return {}
    flat_mask = mask.bool()
    if not bool(flat_mask.any()):
        return {}
    diff = (old.to(torch.float32) - sampling)[flat_mask]
    raw_ratio = torch.exp(diff)
    ratio = torch.clamp(raw_ratio, max=cap)
    quant = torch.tensor([0.50, 0.95, 0.99], dtype=torch.float32, device=ratio.device)
    p50, p95, p99 = torch.quantile(ratio, quant).tolist()
    abs_p99 = torch.quantile(diff.abs(), quant[2:]).item()
    return {
        "importance_ratio_mean": float(ratio.mean()),
        "importance_ratio_p50": float(p50),
        "importance_ratio_p95": float(p95),
        "importance_ratio_p99": float(p99),
        "importance_ratio_max": float(ratio.max()),
        # Fraction of action tokens whose truncated-IS weight hit the cap: with stale
        # (asynchronous) rollouts this is the share of tokens whose correction is approximate.
        "importance_ratio_capped_fraction": float((raw_ratio >= cap).float().mean()),
        "behavior_log_ratio_mean": float(diff.mean()),
        "behavior_log_ratio_std": float(diff.std(unbiased=False)),
        "behavior_log_ratio_abs_p99": float(abs_p99),
        **_behavior_ratio_raw_stats(diff),
        "behavior_ess_relative": _ess_relative(log_weights=diff),
        "behavior_ess_relative_used": _ess_relative(ratio),
        "sampling_vs_learner_logprob_diff_mean": float(diff.mean()),
        "sampling_vs_learner_logprob_diff_abs_p99": float(abs_p99),
        "sampling_vs_learner_logprob_diff_abs_max": float(diff.abs().max()),
    }


__all__ = [
    "CONTROLLED_ARGS",
    "TRL_VERSIONS",
    "DEFAULT_LR_SCHEDULER",
    "align_to_learner_window",
    "HORIZON_FREE_SCHEDULERS",
    "METRIC_DRIVEN_SCHEDULERS",
    "build_grpo_config",
    "config_field_names",
    "make_trainer_class",
    "micro_batch_chunks",
    "scheduler_needs_horizon",
    "trim_micro_batch_padding",
    "validate_args",
    "validate_lr_schedule",
]


def _detached_old_class(parent, record=None):
    """Keep Zoo's saved-gradient backward; only supply its missing detached old input."""

    class DetachedOldEfficientGRPO(parent):
        @staticmethod
        def forward(ctx, new, old, ref, sampling, *args):
            if old is None and sampling is not None:
                old = new.detach()
            if record is not None:
                record(new, old, ref, sampling, args[2], args[3])
            return parent.forward(ctx, new, old, ref, sampling, *args)

    return DetachedOldEfficientGRPO


def _algorithm_compute_loss(trainer, original, spec):
    from moongazer_rl.trainer.unsloth_native.memory import copy_function_globals

    key = (spec, getattr(original, "__func__", original))
    cached = getattr(trainer, "_arl_algorithm_compute_loss_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    fn = inspect.unwrap(getattr(original, "__func__", original))
    upstream = fn.__globals__.get("grpo_accumulated_loss")
    if upstream is None or "grpo_accumulated_loss" not in fn.__code__.co_names:
        raise RuntimeError("unsupported Unsloth compute_loss: direct grpo_accumulated_loss missing")

    def record(new, old, ref, sampling, mask, advantages):
        aux = loss_diagnostics(new, old, sampling, mask, advantages, spec=spec, ref=ref)
        _attach_row_indices(aux, getattr(trainer, "_arl_current_rows", None))
        trainer._arl_loss_stats.append(aux)

    trainer._arl_record_logps = record
    accumulated = inspect.unwrap(upstream)
    parent = accumulated.__globals__.get("UnslothEfficientGRPO")
    if parent is not None:
        accumulated = copy_function_globals(
            accumulated,
            {
                "UnslothEfficientGRPO": _detached_old_class(
                    parent,
                    record,
                )
            },
        )
    elif accumulated.__module__ != "moongazer_rl.trainer.unsloth_native.memory":
        raise RuntimeError("unsupported accumulated loss: UnslothEfficientGRPO missing")
    else:
        # Preserve memory.py's torch.compiler.disable boundary when dispatching.
        accumulated = upstream

    # COMPAT(unsloth==2026.9): the generated compute_loss probes
    # `inspect.signature(grpo_accumulated_loss).parameters` for "tool_mask" / "num_images" before
    # calling it (rl_replacements.py:2716-2749) and raises "requires an unsloth_zoo build ..."
    # otherwise. functools.wraps sets __wrapped__, which inspect.signature follows, so the
    # wrapper advertises the upstream signature.
    @functools.wraps(accumulated)
    def capture(*args, **kwargs):
        kwargs.update(zoo_loss_kwargs(spec))
        before = len(trainer._arl_loss_stats)
        result = accumulated(*args, **kwargs)
        # Signed logprob differences cannot be recovered from Zoo's abs(delta).
        # Capture them inside the kernel below; this tuple still supplies aligned masks,
        # corrected ratios and KL on every micro-batch including the memory path.
        loss, length, kl, delta, ratio, coef, mask = result
        stats = {
            "mask": mask.detach(),
            "clip_fraction": mask.new_zeros(mask.shape),
            "kl_mean": kl.detach(),
            "kl_rows": mask.shape[0],
        }
        _attach_row_indices(stats, getattr(trainer, "_arl_current_rows", None))
        if ratio is not None and ratio.numel():
            stats["importance_ratio"] = ratio.detach()
            stats["abs_logprob_diff"] = delta.detach()
        if len(trainer._arl_loss_stats) == before:
            trainer._arl_loss_stats.append(stats)
        return result

    # Capture signed diagnostics where both logps are still available. Function globals
    # are private to this trainer, including the nested custom-autograd kernel.
    def capture_kernel(kernel):
        def wrapped(ref, new, old, sampling, ids, mask, beta, advantages, **kwargs):
            kwargs.update(zoo_loss_kwargs(spec))
            kwargs["use_vllm"] = spec.behavior_correction != "none"
            diagnostic_sampling = sampling
            if spec.behavior_correction == "none":
                sampling = None
            if old is None and sampling is not None:
                old = new.detach()
            result = kernel(ref, new, old, sampling, ids, mask, beta, advantages, **kwargs)
            record(new, old, ref, diagnostic_sampling, mask, advantages)
            return result

        return wrapped

    # Record at the custom Function boundary, outside Zoo's compiled loss kernel.
    # The direct-logprob fallback has its own runtime global replacement.
    replacements = {"grpo_accumulated_loss": capture}
    slow = fn.__globals__.get("grpo_compute_loss_slow")
    if slow is not None:
        replacements["grpo_compute_loss_slow"] = capture_kernel(slow)
    wrapped = MethodType(copy_function_globals(fn, replacements), trainer)
    trainer._arl_algorithm_compute_loss_cache = (key, wrapped)
    return wrapped


def _aggregate_loss_stats(records, spec):
    import torch

    def tokens(key):
        return torch.cat(
            [
                r[key].expand_as(r["mask"])[r["mask"].bool()].float().reshape(-1)
                for r in records
                if key in r
            ]
        )

    stats = {}
    for key, metric in (
        ("entropy", "entropy"),
        ("kl", "kl"),
        ("clip_fraction", "clipped_token_fraction"),
        ("proximal_ratio_identity", "proximal_ratio_identity"),
    ):
        if any(key in r for r in records):
            values = tokens(key)
            if values.numel():
                stats[metric] = float(values.mean())
    if any("kl_mean" in r for r in records):
        stats["kl"] = sum(
            float(r["kl_mean"]) * r["kl_rows"] for r in records if "kl_mean" in r
        ) / sum(r["kl_rows"] for r in records if "kl_mean" in r)
    if any("behavior_log_ratio" in r for r in records):
        log_ratio = tokens("behavior_log_ratio").to(dtype=torch.float64)
        if log_ratio.numel():
            stats.update(
                behavior_log_ratio_mean=float(log_ratio.mean()),
                behavior_log_ratio_std=float(log_ratio.std(unbiased=False)),
                behavior_log_ratio_abs_p99=float(torch.quantile(log_ratio.abs(), 0.99)),
                **_behavior_ratio_raw_stats(log_ratio),
                behavior_ess_relative=_ess_relative(log_weights=log_ratio),
            )
            if any("importance_ratio" in r for r in records):
                stats["behavior_ess_relative_used"] = _ess_relative(tokens("importance_ratio"))
    if spec is None or spec.behavior_correction == "none":
        return stats
    if any("importance_ratio" in r for r in records):
        ratio = tokens("importance_ratio")
        if ratio.numel():
            q = torch.quantile(ratio, torch.tensor([0.50, 0.95, 0.99], device=ratio.device))
            stats.update(
                importance_ratio_mean=float(ratio.mean()),
                importance_ratio_p50=float(q[0]),
                importance_ratio_p95=float(q[1]),
                importance_ratio_p99=float(q[2]),
                importance_ratio_max=float(ratio.max()),
            )
            if any("capped_fraction" in r for r in records):
                stats["importance_ratio_capped_fraction"] = float(tokens("capped_fraction").mean())
            elif spec.behavior_correction == "truncate":
                stats["importance_ratio_capped_fraction"] = float(
                    (ratio >= zoo_loss_kwargs(spec)["vllm_importance_sampling_cap"]).float().mean()
                )
    signed = any("logprob_diff" in r for r in records)
    key = "logprob_diff" if signed else "abs_logprob_diff"
    if any(key in r for r in records):
        diff = tokens(key)
        if diff.numel():
            if signed:
                stats["sampling_vs_learner_logprob_diff_mean"] = float(diff.mean())
            stats["sampling_vs_learner_logprob_diff_abs_p99"] = float(
                torch.quantile(diff.abs(), 0.99)
            )
            stats["sampling_vs_learner_logprob_diff_abs_max"] = float(diff.abs().max())
    return stats


def _attach_row_indices(record, row_indices) -> None:
    if row_indices is not None and len(row_indices) == record["mask"].shape[0]:
        record["row_indices"] = tuple(int(i) for i in row_indices)


def _ess_relative(weights=None, *, log_weights=None) -> float:
    """Relative ESS from weights or precomputed log-weights, without squaring in value space."""
    import math

    import torch

    if log_weights is None:
        if weights is None:
            raise ValueError("weights or log_weights is required")
        log_weights = weights.to(dtype=torch.float64).log()
    else:
        log_weights = log_weights.to(dtype=torch.float64)
    log_weights = log_weights.reshape(-1)
    count = log_weights.numel()
    if count == 0:
        return 0.0
    log_sum = torch.logsumexp(log_weights, dim=0)
    if not bool(torch.isfinite(log_sum)):
        return float("nan")
    log_square_sum = torch.logsumexp(2 * log_weights, dim=0)
    log_relative = 2 * log_sum - log_square_sum - math.log(count)
    return float(log_relative.clamp(max=0.0).exp())


def _behavior_ratio_raw_stats(log_ratio) -> dict[str, float]:
    """Raw behavior-ratio summaries computed without a float32 exponential."""
    import math

    import torch

    log_ratio = log_ratio.reshape(-1).to(dtype=torch.float64)
    if not log_ratio.numel():
        return {}
    quantiles = torch.quantile(
        log_ratio,
        torch.tensor([0.50, 0.95, 0.99], dtype=torch.float64, device=log_ratio.device),
    )
    log_mean = torch.logsumexp(log_ratio, dim=0) - math.log(log_ratio.numel())
    return {
        "behavior_ratio_raw_p50": float(quantiles[0].exp()),
        "behavior_ratio_raw_p95": float(quantiles[1].exp()),
        "behavior_ratio_raw_p99": float(quantiles[2].exp()),
        "behavior_ratio_raw_max": float(log_ratio.max().exp()),
        "behavior_ratio_raw_mean": float(log_mean.exp()),
    }


def _off_policy_bucket(records, spec) -> dict[str, float | int]:
    import torch

    def tokens(key):
        return torch.cat(
            [
                record[key].expand_as(record["mask"])[record["mask"].bool()].float().reshape(-1)
                for record in records
                if key in record
            ]
        )

    mask_tokens = sum(int(record["mask"].bool().sum()) for record in records)
    result: dict[str, float | int] = {"num_action_tokens": mask_tokens}
    if mask_tokens == 0:
        return result
    if any("behavior_log_ratio" in record for record in records):
        raw = tokens("behavior_log_ratio").to(dtype=torch.float64)
        if raw.numel():
            result.update(
                behavior_log_ratio_mean=float(raw.mean()),
                **_behavior_ratio_raw_stats(raw),
                behavior_ess_relative=_ess_relative(log_weights=raw),
            )
    correction_enabled = spec is not None and spec.behavior_correction != "none"
    if correction_enabled and any("importance_ratio" in record for record in records):
        result["importance_ratio_mean"] = float(tokens("importance_ratio").mean())
    if correction_enabled and any("capped_fraction" in record for record in records):
        result["importance_ratio_capped_fraction"] = float(tokens("capped_fraction").mean())
    if any("logprob_diff" in record for record in records):
        result["sampling_vs_learner_logprob_diff_mean"] = float(tokens("logprob_diff").mean())
    return result


def _aggregate_off_policy_by_staleness(records, row_staleness, spec):
    if not records:
        return {}, True
    if not any(
        "behavior_log_ratio" in record or "importance_ratio" in record for record in records
    ):
        return {}, True
    complete = isinstance(row_staleness, dict) and all(
        "row_indices" in record
        and len(record["row_indices"]) == record["mask"].shape[0]
        and all(index in row_staleness for index in record["row_indices"])
        for record in records
    )
    if not complete:
        return {"all": _off_policy_bucket(records, spec)}, True
    buckets = {}
    for staleness in sorted(set(row_staleness.values())):
        selected = []
        for record in records:
            positions = [
                pos
                for pos, index in enumerate(record["row_indices"])
                if row_staleness[index] == staleness
            ]
            if not positions:
                continue
            sliced = {
                key: value[positions]
                if hasattr(value, "shape")
                and value.ndim > 0
                and value.shape[0] == record["mask"].shape[0]
                else value
                for key, value in record.items()
                if key != "row_indices"
            }
            selected.append(sliced)
        if selected:
            buckets[str(staleness)] = _off_policy_bucket(selected, spec)
    return buckets, False
