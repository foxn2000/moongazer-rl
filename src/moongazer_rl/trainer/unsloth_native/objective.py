"""Frozen algorithm semantics and a torch-only reference for the Zoo loss boundary."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class AlgorithmSpec:
    advantage_estimator: str = "group_zscore"
    std_scope: str = "group"
    policy_objective: str = "grpo_clip"
    ratio_reference: str = "proximal"
    behavior_correction: str = "truncate"
    behavior_cap: float = 2.0
    behavior_clip_min: float | None = None
    behavior_clip_max: float | None = None
    loss_normalization: str = "action_token"
    fixed_length: int | None = None
    mask_truncated_completions: bool = True
    skip_old_logprob_forward: bool = True
    epsilon_low: float = 0.2
    epsilon_high: float = 0.2
    delta: float | None = None
    beta: float = 0.0
    importance_sampling_level: str = "token"


def resolve_algorithm(
    section, trainer_args: dict, *, max_completion_length: int, lora_dropout: float
) -> tuple[AlgorithmSpec, list[str]]:
    if section is None:
        explicit = {}
    elif isinstance(section, Mapping):
        explicit = dict(section)
    elif hasattr(section, "model_dump"):
        explicit = section.model_dump(exclude_unset=True)
    else:
        explicit = vars(section).copy()
    explicit = {k: v for k, v in explicit.items() if v is not None}
    problems = []
    values = {}

    def legacy(key, mapping):
        for name, value in mapping.items():
            if name in explicit and explicit[name] != value:
                problems.append(f"trainer.algorithm.{name} conflicts with trainer.args.{key}")
            values[name] = value

    loss_type = trainer_args.get("loss_type", "dapo")
    normalizations = {
        "dapo": "action_token",
        "bnpo": "action_token",
        "grpo": "sequence",
        "dr_grpo": "fixed_length",
    }
    if loss_type not in normalizations:
        problems.append(f"unsupported legacy trainer.args.loss_type={loss_type!r}")
    elif "loss_type" in trainer_args:
        mapped = dict(policy_objective="grpo_clip", loss_normalization=normalizations[loss_type])
        if loss_type == "dr_grpo":
            mapped["fixed_length"] = int(
                trainer_args.get("max_completion_length", max_completion_length)
            )
        legacy("loss_type", mapped)
    if loss_type == "bnpo":
        warnings.warn(
            "bnpo maps to global action_token normalization; identical only for one micro-batch",
            UserWarning,
            stacklevel=2,
        )
    if "scale_rewards" in trainer_args:
        scale = trainer_args["scale_rewards"]
        if scale in ("group", True):
            legacy("scale_rewards", dict(advantage_estimator="group_zscore", std_scope="group"))
        elif scale == "batch":
            legacy("scale_rewards", dict(advantage_estimator="group_zscore", std_scope="batch"))
        elif scale in ("none", False):
            legacy("scale_rewards", dict(advantage_estimator="group_center"))
        else:
            problems.append("invalid trainer.args.scale_rewards")
    values.setdefault("fixed_length", int(max_completion_length))
    for key, name in {
        "epsilon": "epsilon_low",
        "epsilon_high": "epsilon_high",
        "delta": "delta",
        "beta": "beta",
        "importance_sampling_level": "importance_sampling_level",
        "vllm_importance_sampling_cap": "behavior_cap",
    }.items():
        if key in trainer_args and trainer_args[key] is not None:
            legacy(key, {name: trainer_args[key]})
    values.update(explicit)
    values.setdefault(
        "epsilon_high",
        values.get("behavior_cap", 2.0)
        if values.get("policy_objective") == "cispo"
        else values.get("epsilon_low", 0.2),
    )
    values["mask_truncated_completions"] = trainer_args.get(
        "mask_truncated_completions", loss_type == "dapo"
    )
    skip = values.pop("skip_old_logprob_forward", "auto")
    safe = (
        lora_dropout == 0
        and trainer_args.get("num_iterations", 1) == 1
        and not trainer_args.get("sync_ref_model", False)
    )
    behavior = values.get("ratio_reference", "proximal") == "behavior"
    if skip == "auto":
        skip = behavior or safe
    elif not isinstance(skip, bool):
        problems.append("skip_old_logprob_forward must be auto or bool")
        skip = False
    if skip and not safe and not behavior:
        problems.append(
            "skip_old_logprob_forward requires zero LoRA dropout, num_iterations=1 "
            "and no ref snapshot synchronization"
        )
    if behavior and not skip:
        problems.append("ratio_reference=behavior requires skip_old_logprob_forward")
    values["skip_old_logprob_forward"] = skip
    spec = AlgorithmSpec(**values)
    for name, choices in {
        "advantage_estimator": {
            "group_zscore",
            "group_center",
            "rloo",
        },
        "std_scope": {"group", "batch"},
        "policy_objective": {"grpo_clip", "cispo", "reinforce"},
        "ratio_reference": {"proximal", "behavior"},
        "behavior_correction": {"none", "truncate", "mask"},
        "loss_normalization": {"sequence", "action_token", "fixed_length"},
        "importance_sampling_level": {"token", "sequence"},
    }.items():
        if getattr(spec, name) not in choices:
            problems.append(f"invalid {name}={getattr(spec, name)!r}")
    if spec.policy_objective == "grpo_clip" and behavior:
        problems.append("grpo_clip + ratio_reference=behavior is not expressible by the Zoo path")
    if requires_equal_chunks(spec) and spec.policy_objective != "grpo_clip":
        problems.append(
            "sequence + beta != 0 requires grpo_clip: native Zoo grpo changes "
            "the CISPO/REINFORCE objective"
        )
    if (
        spec.policy_objective == "reinforce"
        and spec.ratio_reference == "proximal"
        and spec.behavior_correction != "none"
        and lora_dropout != 0
    ):
        problems.append(
            "reinforce with proximal behavior correction requires zero LoRA dropout: "
            "Zoo's unit-ratio route also uses current.detach() for the behavior weight"
        )
    if spec.policy_objective == "cispo" and behavior:
        if spec.behavior_correction != "truncate":
            problems.append(
                "cispo + behavior requires behavior_correction=truncate to express CISPO clipping"
            )
    if spec.policy_objective == "cispo" and spec.epsilon_high < 1:
        problems.append("cispo requires epsilon_high >= 1 (an IS ratio cap, not a clip width)")
    if "std_scope" in explicit and spec.advantage_estimator != "group_zscore":
        problems.append("std_scope is only valid for group_zscore")
    for name in ("behavior_cap", "fixed_length"):
        if not math.isfinite(getattr(spec, name)) or getattr(spec, name) <= 0:
            problems.append(f"{name} must be positive and finite")
    for name in ("epsilon_low", "epsilon_high", "beta"):
        if not math.isfinite(getattr(spec, name)) or getattr(spec, name) < 0:
            problems.append(f"{name} must be nonnegative and finite")
    if spec.behavior_clip_min is not None and spec.behavior_clip_min < 0:
        problems.append("behavior_clip_min must be nonnegative")
    upper = spec.behavior_clip_max if spec.behavior_clip_max is not None else spec.behavior_cap
    if upper <= 0 or (spec.behavior_clip_min is not None and spec.behavior_clip_min > upper):
        problems.append("invalid behavior clip interval")
    return spec, problems


def requires_equal_chunks(spec: AlgorithmSpec) -> bool:
    return spec.loss_normalization == "sequence" and spec.beta != 0


def zoo_loss_kwargs(spec: AlgorithmSpec) -> dict:
    cap = (
        spec.epsilon_high
        if spec.policy_objective == "cispo" and spec.ratio_reference == "behavior"
        else spec.behavior_cap
    )
    truncate = spec.behavior_correction != "mask"
    return {
        "loss_type": "grpo"
        if requires_equal_chunks(spec)
        else ("dapo" if spec.policy_objective == "grpo_clip" else "cispo"),
        "epsilon_low": spec.epsilon_low,
        # REINFORCE has no proximal ratio or clipping, so r=1 must pass unchanged.
        "epsilon_high": max(1.0, spec.epsilon_high)
        if spec.policy_objective == "reinforce"
        else spec.epsilon_high,
        "vllm_importance_sampling_mode": (
            spec.importance_sampling_level + ("_truncate" if truncate else "_mask")
        ),
        "vllm_importance_sampling_cap": cap,
        "vllm_importance_sampling_clip_min": None if truncate else spec.behavior_clip_min,
        "vllm_importance_sampling_clip_max": cap
        if truncate
        else (spec.behavior_clip_max if spec.behavior_clip_max is not None else cap),
        "importance_sampling_level": spec.importance_sampling_level,
    }


def global_denominator(spec: AlgorithmSpec, rows: Sequence[Any]) -> float:
    if spec.loss_normalization == "action_token":
        return float(
            max(
                1,
                sum(
                    row.num_action_tokens
                    for row in rows
                    if not (spec.mask_truncated_completions and getattr(row, "truncated", False))
                ),
            )
        )
    return float(
        max(1, len(rows) * (spec.fixed_length if spec.loss_normalization == "fixed_length" else 1))
    )


def row_advantage_scale(spec: AlgorithmSpec, row: Any) -> float:
    return (
        1.0 / max(1, row.num_action_tokens)
        if spec.loss_normalization == "sequence" and not requires_equal_chunks(spec)
        else 1.0
    )


def policy_gradient_loss(
    new,
    old,
    sampling,
    mask,
    advantages,
    *,
    spec: AlgorithmSpec,
    denominator: float,
    ref=None,
    num_chunks: int = 1,
) -> tuple[torch.Tensor, dict]:
    """One global-normalized micro-batch. Advantages have already received row scaling.

    The proximal ratio uses the masked mean log difference for sequence IS; the
    independent behavior correction uses the masked SUM, exactly as Zoo 2026.9.1.
    capped_fraction counts weights at/above the cap in truncate mode and weights
    strictly outside either bound in mask mode (importance_ratio_capped_fraction).
    """
    keep = mask.bool()
    new = torch.where(keep, new, 0.0)
    old = new.detach() if old is None else torch.where(keep, old, 0.0)
    if spec.ratio_reference == "behavior":
        old = new.detach()
    log_ratio = new - old
    if spec.policy_objective == "reinforce":
        log_ratio = new - new.detach()
    if spec.importance_sampling_level == "sequence":
        log_ratio = (log_ratio * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp(
            min=1
        )
    ratio = log_ratio.exp()
    a = advantages[:, None] if advantages.ndim == 1 else advantages
    if spec.policy_objective == "grpo_clip":
        clipped = ratio.clamp(1 - spec.epsilon_low, 1 + spec.epsilon_high)
        first = ratio if spec.delta is None else ratio.clamp(max=spec.delta)
        token_loss = -torch.minimum(first * a, clipped * a)
        clip = ((ratio < 1 - spec.epsilon_low) & (a < 0)) | (
            (ratio > 1 + spec.epsilon_high) & (a > 0)
        )
    elif spec.policy_objective == "cispo":
        token_loss = -ratio.clamp(max=spec.epsilon_high).detach() * a * new
        clip = ratio > spec.epsilon_high
    else:
        token_loss = -a * new
        clip = torch.zeros_like(new, dtype=torch.bool)
    diff = torch.zeros_like(new)
    behavior_log_ratio = torch.zeros_like(new)
    behavior_ratio_raw = torch.ones_like(new)
    weight = torch.ones_like(new)
    capped = torch.zeros_like(new, dtype=torch.bool)
    if sampling is not None:
        diff = torch.where(keep, old.detach() - sampling.detach(), 0.0)
        behavior_log_ratio = diff
        if spec.importance_sampling_level == "sequence":
            behavior_log_ratio = diff.sum(-1, keepdim=True)
        behavior_ratio_raw = behavior_log_ratio.exp()
    if spec.behavior_correction != "none" and sampling is not None:
        raw = behavior_ratio_raw
        kw = zoo_loss_kwargs(spec)
        lower, upper = (
            kw["vllm_importance_sampling_clip_min"],
            kw["vllm_importance_sampling_clip_max"],
        )
        lower_bound = lower if lower is not None else -math.inf
        capped = raw >= upper
        if spec.behavior_correction == "truncate":
            weight = raw.clamp(max=upper)
        else:
            capped = (raw < lower_bound) | (raw > upper)
            weight = torch.where((raw >= lower_bound) & (raw <= upper), raw, 0.0)
        token_loss = token_loss * weight
    kl = torch.zeros_like(new)
    if spec.beta != 0:
        ref = torch.where(keep, ref, 0.0)
        d = ref - new
        kl = d.exp() - d - 1
        token_loss = token_loss + spec.beta * kl
    if requires_equal_chunks(spec):
        loss = ((token_loss * mask.float()).sum(-1) / mask.sum(-1).clamp(min=1)).mean() / num_chunks
    else:
        loss = (token_loss * mask.float()).sum() / denominator
    aux = {
        "importance_ratio": (weight * mask).detach(),
        "logprob_diff": diff.detach(),
        "capped_fraction": (capped * mask).detach(),
        "clip_fraction": (clip * mask).detach(),
        "policy_ratio": (ratio * mask).detach(),
        "proximal_ratio_identity": mask.detach()
        * float(
            spec.skip_old_logprob_forward
            or spec.ratio_reference == "behavior"
            or spec.policy_objective == "reinforce"
        ),
        "kl": (kl * mask).detach(),
        "mask": mask.detach(),
    }
    if sampling is not None:
        aux["behavior_log_ratio"] = (behavior_log_ratio * mask).detach()
        # Kept for direct CPU inspection only. Aggregate raw-ratio metrics are derived from
        # behavior_log_ratio in float64 so this float32 exponential cannot poison diagnostics.
        aux["behavior_ratio_raw"] = (behavior_ratio_raw * mask).detach()
    return loss, aux


@torch.no_grad()
def loss_diagnostics(new, old, sampling, mask, advantages, *, spec: AlgorithmSpec, ref=None):
    """Compute detached action-token statistics without constructing policy loss.

    ``capped_fraction`` feeds ``importance_ratio_capped_fraction``: truncate counts
    weights at/above the cap; mask counts weights strictly outside either bound.
    """
    keep = mask.bool()
    new = torch.where(keep, new.detach(), 0.0)
    old = (
        new
        if old is None or spec.ratio_reference == "behavior"
        else torch.where(keep, old.detach(), 0.0)
    )
    log_ratio = torch.zeros_like(new) if spec.policy_objective == "reinforce" else new - old
    if spec.importance_sampling_level == "sequence":
        log_ratio = log_ratio.sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp(min=1)
    ratio = log_ratio.exp()
    a = advantages[:, None] if advantages.ndim == 1 else advantages
    if spec.policy_objective == "grpo_clip":
        clip = ((ratio < 1 - spec.epsilon_low) & (a < 0)) | (
            (ratio > 1 + spec.epsilon_high) & (a > 0)
        )
    elif spec.policy_objective == "cispo":
        clip = ratio > spec.epsilon_high
    else:
        clip = torch.zeros_like(new, dtype=torch.bool)
    diff = torch.zeros_like(new)
    behavior_log_ratio = torch.zeros_like(new)
    behavior_ratio_raw = torch.ones_like(new)
    weight = torch.ones_like(new)
    capped = torch.zeros_like(new, dtype=torch.bool)
    if sampling is not None:
        diff = torch.where(keep, old - sampling.detach(), 0.0)
        behavior_log_ratio = (
            diff.sum(-1, keepdim=True) if spec.importance_sampling_level == "sequence" else diff
        )
        behavior_ratio_raw = behavior_log_ratio.exp()
    if spec.behavior_correction != "none" and sampling is not None:
        raw = behavior_ratio_raw
        kw = zoo_loss_kwargs(spec)
        lower, upper = (
            kw["vllm_importance_sampling_clip_min"],
            kw["vllm_importance_sampling_clip_max"],
        )
        lower_bound = lower if lower is not None else -math.inf
        if spec.behavior_correction == "truncate":
            capped = raw >= upper
            weight = raw.clamp(max=upper)
        else:
            capped = (raw < lower_bound) | (raw > upper)
            weight = torch.where(capped, 0.0, raw)
    kl = torch.zeros_like(new)
    if spec.beta != 0:
        d = torch.where(keep, ref.detach(), 0.0) - new
        kl = d.exp() - d - 1
    aux = {
        "importance_ratio": weight * mask,
        "logprob_diff": diff,
        "capped_fraction": capped * mask,
        "clip_fraction": clip * mask,
        "policy_ratio": ratio * mask,
        "proximal_ratio_identity": mask.detach()
        * float(
            spec.skip_old_logprob_forward
            or spec.ratio_reference == "behavior"
            or spec.policy_objective == "reinforce"
        ),
        "kl": kl * mask,
        "mask": mask.detach(),
    }
    if sampling is not None:
        aux["behavior_log_ratio"] = behavior_log_ratio * mask
        # Kept for direct CPU inspection only; metric aggregation uses behavior_log_ratio.
        aux["behavior_ratio_raw"] = behavior_ratio_raw * mask
    return aux
