"""Aggregate stats computed purely from canonical data (``RolloutGroup``/``StepResult``).

``group_stats`` never invents a value: a quantity with no supporting data (e.g. no model
call ever reported logprobs) is simply absent from the returned dict rather than reported
as ``0.0`` — the one exception being counts (``num_groups``, ``num_trajectories``,
``*_total``), where "zero" and "no data" are the same fact.
"""

from __future__ import annotations

import math
import statistics

from moongazer_rl.rollout.schema import RolloutGroup
from moongazer_rl.trainer.base import StepResult

_TIMING_KEYS = ("setup_s", "generation_s", "finalize_s", "scoring_s")


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile (numpy's default method), ``p`` in ``[0, 1]``.

    Kept local instead of ``statistics.quantiles`` because that function raises for a
    single-element input, which a small run (or a single-group step) hits often.
    """
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[int(f)] * (c - k) + s[int(c)] * (k - f)


def percentile(values: list[float], p: float) -> float | None:
    """Return the linear-interpolation percentile, or ``None`` for no observations."""
    if not values:
        return None
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1]")
    return _percentile(values, p)


def group_stats(groups: list[RolloutGroup]) -> dict[str, float]:
    """Everything obtainable from a batch of ``RolloutGroup``s (one training step's
    worth, typically). See the module docstring for the "never invent a value" rule.

    ``judge_error_rate`` heuristic: a trajectory counts as a judge error when at least
    one of its ``errors[].type`` contains the substring ``"Judge"`` (case-sensitive) —
    verifiers has no separate "this was a judge failure" flag, so this is a naming
    convention, not a guarantee; environments whose judge error types don't mention
    "Judge" will under-count here.
    """
    stats: dict[str, float] = {"num_groups": float(len(groups))}
    all_trajs = [t for g in groups for t in g.trajectories]
    stats["num_trajectories"] = float(len(all_trajs))
    if not all_trajs:
        return stats

    rewards = [t.reward for t in all_trajs]
    finite_rewards = [r for r in rewards if math.isfinite(r)]
    if finite_rewards:
        stats["reward_mean"] = statistics.fmean(finite_rewards)
        stats["reward_std"] = statistics.pstdev(finite_rewards) if len(finite_rewards) >= 2 else 0.0
        stats["reward_min"] = min(finite_rewards)
        stats["reward_max"] = max(finite_rewards)

    stats["zero_std_group_fraction"] = sum(1 for g in groups if g.is_zero_std) / len(groups)

    comp_sums: dict[str, float] = {}
    comp_counts: dict[str, int] = {}
    for t in all_trajs:
        for key, value in t.reward_components.items():
            comp_sums[key] = comp_sums.get(key, 0.0) + value
            comp_counts[key] = comp_counts.get(key, 0) + 1
    for key, total in comp_sums.items():
        stats[f"reward_component_mean.{key}"] = total / comp_counts[key]

    n = len(all_trajs)
    stats["error_rate"] = sum(1 for t in all_trajs if t.has_error) / n
    stats["truncation_rate"] = sum(1 for t in all_trajs if t.is_truncated) / n
    stats["completed_rate"] = sum(1 for t in all_trajs if t.is_completed) / n

    error_type_counts: dict[str, int] = {}
    for t in all_trajs:
        for err in t.errors:
            error_type_counts[err.type] = error_type_counts.get(err.type, 0) + 1
    for err_type, count in error_type_counts.items():
        stats[f"error_type_counts.{err_type}"] = float(count)

    stats["judge_error_rate"] = (
        sum(1 for t in all_trajs if any("Judge" in err.type for err in t.errors)) / n
    )
    stats["judge_calls"] = float(sum(len(t.extra_usage) for t in all_trajs))

    turns = [t.num_turns for t in all_trajs]
    stats["turns_mean"] = statistics.fmean(turns)
    stats["turns_max"] = float(max(turns))

    input_tokens = [t.num_input_tokens for t in all_trajs]
    output_tokens = [t.num_output_tokens for t in all_trajs]
    stats["input_tokens_mean"] = statistics.fmean(input_tokens)
    stats["output_tokens_mean"] = statistics.fmean(output_tokens)
    stats["output_tokens_p95"] = _percentile([float(x) for x in output_tokens], 0.95)
    stats["action_tokens_total"] = float(sum(t.num_action_tokens for t in all_trajs))

    calls = [c for t in all_trajs for c in t.model_calls]
    stats["model_calls_total"] = float(len(calls))
    stats["calls_per_trajectory_mean"] = len(calls) / n

    all_logprobs = [lp for c in calls if c.has_logprobs for lp in c.sampling_logprobs]
    if all_logprobs:
        stats["sampling_logprob_mean"] = statistics.fmean(all_logprobs)
    if calls:
        stats["missing_logprob_fraction"] = sum(1 for c in calls if not c.has_logprobs) / len(calls)

    stats["fidelity_problem_fraction"] = (
        sum(1 for t in all_trajs if t.metadata.get("fidelity_problems")) / n
    )

    stats["policy_version_min"] = float(min(g.policy_version for g in groups))
    stats["policy_version_max"] = float(max(g.policy_version for g in groups))

    for key in _TIMING_KEYS:
        values = [t.timing[key] for t in all_trajs if key in t.timing]
        if values:
            stats[f"time_{key}_mean"] = statistics.fmean(values)

    return stats


def throughput(groups: list[RolloutGroup], rollout_wall_time_s: float | None) -> dict[str, float]:
    """Rollout-phase throughput for one step's groups. ``rollout_tokens_per_s`` counts
    output (generated) tokens, since that is what the rollout phase is generating; input
    tokens are prompt-side and dominated by context reuse, not generation cost. Absent
    (not zero) when there is no wall-time or no data to divide."""
    if not groups or not rollout_wall_time_s or rollout_wall_time_s <= 0:
        return {}
    all_trajs = [t for g in groups for t in g.trajectories]
    if not all_trajs:
        return {}
    total_output_tokens = sum(t.num_output_tokens for t in all_trajs)
    return {
        "rollout_tokens_per_s": total_output_tokens / rollout_wall_time_s,
        "trajectories_per_s": len(all_trajs) / rollout_wall_time_s,
    }


def train_throughput(result: StepResult, train_wall_time_s: float | None) -> dict[str, float]:
    """Train-phase throughput for one ``StepResult``: action tokens and trajectories
    consumed by the optimizer step per second of training wall time."""
    if not train_wall_time_s or train_wall_time_s <= 0:
        return {}
    out: dict[str, float] = {
        "train_trajectories_per_s": result.num_trajectories_used / train_wall_time_s,
    }
    if result.num_action_tokens:
        out["train_tokens_per_s"] = result.num_action_tokens / train_wall_time_s
    if "num_loss_action_tokens" in result.metrics:
        out["train_loss_tokens_per_s"] = (
            result.metrics["num_loss_action_tokens"] / train_wall_time_s
        )
    return out


__all__ = ["group_stats", "percentile", "throughput", "train_throughput"]
