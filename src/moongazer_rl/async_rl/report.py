"""Measure overlap from recorded intervals, without assuming GPU kernel concurrency."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from moongazer_rl.logging.metrics import percentile

Interval = tuple[float, float]


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _union(intervals: list[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for start, end in sorted(intervals):
        if end < start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = merged[-1][0], max(end, merged[-1][1])
        else:
            merged.append((start, end))
    return merged


def _length(intervals: list[Interval]) -> float:
    return sum(end - start for start, end in intervals)


def _intersection(left: list[Interval], right: list[Interval]) -> float:
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        total += max(0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def _paired(events: list[dict], prefix: str, identity: str) -> list[Interval]:
    """Pair intervals within one process generation.

    A failed train is closed at the failure event's wall timestamp. This avoids extending an
    abandoned start to a later train while still counting the work completed before failure.
    """
    starts: dict[tuple[Any, Any], float] = {}
    intervals: list[Interval] = []
    for row in events:
        key = row.get("run_gen"), row.get(identity)
        if row.get("kind") == f"{prefix}_started":
            starts[key] = row["ts"]
        elif row.get("kind") == f"{prefix}_finished" and key in starts:
            intervals.append((starts.pop(key), row["ts"]))
        elif row.get("kind") in (f"{prefix}_failed", f"{prefix}_skipped") and key in starts:
            # A failed or skipped train (rollout) still occupied its side until that event.
            intervals.append((starts.pop(key), row["ts"]))
    return _union(intervals)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean_s": fmean(values) if values else None,
        "max_s": max(values) if values else None,
    }


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    """Small JSON-safe distribution summary for optional observability values."""
    return {
        **_summary(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
    }


def _numbers(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if isinstance(row.get(key), int | float)]


def _step_metric_values(steps: list[dict[str, Any]], key: str) -> list[float]:
    return [
        float(metrics[key])
        for row in steps
        if isinstance(metrics := row.get("metrics"), dict)
        and isinstance(metrics.get(key), int | float)
    ]


def _total_mean(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "total_s": sum(values),
        "mean_s": fmean(values) if values else None,
    }


def _mean_and_worst(
    values: list[float], *, absolute_worst: bool = False
) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": fmean(values) if values else None,
        "worst": (max(values, key=abs) if absolute_worst else max(values)) if values else None,
    }


def _mean_and_min(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": fmean(values) if values else None,
        "min": min(values) if values else None,
    }


def _aggregate_off_policy_by_staleness(steps: list[dict[str, Any]]) -> dict[str, dict]:
    totals: dict[str, int] = defaultdict(int)
    weighted: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    maxima: dict[str, dict[str, float]] = defaultdict(dict)
    for row in steps:
        extra = row.get("extra")
        buckets = extra.get("off_policy_by_staleness") if isinstance(extra, dict) else None
        if not isinstance(buckets, dict):
            continue
        for staleness, values in buckets.items():
            if not isinstance(values, dict):
                continue
            count = values.get("num_action_tokens")
            if not isinstance(count, int | float) or count <= 0:
                continue
            key = str(staleness)
            tokens = int(count)
            totals[key] += tokens
            for metric, value in values.items():
                if metric == "num_action_tokens" or not isinstance(value, int | float):
                    continue
                if metric.endswith("_max"):
                    maxima[key][metric] = max(maxima[key].get(metric, float("-inf")), value)
                else:
                    weighted[key][metric] += float(value) * tokens
    result = {}
    for staleness, total in totals.items():
        result[staleness] = {
            "num_action_tokens": total,
            **{metric: value / total for metric, value in weighted[staleness].items()},
            **maxima[staleness],
        }
    return dict(
        sorted(
            result.items(),
            key=lambda item: (item[0] == "all", int(item[0]) if item[0] != "all" else 0),
        )
    )


def analyze(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    events = _rows(run_dir / "events.jsonl")
    calls = _rows(run_dir / "model_calls.jsonl")
    steps = _rows(run_dir / "steps.jsonl")
    trains = _paired(events, "train", "train_id")
    episodes = _paired(events, "rollout", "group_id")
    call_intervals: list[Interval] = []
    missing_intervals = 0
    for call in calls:
        extra = call.get("extra") or {}
        start = call.get("started_at", extra.get("started_at"))
        end = call.get("ended_at", extra.get("ended_at"))
        if isinstance(start, int | float) and isinstance(end, int | float) and end >= start:
            call_intervals.append((start, end))
        else:
            missing_intervals += 1
    model = _union(call_intervals)
    train_s = _length(trains)
    model_s = _intersection(model, trains)
    episode_s = _intersection(episodes, trains)
    generations: dict[str, list[float]] = defaultdict(list)
    for row in events:
        if "ts" in row:
            generations[row.get("run_gen", "legacy")].append(row["ts"])
    duration_s = sum(max(ts) - min(ts) for ts in generations.values())
    kinds: dict[str, list[dict]] = defaultdict(list)
    for row in events:
        kinds[row["kind"]].append(row)
    histogram = Counter(str(row["staleness"]) for row in kinds["group_dequeued"])
    zero_std_dropped_by_reason = Counter(
        row.get("reason", "unknown") for row in kinds["zero_std_group_dropped"]
    )
    zero_std_dropped_by_stage = Counter(
        row.get("stage", "learner") for row in kinds["zero_std_group_dropped"]
    )
    rollout_tokens = sum(row.get("num_tokens", 0) for row in kinds["rollout_finished"])
    learner_tokens = sum(row.get("num_action_tokens", 0) for row in steps)
    rollout_finished = kinds["rollout_finished"]
    rollout_group_rates = _numbers(rollout_finished, "tokens_per_s")
    rollout_durations = _numbers(rollout_finished, "duration")
    train_durations = _numbers(kinds["train_finished"], "duration")
    learner_idle_s = sum(_numbers(kinds["learner_idle"], "duration"))
    producer_idle_s = sum(_numbers(kinds["producer_idle"], "duration"))
    publish_requested = kinds["publish_requested"]
    publish_finished = kinds["publish_finished"]
    behavior_versions = _numbers(kinds["rollout_started"], "behavior_version")
    behavior_versions += _numbers(kinds["group_dequeued"], "behavior_version")
    learner_versions = _numbers(kinds["group_dequeued"], "learner_version")
    learner_versions += _numbers(kinds["train_finished"], "learner_version")
    microbatch_counts = [
        len(extra["microbatches"])
        for row in steps
        if isinstance(extra := row.get("extra"), dict)
        and isinstance(extra.get("microbatches"), list)
    ]
    action_tokens = _numbers(steps, "num_action_tokens")
    loss_action_tokens = _step_metric_values(steps, "num_loss_action_tokens")
    forward_tokens = _step_metric_values(steps, "num_forward_tokens")
    model_tokens = _step_metric_values(steps, "num_tokens")
    off_policy_by_staleness_partial_steps = sum(
        1
        for row in steps
        if isinstance(extra := row.get("extra"), dict)
        and extra.get("off_policy_by_staleness_partial") is True
    )
    fidelity_dropped_trajectories = sum(
        int(value)
        for row in steps
        if isinstance(extra := row.get("extra"), dict)
        and isinstance(value := extra.get("num_dropped_fidelity"), int | float)
    )
    replica_loads: dict[str, list[float]] = defaultdict(list)
    for row in kinds["publish_replica_finished"]:
        url = row.get("url")
        if isinstance(url, str):
            load_s = row.get("load_s")
            if isinstance(load_s, int | float):
                replica_loads[url].append(float(load_s))
            else:
                replica_loads.setdefault(url, [])
    time_breakdown = {
        key.removeprefix("time/"): _total_mean(_step_metric_values(steps, key))
        for key in (
            "time/score_forward_s", "time/forward_backward_s", "time/grad_clip_s",
            "time/optimizer_s", "time/loss_stats_s", "time/step_s",
        )
    }
    sampling = {
        key: _mean_and_worst(
            _step_metric_values(steps, key),
            absolute_worst=key == "sampling_vs_learner_logprob_diff_mean",
        )
        for key in (
            "importance_ratio_mean",
            "importance_ratio_p95",
            "importance_ratio_p99",
            "importance_ratio_capped_fraction",
            "sampling_vs_learner_logprob_diff_mean",
        )
    }
    sampling["behavior_ess_relative"] = _mean_and_min(
        _step_metric_values(steps, "behavior_ess_relative")
    )
    sampling["behavior_ratio_raw_p99"] = _mean_and_worst(
        _step_metric_values(steps, "behavior_ratio_raw_p99")
    )
    off_policy_by_staleness = _aggregate_off_policy_by_staleness(steps)
    exact_model_overlap = missing_intervals == 0
    return {
        "run_duration_s": duration_s,
        "train_s": train_s,
        "model_call_s": _length(model),
        "model_train_overlap_s": model_s if exact_model_overlap else None,
        "model_train_overlap_fraction": (
            model_s / train_s if train_s and exact_model_overlap else None
        ),
        "model_train_overlap_lower_bound_s": model_s if not exact_model_overlap else None,
        "model_train_overlap_lower_bound_fraction": (
            model_s / train_s if train_s and not exact_model_overlap else None
        ),
        "model_calls_missing_intervals": missing_intervals,
        "episode_s": _length(episodes),
        "episode_train_overlap_s": episode_s,
        "episode_train_overlap_fraction": episode_s / train_s if train_s else None,
        "staleness_histogram": dict(sorted(histogram.items(), key=lambda kv: int(kv[0]))),
        "stale_dropped": len(kinds["stale_group_dropped"]),
        "zero_std_dropped": len(kinds["zero_std_group_dropped"]),
        "zero_std_dropped_by_reason": dict(sorted(zero_std_dropped_by_reason.items())),
        "zero_std_dropped_by_stage": dict(sorted(zero_std_dropped_by_stage.items())),
        "train_skipped": len(kinds["train_skipped"]),
        "fidelity_violations": len(kinds["fidelity_violation"]),
        "fidelity_dropped_trajectories": fidelity_dropped_trajectories,
        "lease_unavailable": len(kinds["lease_unavailable"]),
        "publish_latency": _summary(_numbers(kinds["publish_finished"], "latency_s")),
        "publisher_lag": _summary(_numbers(kinds["publish_finished"], "lag_s")),
        "publish_coalesced": len(kinds["publish_coalesced"]),
        "rollout_tokens": rollout_tokens,
        "rollout_tokens_per_s": rollout_tokens / duration_s if duration_s else None,
        "learner_action_tokens": learner_tokens,
        "learner_action_tokens_per_s": learner_tokens / duration_s if duration_s else None,
        "learner_loss_action_tokens": sum(loss_action_tokens) if loss_action_tokens else None,
        "learner_loss_action_tokens_per_s": (
            sum(loss_action_tokens) / duration_s if loss_action_tokens and duration_s else None
        ),
        "learner_forward_tokens": sum(forward_tokens) if forward_tokens else None,
        "learner_forward_tokens_per_s": (
            sum(forward_tokens) / duration_s if forward_tokens and duration_s else None
        ),
        "queue": {
            "max_occupancy": max(
                (row.get("queue_size", 0) for row in kinds["group_enqueued"]), default=0
            ),
            "enqueue_wait": _summary([row.get("wait", 0) for row in kinds["group_enqueued"]]),
            "queue_wait": _summary(
                [
                    row["queue_wait_s"]
                    for row in kinds["group_dequeued"]
                    if row.get("queue_wait_s") is not None
                ]
            ),
        },
        "learner_idle_s": learner_idle_s,
        "producer_admission_wait_s": producer_idle_s,
        "rollout": {
            "tokens_per_s_run": rollout_tokens / duration_s if duration_s else None,
            "tokens_per_s_completed_groups": _distribution(rollout_group_rates),
            "group_latency_s": _distribution(rollout_durations),
            "producer_idle_s": producer_idle_s,
        },
        "learner": {
            "step_time_s": _total_mean(train_durations),
            "time_breakdown_s": time_breakdown,
            "idle_s": learner_idle_s,
            "idle_fraction_of_run": learner_idle_s / duration_s if duration_s else None,
        },
        "publish": {
            "snapshot_s": _summary(_numbers(publish_requested, "adapter_snapshot_s")),
            "export_s": _summary(_numbers(publish_finished, "export_s")),
            "load_s": _summary(_numbers(publish_finished, "load_s")),
            "latency_s": _summary(_numbers(publish_finished, "latency_s")),
            "coalesced": len(kinds["publish_coalesced"]),
        },
        "versions": {
            "staleness": dict(sorted(histogram.items(), key=lambda kv: int(kv[0]))),
            "behavior_version_range": (
                {"min": min(behavior_versions), "max": max(behavior_versions)}
                if behavior_versions
                else None
            ),
            "learner_version_range": (
                {"min": min(learner_versions), "max": max(learner_versions)}
                if learner_versions
                else None
            ),
        },
        "sampling_vs_learner": sampling,
        "off_policy_by_staleness": off_policy_by_staleness,
        "off_policy_by_staleness_partial_steps": off_policy_by_staleness_partial_steps,
        "tokens": {
            "action_tokens_total": sum(action_tokens),
            "action_tokens_mean_per_step": fmean(action_tokens) if action_tokens else None,
            "model_tokens_total": sum(model_tokens),
            "loss_action_tokens_total": sum(loss_action_tokens),
            "forward_tokens_total": sum(forward_tokens),
            "microbatches_per_step": _distribution([float(value) for value in microbatch_counts]),
        },
        "drops": {
            "stale": len(kinds["stale_group_dropped"]),
            "zero_std_by_reason": dict(sorted(zero_std_dropped_by_reason.items())),
            "zero_std_by_stage": dict(sorted(zero_std_dropped_by_stage.items())),
            "rollout_failed": len(kinds["rollout_failed"]),
            "train_skipped": len(kinds["train_skipped"]),
            "fidelity_violations": len(kinds["fidelity_violation"]),
            "fidelity_dropped_trajectories": fidelity_dropped_trajectories,
            "lease_unavailable": len(kinds["lease_unavailable"]),
            "no_progress_timeout": len(kinds["no_progress_timeout"]),
        },
        "replica": {
            "by_url": {
                url: {"load_count": len(loads), "load_s_mean": fmean(loads) if loads else None}
                for url, loads in sorted(replica_loads.items())
            },
            "unhealthy_count": len(kinds["replica_unhealthy"]),
        },
    }


def main(run_dir: Path, *, as_json: bool = False) -> int:
    report = analyze(run_dir)
    if as_json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        sections = {
            "rollout": "rollout",
            "learner": "learner",
            "publish": "publish",
            "versions": "versions",
            "sampling_vs_learner": "sampling-vs-learner",
            "off_policy_by_staleness": "off-policy-by-staleness",
            "tokens": "tokens",
            "drops": "drops",
            "replica": "replica",
        }
        for key, value in report.items():
            if key in sections:
                print(f"[{sections[key]}]")
            if isinstance(value, float):
                rendered = f"{value:.6f}"
            elif isinstance(value, dict):
                rendered = json.dumps(value, ensure_ascii=False)
            else:
                rendered = str(value)
            print(f"{key:34} {rendered}")
    return 0


__all__ = ["analyze", "main"]
