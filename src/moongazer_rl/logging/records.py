"""Pure functions building JSON-safe rows for the run-directory jsonl logs, plus small
jsonl/JSON I/O helpers shared by the rest of ``moongazer_rl.logging``.

Nothing here touches disk except the I/O helpers at the bottom, and nothing here imports
torch, verifiers, or wandb: a row builder takes the canonical objects (``RolloutGroup``,
``Trajectory``, ``ModelCall``, ``StepResult``) already built by upstream code and returns a
plain ``dict`` ready for ``json.dumps``.
"""

from __future__ import annotations

import json
import os
import statistics
import tempfile
import time
from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from moongazer_rl.rollout.schema import ModelCall, RolloutGroup, Trajectory
from moongazer_rl.trainer.base import StepResult

# --- row builders --------------------------------------------------------------------


def rollout_row(group: RolloutGroup, traj: Trajectory, *, step: int) -> dict[str, Any]:
    """One ``rollouts.jsonl`` line: everything the ``inspect`` CLI and a
    later analysis need about one trajectory, flattened out of the canonical objects.

    Key set is frozen; do not rename or drop keys here without checking the
    readers (``cli.py`` inspect).
    """
    first_error = traj.errors[0] if traj.errors else None
    thinking_leak = traj.metadata.get("thinking_leak")
    row: dict[str, Any] = {
        "step": step,
        "group_id": group.group_id,
        "task_id": group.task_id,
        "task_idx": group.task_idx,
        "task_name": group.task_name,
        "trajectory_id": traj.trajectory_id,
        "rollout_index": traj.rollout_index,
        "reward": traj.reward,
        "reward_components": dict(traj.reward_components),
        "metrics": dict(traj.metrics),
        "has_error": traj.has_error,
        "error_type": first_error.type if first_error is not None else None,
        "error_message": first_error.message if first_error is not None else None,
        "is_truncated": traj.is_truncated,
        "stop_condition": traj.stop_condition,
        "is_completed": traj.is_completed,
        "num_turns": traj.num_turns,
        "num_input_tokens": traj.num_input_tokens,
        "num_output_tokens": traj.num_output_tokens,
        "num_model_calls": len(traj.model_calls),
        "num_sequences": len(traj.sequences),
        "num_action_tokens": traj.num_action_tokens,
        "policy_version": group.policy_version,
        "fidelity_problems": list(traj.metadata.get("fidelity_problems", [])),
        "thinking_leak_detected": (
            thinking_leak.get("detected") is True
            if isinstance(thinking_leak, Mapping)
            else False
        ),
        "timing": dict(traj.timing),
        "has_complete_logprobs": traj.has_complete_logprobs,
        "is_trainable": traj.is_trainable,
    }
    return row


def model_call_row(
    group: RolloutGroup,
    traj: Trajectory,
    call: ModelCall,
    *,
    step: int,
    include_ids: bool,
) -> dict[str, Any]:
    """One ``model_calls.jsonl`` line. Token ids/logprobs are included only when
    ``include_ids`` (``RunLogger.save_token_ids``) is set — lengths and logprob summary
    stats are always present regardless, so per-call cost/quality can be read cheaply."""
    logprobs = call.sampling_logprobs
    row: dict[str, Any] = {
        "step": step,
        "group_id": group.group_id,
        "trajectory_id": traj.trajectory_id,
        "call_index": call.index,
        "num_prompt_tokens": len(call.prompt_ids),
        "num_completion_tokens": len(call.completion_ids),
        "finish_reason": call.finish_reason,
        "policy_version": call.policy_version,
        "duration_s": call.duration_s,
        "logprob_mean": statistics.fmean(logprobs) if logprobs else None,
        "logprob_min": min(logprobs) if logprobs else None,
        "has_logprobs": call.has_logprobs,
        "request_id": call.request_id,
        "extra": _json_safe(dict(call.extra)),
    }
    if include_ids:
        row["prompt_ids"] = list(call.prompt_ids)
        row["completion_ids"] = list(call.completion_ids)
        row["sampling_logprobs"] = list(logprobs)
    return row


def step_row(
    result: StepResult,
    group_stats: dict[str, Any],
    *,
    rollout_wall_time_s: float | None,
    train_wall_time_s: float | None,
    step: int | None = None,
) -> dict[str, Any]:
    """One ``steps.jsonl`` line: the trainer's ``StepResult`` plus the group-level
    aggregate stats (``metrics.group_stats``) computed over the groups fed into this step.

    ``step`` is the orchestrator's batch position (the same number stamped on this step's
    rollouts); it defaults to the learner's ``global_step`` but the two diverge when a batch
    had nothing trainable (the learner skips the optimizer step, the batch is still consumed).
    ``learner_global_step`` always records the learner's own counter."""
    return {
        "step": result.global_step if step is None else step,
        "learner_global_step": result.global_step,
        "weight_version": result.weight_version,
        "wall_time_s": result.wall_time_s,
        "rollout_wall_time_s": rollout_wall_time_s,
        "train_wall_time_s": train_wall_time_s,
        "num_groups_received": result.num_groups_received,
        "num_groups_used": result.num_groups_used,
        "num_trajectories_used": result.num_trajectories_used,
        "num_sequences_used": result.num_sequences_used,
        "num_action_tokens": result.num_action_tokens,
        "metrics": dict(result.metrics),
        "group_stats": dict(group_stats),
        "trajectory_ids_used": list(result.trajectory_ids_used),
        "extra": _json_safe(result.extra),
    }


def _json_safe(value: Any) -> Any:
    """Backend diagnostics (``StepResult.extra``) may hold tensors, paths, numpy scalars."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    item = getattr(value, "item", None)  # 0-dim tensors / numpy scalars
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:  # noqa: BLE001 - fall through to str()
            pass
    return str(value)


def event_row(kind: str, **data: Any) -> dict[str, Any]:
    """One ``events.jsonl`` line: lifecycle/error/resume/checkpoint events. ``ts`` is a
    unix float (cheap to sort/diff), ``time`` is the same instant as an ISO-8601 UTC
    string (human-readable). Extra ``**data`` is merged in verbatim."""
    ts = time.time()
    row: dict[str, Any] = {
        "kind": kind,
        "ts": ts,
        "time": datetime.fromtimestamp(ts, tz=UTC).isoformat(),
    }
    row.update(data)
    # Callers may supply the measured event timestamp; keep its human rendering paired with it.
    if row["ts"] != ts and isinstance(row["ts"], int | float):
        row["time"] = datetime.fromtimestamp(row["ts"], tz=UTC).isoformat()
    return row


# --- jsonl / json I/O ------------------------------------------------------------------


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream a jsonl file as dicts, skipping blank lines. Missing file -> no rows."""
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read an entire jsonl file into a list. Missing file -> ``[]``."""
    return list(iter_jsonl(path))


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Append rows to a jsonl file, one JSON object per line. Creates parent dirs and
    the file if needed; never rewrites existing lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _default_file_mode() -> int:
    """The mode an ordinary ``open(...)`` would create a file with, under this umask
    (``mkstemp`` hardcodes 0600, which would make an atomically-replaced artifact file
    less readable than every other file in the run directory)."""
    umask = os.umask(0)  # read-only use; there is no getumask()
    os.umask(umask)
    return 0o666 & ~umask


def write_json_atomic(path: str | Path, obj: Any) -> None:
    """Write ``obj`` as indented JSON to ``path``, atomically (temp file in the same
    directory + ``Path.replace``): a reader never sees a half-written file, and a crash
    mid-write leaves the previous complete file (or none) intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            os.fchmod(f.fileno(), _default_file_mode())
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.write("\n")
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


__all__ = [
    "append_jsonl",
    "event_row",
    "iter_jsonl",
    "model_call_row",
    "read_jsonl",
    "rollout_row",
    "step_row",
    "write_json_atomic",
]
