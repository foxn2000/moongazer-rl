"""``RunLogger``: the run directory (layout in docs/run-directory.md) as one object.

Writes/append-only-logs config, runtime metadata, rollouts/model-calls/traces/steps/events,
and keeps ``summary.json`` up to date. Resume repairs at most a truncated *last* line of
each jsonl log (an interrupted append) the same way ``vf_runner.results.recover_trace_tail``
does: the damaged tail is quarantined to ``<name>.partial.<UTC>.json`` next to the file,
never silently dropped, and a bad line anywhere *before* the last one raises instead of
truncating real data.

This module must stay importable without torch: every heavy/optional package (torch,
verifiers, trl, transformers, unsloth, git) is only ever imported lazily, inside
``runtime_info()``, so a bare ``import moongazer_rl.logging.run_dir`` never pulls torch
into ``sys.modules``.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass, field, is_dataclass
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import yaml

from moongazer_rl.logging import metrics as metrics_mod
from moongazer_rl.logging import records as records_mod
from moongazer_rl.rollout.schema import RolloutGroup
from moongazer_rl.trainer.base import StepResult

CONFIG_FILE = "config.yaml"
CONFIG_RESOLVED_FILE = "config.resolved.yaml"
RUNTIME_FILE = "runtime.json"
ROLLOUTS_FILE = "rollouts.jsonl"
MODEL_CALLS_FILE = "model_calls.jsonl"
TRACES_FILE = "traces.jsonl"
STEPS_FILE = "steps.jsonl"
EVENTS_FILE = "events.jsonl"
SUMMARY_FILE = "summary.json"
CHECKPOINTS_DIR = "checkpoints"

_JSONL_FILES = (ROLLOUTS_FILE, MODEL_CALLS_FILE, TRACES_FILE, STEPS_FILE, EVENTS_FILE)

_SUMMARY_WINDOW_STEPS = 20
"""How many trailing steps' rollouts feed the "recent" running reward mean in
summary.json, alongside the mean over the whole run."""

_RUNTIME_PACKAGES = (
    "verifiers",
    "renderers",
    "trl",
    "transformers",
    "datasets",
    "torch",
    "peft",
    "unsloth",
    "unsloth_zoo",
)


# --- resume ----------------------------------------------------------------------------


@dataclass
class ResumeState:
    """What ``open_for_resume`` found already on disk."""

    last_step: int | None
    last_weight_version: int | None
    num_rollout_rows: int
    checkpoints: list[Path] = field(default_factory=list)


def _is_loadable_json_line(raw: bytes) -> bool:
    if not raw.strip():
        return True
    try:
        json.loads(raw)
    except Exception:
        return False
    return True


def _scan_to_last_line(path: Path) -> tuple[int, bytes] | None:
    """``(byte offset, raw final line)``, validating every earlier line along the way.
    Raises ``ValueError`` on damage before the last line. ``None`` for an empty file."""
    pending: tuple[int, bytes] | None = None
    offset = 0
    with path.open("rb") as f:
        for raw in f:
            if pending is not None:
                start, previous = pending
                if not _is_loadable_json_line(previous):
                    raise ValueError(
                        f"{path}: malformed line at byte offset {start} is not the last "
                        "line; refusing to truncate (only an interrupted final append is "
                        "recoverable)"
                    )
            pending = (offset, raw)
            offset += len(raw)
    return pending


def _repair_jsonl_tail(path: Path) -> Path | None:
    """Quarantine and drop a half-written final line of ``path``, if there is one.

    Returns the quarantine file's path when a tail was recovered, or ``None`` when
    there was nothing to do (missing/empty file, or the file already ends cleanly).
    """
    if not path.exists():
        return None
    last = _scan_to_last_line(path)
    if last is None:
        return None
    start, raw = last
    if raw.endswith(b"\n") and _is_loadable_json_line(raw):
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    quarantine = path.parent / f"{path.stem}.partial.{stamp}.json"
    quarantine.write_bytes(raw)
    with path.open("r+b") as f:
        f.truncate(start)
    return quarantine


# --- runtime info ------------------------------------------------------------------------


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def _cuda_available() -> bool | None:
    try:
        import torch  # local, lazy: never imported at module load time
    except Exception:
        return None
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return None


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _RUNTIME_PACKAGES:
        try:
            versions[name] = importlib_metadata.version(name)
        except Exception:
            continue
    return versions


def runtime_info(**extra: Any) -> dict[str, Any]:
    """Collect versions/platform/git/CUDA facts for ``runtime.json``. Never raises:
    every individual probe (a package's version, git sha, CUDA availability, hostname)
    is best-effort and silently omitted/``None`` on failure. ``**extra`` (e.g. a
    tokenizer or chat-template fingerprint, added later by the trainer) is merged in
    verbatim and wins over any key computed here."""
    info: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": _package_versions(),
        "git_sha": _git_sha(),
        "cuda_available": _cuda_available(),
    }
    try:
        info["hostname"] = socket.gethostname()
    except Exception:
        info["hostname"] = None
    info.update(extra)
    return info


# --- RunLogger ---------------------------------------------------------------------------


class RunLogger:
    """Bound to one run directory. Construct via ``create`` (new run) or
    ``open_for_resume`` (existing run); the plain constructor assumes the directory
    already has its config files."""

    def __init__(self, run_dir: str | Path, *, save_token_ids: bool = False) -> None:
        self.run_dir = Path(run_dir)
        self.save_token_ids = save_token_ids
        self._summary: _SummaryState | None = None
        """Running totals behind ``summary.json``; loaded from the jsonl logs once (lazily),
        then updated incrementally so ``log_step`` stays O(1) instead of re-reading the
        whole history every step."""

    def _state(self) -> _SummaryState:
        if self._summary is None:
            self._summary = _SummaryState.load(self.run_dir)
        return self._summary

    def _path(self, name: str) -> Path:
        return self.run_dir / name

    # -- construction --

    @classmethod
    def create(
        cls,
        run_dir: str | Path,
        *,
        config_raw_text: str,
        resolved: dict[str, Any],
        runtime_info: dict[str, Any],
        overwrite: bool = False,
        save_token_ids: bool = False,
    ) -> RunLogger:
        """Initialize a fresh run directory: ``config.yaml``, ``config.resolved.yaml``,
        ``runtime.json``, and a ``run_started`` event. Raises ``FileExistsError`` if the
        directory already holds a run (has ``config.yaml``) and ``overwrite`` is False."""
        run_dir = Path(run_dir)
        if (run_dir / CONFIG_FILE).exists() and not overwrite:
            raise FileExistsError(
                f"{run_dir} already contains a run (config.yaml); pass overwrite=True to reuse it"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / CONFIG_FILE).write_text(config_raw_text, encoding="utf-8")
        (run_dir / CONFIG_RESOLVED_FILE).write_text(
            yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        records_mod.write_json_atomic(run_dir / RUNTIME_FILE, runtime_info)
        logger = cls(run_dir, save_token_ids=save_token_ids)
        logger.event("run_started", run_name=run_dir.name)
        logger.write_summary(status="running")
        return logger

    @classmethod
    def open_for_resume(
        cls,
        run_dir: str | Path,
        *,
        save_token_ids: bool = False,
    ) -> tuple[RunLogger, ResumeState]:
        """Validate an existing run directory, repair a truncated last line of each
        jsonl log (never a mid-file one — that raises), and return the bound logger plus
        what was found (last step/weight version, rollout count, checkpoints)."""
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"not a directory: {run_dir}")
        if not (run_dir / CONFIG_FILE).exists():
            raise FileNotFoundError(f"not a run directory (missing config.yaml): {run_dir}")

        repaired: dict[str, str] = {}
        for name in _JSONL_FILES:
            quarantine = _repair_jsonl_tail(run_dir / name)
            if quarantine is not None:
                repaired[name] = quarantine.name

        logger = cls(run_dir, save_token_ids=save_token_ids)

        steps = records_mod.read_jsonl(run_dir / STEPS_FILE)
        last_step = steps[-1]["step"] if steps else None
        last_weight_version = steps[-1].get("weight_version") if steps else None
        num_rollout_rows = sum(1 for _ in records_mod.iter_jsonl(run_dir / ROLLOUTS_FILE))
        checkpoints_dir = run_dir / CHECKPOINTS_DIR
        checkpoints = sorted(checkpoints_dir.iterdir()) if checkpoints_dir.is_dir() else []

        state = ResumeState(
            last_step=last_step,
            last_weight_version=last_weight_version,
            num_rollout_rows=num_rollout_rows,
            checkpoints=checkpoints,
        )
        if repaired:
            logger.event("resume_repair", files=repaired)
        logger.event("run_resumed", last_step=last_step, num_rollout_rows=num_rollout_rows)
        # A resumed run is running again; otherwise summary.json keeps saying
        # "completed"/"failed" until the resumed run finishes.
        logger.write_summary(status="running")
        return logger, state

    # -- logging --

    def log_group(
        self,
        group: RolloutGroup,
        *,
        step: int,
        trace_records: list[dict[str, Any]] | None = None,
    ) -> None:
        """Append one ``RolloutGroup``'s data: a rollout row per trajectory, a model-call
        row per call, and (if given) the runtime's own trace records verbatim. Kept
        decoupled from verifiers: ``trace_records`` are plain dicts (``trace.to_record()``
        from the caller), never a verifiers type."""
        rollout_rows = [
            records_mod.rollout_row(group, traj, step=step) for traj in group.trajectories
        ]
        if rollout_rows:
            records_mod.append_jsonl(self._path(ROLLOUTS_FILE), rollout_rows)
            state = self._state()
            for row in rollout_rows:
                state.add_rollout(row)

        call_rows = [
            records_mod.model_call_row(
                group, traj, call, step=step, include_ids=self.save_token_ids
            )
            for traj in group.trajectories
            for call in traj.model_calls
        ]
        if call_rows:
            records_mod.append_jsonl(self._path(MODEL_CALLS_FILE), call_rows)

        if trace_records:
            records_mod.append_jsonl(self._path(TRACES_FILE), trace_records)

    def log_step(
        self,
        result: StepResult,
        groups: list[RolloutGroup],
        *,
        rollout_wall_time_s: float | None = None,
        train_wall_time_s: float | None = None,
        step: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one ``steps.jsonl`` row (``StepResult`` + ``metrics.group_stats(groups)`` +
        throughput) and refresh ``summary.json``. ``step`` is the orchestrator's batch
        position (see ``records.step_row``). Returns the row written."""
        gstats = metrics_mod.group_stats(groups)
        gstats.update(metrics_mod.throughput(groups, rollout_wall_time_s))
        gstats.update(metrics_mod.train_throughput(result, train_wall_time_s))
        row = records_mod.step_row(
            result,
            gstats,
            rollout_wall_time_s=rollout_wall_time_s,
            train_wall_time_s=train_wall_time_s,
            step=step,
        )
        if extra is not None:
            row["async"] = dict(extra)
        records_mod.append_jsonl(self._path(STEPS_FILE), [row])
        self._state().add_step(row)
        self.write_summary()
        return row

    def event(self, kind: str, **data: Any) -> dict[str, Any]:
        row = records_mod.event_row(kind, **data)
        records_mod.append_jsonl(self._path(EVENTS_FILE), [row])
        return row

    def checkpoint_saved(self, info: Any) -> dict[str, Any]:
        """Log a ``checkpoint_saved`` event from a ``CheckpointInfo``-like object (a
        dataclass, or a plain dict with the same fields)."""
        if isinstance(info, dict):
            data = dict(info)
        elif is_dataclass(info) and not isinstance(info, type):
            data = {k: getattr(info, k) for k in info.__dataclass_fields__}
        else:
            data = dict(vars(info))
        if isinstance(data.get("path"), Path):
            data["path"] = str(data["path"])
        return self.event("checkpoint_saved", **data)

    def finish(self, status: str, **data: Any) -> dict[str, Any]:
        """Log a ``run_finished`` event and write the final ``summary.json``."""
        event = self.event("run_finished", status=status, **data)
        self.write_summary(status=status)
        return event

    # -- summary --

    def write_summary(self, *, status: str | None = None) -> dict[str, Any]:
        """Rewrite ``summary.json`` atomically from the running totals: run name, status,
        last step, counts, last step's metrics, the reward mean over every rollout and over
        the last ``_SUMMARY_WINDOW_STEPS`` steps, wall-clock totals, and timestamps. Safe to
        call any time (mid-run after ``log_step``, or once more at ``finish``)."""
        state = self._state()
        if status is not None:
            state.status = status
        summary = state.to_summary(run_name=self.run_dir.name)
        records_mod.write_json_atomic(self._path(SUMMARY_FILE), summary)
        return summary


@dataclass
class _SummaryState:
    """Everything ``summary.json`` needs, kept in memory and updated per row."""

    status: str = "running"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    num_steps: int = 0
    last_step: int | None = None
    last_step_metrics: dict[str, Any] = field(default_factory=dict)
    wall_time_s_total: float = 0.0
    rollout_wall_time_s_total: float = 0.0
    train_wall_time_s_total: float = 0.0
    num_rollouts: int = 0
    reward_sum: float = 0.0
    reward_count: int = 0
    step_order: list[int] = field(default_factory=list)
    """Steps in log order; the recent window is its last ``_SUMMARY_WINDOW_STEPS`` entries."""
    per_step_reward: dict[int, tuple[float, int]] = field(default_factory=dict)
    """step -> (reward sum, reward count) over that step's rollouts."""

    @classmethod
    def load(cls, run_dir: Path) -> _SummaryState:
        """One full scan of the logs (new run: empty files; resume: the whole history)."""
        state = cls()
        summary_path = run_dir / SUMMARY_FILE
        if summary_path.exists():
            try:
                existing = json.loads(summary_path.read_text(encoding="utf-8"))
                state.status = existing.get("status", state.status)
                state.created_at = existing.get("created_at", state.created_at)
            except Exception:  # noqa: BLE001 - a corrupt summary is rebuilt from the logs
                pass
        else:
            events = records_mod.read_jsonl(run_dir / EVENTS_FILE)
            if events and "time" in events[0]:
                state.created_at = events[0]["time"]
        for row in records_mod.iter_jsonl(run_dir / ROLLOUTS_FILE):
            state.add_rollout(row)
        # Replay steps without pruning: the rollout totals for every step are already
        # loaded, and pruning per step here would drop later steps' rewards early.
        for row in records_mod.iter_jsonl(run_dir / STEPS_FILE):
            state.add_step(row, prune=False)
        state.prune_window()
        return state

    def add_rollout(self, row: dict[str, Any]) -> None:
        self.num_rollouts += 1
        reward = row.get("reward")
        if reward is None:
            return
        self.reward_sum += reward
        self.reward_count += 1
        step = row.get("step")
        if step is not None:
            total, count = self.per_step_reward.get(step, (0.0, 0))
            self.per_step_reward[step] = (total + reward, count + 1)

    def add_step(self, row: dict[str, Any], *, prune: bool = True) -> None:
        self.num_steps += 1
        self.last_step = row.get("step")
        self.last_step_metrics = dict(row.get("metrics") or {})
        self.wall_time_s_total += row.get("wall_time_s") or 0.0
        self.rollout_wall_time_s_total += row.get("rollout_wall_time_s") or 0.0
        self.train_wall_time_s_total += row.get("train_wall_time_s") or 0.0
        if self.last_step is not None:
            self.step_order.append(self.last_step)
            if prune:
                self.prune_window()

    def prune_window(self) -> None:
        """Keep per-step reward totals only for the recent window."""
        keep = set(self.step_order[-_SUMMARY_WINDOW_STEPS:])
        for stale in [k for k in self.per_step_reward if k not in keep]:
            self.per_step_reward.pop(stale, None)

    def to_summary(self, *, run_name: str) -> dict[str, Any]:
        recent = {s for s in self.step_order[-_SUMMARY_WINDOW_STEPS:]}
        recent_sum = sum(self.per_step_reward[s][0] for s in recent if s in self.per_step_reward)
        recent_count = sum(self.per_step_reward[s][1] for s in recent if s in self.per_step_reward)
        return {
            "run_name": run_name,
            "status": self.status,
            "last_step": self.last_step,
            "num_steps": self.num_steps,
            "num_rollouts": self.num_rollouts,
            "last_step_metrics": self.last_step_metrics,
            "reward_mean_all_rollouts": (
                self.reward_sum / self.reward_count if self.reward_count else None
            ),
            f"reward_mean_last_{_SUMMARY_WINDOW_STEPS}_steps": (
                recent_sum / recent_count if recent_count else None
            ),
            "wall_time_s_total": self.wall_time_s_total,
            "rollout_wall_time_s_total": self.rollout_wall_time_s_total,
            "train_wall_time_s_total": self.train_wall_time_s_total,
            "created_at": self.created_at,
            "updated_at": datetime.now(UTC).isoformat(),
        }


__all__ = [
    "CHECKPOINTS_DIR",
    "EVENTS_FILE",
    "MODEL_CALLS_FILE",
    "ROLLOUTS_FILE",
    "STEPS_FILE",
    "SUMMARY_FILE",
    "TRACES_FILE",
    "ResumeState",
    "RunLogger",
    "runtime_info",
]
