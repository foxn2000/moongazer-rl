"""``lineage.json``: where a run's config came from, and its resume history.

Deliberately small and explicit — a dict of plain JSON values, no scheduling or retry
logic. Every write goes through ``records.write_json_atomic`` so a reader never sees a
half-written file.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from moongazer_rl.logging.records import write_json_atomic

LINEAGE_FILE = "lineage.json"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _lineage_path(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    return run_dir if run_dir.is_file() else run_dir / LINEAGE_FILE


def init_lineage(
    run_dir: str | Path,
    *,
    config_hash: str,
    parent_run: str | Path | None = None,
) -> dict[str, Any]:
    """Write a fresh ``lineage.json``: the config's identity, the parent run this run
    was started from (if any), and an empty resume history."""
    run_dir = Path(run_dir)
    lineage: dict[str, Any] = {
        "config_hash": config_hash,
        "parent_run": str(parent_run) if parent_run is not None else None,
        "created_at": _now_iso(),
        "resume_history": [],
    }
    write_json_atomic(run_dir / LINEAGE_FILE, lineage)
    return lineage


def record_resume(
    run_dir: str | Path,
    *,
    from_step: int | None,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Append one resume attempt to ``lineage.json``'s ``resume_history`` and rewrite it
    atomically. Requires ``init_lineage`` to have run first."""
    run_dir = Path(run_dir)
    lineage = read_lineage(run_dir)
    lineage.setdefault("resume_history", []).append(
        {
            "resumed_at": _now_iso(),
            "from_step": from_step,
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
        }
    )
    write_json_atomic(run_dir / LINEAGE_FILE, lineage)
    return lineage


def read_lineage(run_dir: str | Path) -> dict[str, Any]:
    """Read back ``lineage.json``. Accepts a run directory or the file itself."""
    path = _lineage_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"no lineage.json under {run_dir}; call init_lineage first")
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["init_lineage", "read_lineage", "record_resume"]
