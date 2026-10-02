"""``mg-rl``: the launcher CLI. Thin by design — config parsing/validation lives in
``moongazer_rl.config``; the actual train/eval loops live in ``orchestrator.py`` (owned by
the integrator) and are imported lazily so this module (and ``--help``) works even before that
module exists, or when its heavy deps (torch, unsloth, verifiers plugins) are not installed.
"""

from __future__ import annotations

import argparse
import faulthandler
import importlib
import json
import signal
import sys
from pathlib import Path
from typing import Any

from moongazer_rl.config.loader import ConfigError, load_config
from moongazer_rl.config.validate import validate_config

# --------------------------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------------------------


def _add_set_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="dotted.key=value",
        help="Override a config value (repeatable); value is parsed as YAML",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mg-rl",
        description="Run Prime Intellect verifiers v1 environments with Unsloth/TRL GRPO.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="Validate a run config")
    p_validate.add_argument("config", type=Path)
    _add_set_arg(p_validate)

    p_train = sub.add_parser("train", help="Run training")
    p_train.add_argument("config", type=Path)
    _add_set_arg(p_train)
    p_train.add_argument("--resume", action="store_true", help="Resume from the run's checkpoint")

    p_eval = sub.add_parser("eval", help="Run evaluation only")
    p_eval.add_argument("config", type=Path)
    _add_set_arg(p_eval)

    p_resume = sub.add_parser("resume", help="Resume training from a run directory")
    p_resume.add_argument("run_dir", type=Path)
    _add_set_arg(p_resume)

    p_async_report = sub.add_parser("async-report", help="Summarize an asynchronous run")
    p_async_report.add_argument("run_dir", type=Path)
    p_async_report.add_argument("--json", action="store_true", help="Print raw JSON")

    p_inspect = sub.add_parser("inspect", help="Read-only viewer over a run directory")
    p_inspect.add_argument("run_dir", type=Path)
    p_inspect.add_argument("--rollouts", action="store_true", help="Show rollouts.jsonl rows")
    p_inspect.add_argument("--steps", action="store_true", help="Show steps.jsonl rows")
    p_inspect.add_argument("--events", action="store_true", help="Show events.jsonl rows")
    p_inspect.add_argument("--task", type=int, default=None, help="Filter rollouts by task_idx")
    p_inspect.add_argument(
        "--rollout", type=int, default=None, help="Filter rollouts by rollout_index"
    )
    p_inspect.add_argument(
        "--trajectory", type=str, default=None, help="Filter rollouts by trajectory_id"
    )
    p_inspect.add_argument(
        "--failed-only", action="store_true", help="Only rows with has_error true"
    )
    p_inspect.add_argument(
        "--full", action="store_true", help="Also print the matching trace record(s)"
    )
    p_inspect.add_argument("--last", type=int, default=None, help="Only the last N rows")
    p_inspect.add_argument("--json", action="store_true", help="Print raw JSON instead of a table")

    return parser


# --------------------------------------------------------------------------------------------
# validate / train / eval / resume
# --------------------------------------------------------------------------------------------


def _cmd_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.overrides)
    report = validate_config(config)
    print(report.render())
    return 0 if report.ok else 1


def _cmd_train(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.overrides)
    report = validate_config(config)
    for error in report.errors:
        print(f"ERROR: {error}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    if not report.ok:
        return 1
    if config.async_rl is not None:
        run_train = importlib.import_module("moongazer_rl.async_rl.orchestrator").train
    else:
        from moongazer_rl.orchestrator import train as run_train

    run_dir = run_train(config, resume=args.resume)
    print(f"run directory: {run_dir}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.overrides)
    from moongazer_rl.orchestrator import evaluate

    result = evaluate(config)
    print(json.dumps(result, indent=2, default=str))
    return 0


def _cmd_resume(args: argparse.Namespace) -> int:
    run_dir: Path = args.run_dir
    resolved_path = run_dir / "config.resolved.yaml"
    if not resolved_path.is_file():
        raise ConfigError(f"no config.resolved.yaml found in {run_dir}")
    config = load_config(resolved_path, args.overrides)
    # Resume where the run actually lives (it may have moved, or be given as an absolute
    # path from another cwd), unless the caller overrides logging.output_dir explicitly.
    if not any(o.startswith("logging.output_dir=") for o in args.overrides):
        config.logging.output_dir = str(run_dir.resolve())
    report = validate_config(config)
    for error in report.errors:
        print(f"ERROR: {error}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    # Resume uses the same validation gate as train; invalid restored configs must not start.
    if not report.ok:
        return 1
    if config.async_rl is not None:
        run_train = importlib.import_module("moongazer_rl.async_rl.orchestrator").train
    else:
        from moongazer_rl.orchestrator import train as run_train

    out_dir = run_train(config, resume=True)
    print(f"run directory: {out_dir}")
    return 0


def _cmd_async_report(args: argparse.Namespace) -> int:
    report = importlib.import_module("moongazer_rl.async_rl.report")

    return report.main(args.run_dir, as_json=args.json)


# --------------------------------------------------------------------------------------------
# inspect: read-only viewer over a run directory (docs/run-directory.md)
# --------------------------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Tolerant JSONL reader: a missing file is an empty list (callers print a clear note),
    not an exception. Unknown keys in each row are left as-is (ignored by callers that only
    read the keys they know)."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            msg = f"warning: {path}:{lineno}: skipping malformed JSON line ({exc})"
            print(msg, file=sys.stderr)
    return rows


def _apply_last(rows: list[dict[str, Any]], last: int | None) -> list[dict[str, Any]]:
    return rows[-last:] if last else rows


def _print_json_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(json.dumps(row, default=str))


def _print_table(rows: list[dict[str, Any]], columns: list[str]) -> None:
    if not rows:
        print("(none)")
        return
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("  ".join("-" * widths[c] for c in columns))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns))


def _print_summary(run_dir: Path, args: argparse.Namespace) -> None:
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        try:
            print(json.dumps(json.loads(summary_path.read_text()), indent=2))
        except json.JSONDecodeError as exc:
            print(f"warning: {summary_path}: malformed JSON ({exc})", file=sys.stderr)
    else:
        print(f"(no summary.json in {run_dir})")

    steps = _apply_last(_read_jsonl(run_dir / "steps.jsonl"), args.last or 5)
    if not steps and not summary_path.is_file():
        print(f"(no steps.jsonl in {run_dir})")
        return
    print()
    print(f"last {len(steps)} step(s):")
    _print_table(
        steps,
        ["step", "weight_version", "wall_time_s", "num_groups_used", "num_trajectories_used"],
    )


def _print_steps(run_dir: Path, args: argparse.Namespace) -> None:
    path = run_dir / "steps.jsonl"
    rows = _apply_last(_read_jsonl(path), args.last)
    if not rows and not path.is_file():
        print(f"(no steps.jsonl in {run_dir})")
        return
    if args.json:
        _print_json_rows(rows)
        return
    _print_table(
        rows,
        [
            "step",
            "weight_version",
            "wall_time_s",
            "num_groups_received",
            "num_groups_used",
            "num_trajectories_used",
            "num_sequences_used",
            "num_action_tokens",
        ],
    )


def _matches(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.task is not None and row.get("task_idx") != args.task:
        return False
    if args.rollout is not None and row.get("rollout_index") != args.rollout:
        return False
    if args.trajectory is not None and row.get("trajectory_id") != args.trajectory:
        return False
    if args.failed_only and not row.get("has_error"):
        return False
    return True


def _print_rollouts(run_dir: Path, args: argparse.Namespace) -> None:
    path = run_dir / "rollouts.jsonl"
    rows = _read_jsonl(path)
    if not rows and not path.is_file():
        print(f"(no rollouts.jsonl in {run_dir})")
        return
    filtered = [r for r in rows if _matches(r, args)]
    filtered = _apply_last(filtered, args.last)
    if args.json:
        _print_json_rows(filtered)
    else:
        _print_table(
            filtered,
            [
                "step",
                "task_idx",
                "task_name",
                "trajectory_id",
                "rollout_index",
                "reward",
                "has_error",
                "is_truncated",
                "num_turns",
            ],
        )
    if args.full:
        _print_full_traces(run_dir, {r.get("trajectory_id") for r in filtered})


def _print_events(run_dir: Path, args: argparse.Namespace) -> None:
    path = run_dir / "events.jsonl"
    rows = _apply_last(_read_jsonl(path), args.last)
    if not rows and not path.is_file():
        print(f"(no events.jsonl in {run_dir})")
        return
    _print_json_rows(rows)


def _print_full_traces(run_dir: Path, trajectory_ids: set[Any]) -> None:
    path = run_dir / "traces.jsonl"
    traces = _read_jsonl(path)
    if not traces and not path.is_file():
        print(f"(no traces.jsonl in {run_dir}; cannot show --full)")
        return
    by_id = {t.get("id"): t for t in traces}
    for trajectory_id in trajectory_ids:
        record = by_id.get(trajectory_id)
        print()
        print(f"=== trace {trajectory_id} ===")
        if record is None:
            print("(no matching trace record)")
            continue
        _print_trace_transcript(record)


def _message_text(content: Any) -> str:
    if isinstance(content, list):  # verifiers content parts
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return "" if content is None else str(content)


def _print_trace_transcript(record: dict[str, Any]) -> None:
    """Transcript view of a verifiers ``trace.to_record()`` dict: one line per graph node
    (role, sampled flag, token counts, finish reason) followed by the message text, then
    rewards / metrics / stop condition / errors. Unknown shapes fall back to raw JSON so
    nothing is silently hidden."""
    nodes = record.get("nodes")
    if not (isinstance(nodes, list) and nodes and isinstance(nodes[0], dict)):
        print(json.dumps(record, indent=2, default=str))
        return
    for i, node in enumerate(nodes):
        message = node.get("message") or {}
        role = message.get("role", "?")
        token_ids = node.get("token_ids") or []
        mask = node.get("mask") or []
        tag = "sampled" if node.get("sampled") else "prompt"
        head = (
            f"--- node {i} parent={node.get('parent')} role={role} {tag} "
            f"tokens={len(token_ids)} action={sum(1 for m in mask if m)} "
            f"finish={node.get('finish_reason')}"
        )
        print(head)
        if message.get("reasoning_content"):
            print(f"[{role}:reasoning] {message['reasoning_content']}")
        print(f"[{role}] {_message_text(message.get('content'))}")
        for call in message.get("tool_calls") or []:
            print(f"[{role}:tool_call] {call.get('name')}({call.get('arguments')})")
    print(f"rewards: {json.dumps(record.get('rewards'), default=str)}")
    print(f"metrics: {json.dumps(record.get('metrics'), default=str)}")
    stop = record.get("stop_condition")
    print(f"stop_condition: {stop}  is_completed: {record.get('is_completed')}")
    for error in record.get("errors") or []:
        print(f"error: {error.get('type')}: {error.get('message')}")


def _cmd_inspect(args: argparse.Namespace) -> int:
    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        raise ConfigError(f"no such run directory: {run_dir}")

    want_rollouts = (
        args.rollouts
        or args.task is not None
        or args.rollout is not None
        or args.trajectory is not None
        or args.failed_only
    )
    want_steps = args.steps
    want_events = args.events

    if not (want_rollouts or want_steps or want_events):
        _print_summary(run_dir, args)
        return 0

    if want_rollouts:
        _print_rollouts(run_dir, args)
    if want_steps:
        _print_steps(run_dir, args)
    if want_events:
        _print_events(run_dir, args)
    return 0


# --------------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------------

_HANDLERS = {
    "async-report": _cmd_async_report,
    "validate": _cmd_validate,
    "train": _cmd_train,
    "eval": _cmd_eval,
    "resume": _cmd_resume,
    "inspect": _cmd_inspect,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # `kill -USR1 <pid>` dumps every thread's stack to stderr: the only hang diagnostic that
    # works inside containers where py-spy cannot ptrace.
    if hasattr(signal, "SIGUSR1"):
        try:
            faulthandler.register(signal.SIGUSR1, all_threads=True, chain=True)
        except (RuntimeError, ValueError, OSError):  # not the main thread, or no fd
            pass
    handler = _HANDLERS[args.command]
    try:
        return handler(args)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
