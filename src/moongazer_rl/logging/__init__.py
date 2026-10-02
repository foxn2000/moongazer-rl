"""Run-directory logging: jsonl rows, aggregate metrics, run directory, lineage.

Importing this package (or any submodule) never imports torch, verifiers, or wandb;
those are all lazy, inside the functions that actually need them (see ``run_dir.py``).
"""

from moongazer_rl.logging.lineage import init_lineage, read_lineage, record_resume
from moongazer_rl.logging.metrics import group_stats, throughput, train_throughput
from moongazer_rl.logging.records import (
    append_jsonl,
    event_row,
    iter_jsonl,
    model_call_row,
    read_jsonl,
    rollout_row,
    step_row,
    write_json_atomic,
)
from moongazer_rl.logging.run_dir import ResumeState, RunLogger, runtime_info
from moongazer_rl.logging.wandb_sink import WandbSink

__all__ = [
    "ResumeState",
    "RunLogger",
    "WandbSink",
    "append_jsonl",
    "event_row",
    "group_stats",
    "init_lineage",
    "iter_jsonl",
    "model_call_row",
    "read_jsonl",
    "read_lineage",
    "record_resume",
    "rollout_row",
    "runtime_info",
    "step_row",
    "throughput",
    "train_throughput",
    "write_json_atomic",
]
