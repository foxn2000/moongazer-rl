"""Versioned evaluation that runs beside the asynchronous learner."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from moongazer_rl.async_rl.types import Lease, LeasedPolicy
from moongazer_rl.config.schema import RunConfig
from moongazer_rl.logging.metrics import group_stats
from moongazer_rl.logging.records import rollout_row, write_json_atomic
from moongazer_rl.logging.run_dir import RunLogger
from moongazer_rl.rollout.runtime import VerifiersRuntime
from moongazer_rl.rollout.schema import RolloutGroup

TraceCallback = Callable[[RolloutGroup, list[dict]], Awaitable[None] | None]
RuntimeFactory = Callable[[TraceCallback], VerifiersRuntime]


def eval_max_concurrent(config: RunConfig) -> int:
    """Bound evaluation fan-out without adding another configuration field."""
    return min(8, config.rollout.max_concurrent)


class AsyncEvaluator:
    """Run at most one evaluation without pausing the learner.

    ``async.eval_every_steps`` controls scheduling. Evaluation consumes GPU capacity but
    never enters the rollout queue, staleness admission accounting, or learner logs.
    """

    def __init__(
        self,
        *,
        policy: LeasedPolicy,
        runtime_factory: RuntimeFactory,
        logger: RunLogger,
        run_dir: Path,
        config: RunConfig,
    ) -> None:
        self.policy = policy
        self.runtime_factory = runtime_factory
        self.logger = logger
        self.run_dir = Path(run_dir)
        self.config = config
        self.eval_every_steps = (
            config.async_rl.eval_every_steps if config.async_rl is not None else None
        )
        self.max_concurrent = eval_max_concurrent(config)
        self._task: asyncio.Task[None] | None = None
        self._last_scheduled_step: int | None = None

    @property
    def in_progress(self) -> bool:
        return self._task is not None and not self._task.done()

    def maybe_schedule(self, step: int, learner_version: int) -> bool:
        """Lease and schedule an eligible learner step; return whether evaluation started."""
        every = self.eval_every_steps
        if every is None or step <= 0 or step % every != 0:
            return False
        if self.in_progress:
            self.logger.event(
                "eval_skipped",
                reason="in_progress",
                step=step,
                learner_version=learner_version,
            )
            return False
        if self._last_scheduled_step == step:
            return False

        try:
            lease = self.policy.lease()
        except Exception as exc:
            self.logger.event(
                "eval_skipped",
                reason="lease_failed",
                step=step,
                learner_version=learner_version,
                error=str(exc),
            )
            return False
        try:
            self._task = asyncio.create_task(self._evaluate(step, learner_version, lease))
        except BaseException:
            self.policy.release(lease)
            raise
        self._last_scheduled_step = step
        return True

    async def stop(self) -> None:
        """Cancel the active evaluation and wait until its lease has been released."""
        task = self._task
        if task is None:
            return
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _evaluate(self, step: int, learner_version: int, lease: Lease) -> None:
        version = lease.version
        started_at = time.time()
        started_mono = time.monotonic()
        out_dir = self.run_dir / f"eval-v{version}"
        trace_records: list[dict] = []

        async def on_traces(_group: RolloutGroup, records: list[dict]) -> None:
            # This callback is deliberately detached from RunLogger.log_group: evaluation
            # traces can only reach the versioned evaluation directory.
            trace_records.extend(records)

        try:
            runtime = self.runtime_factory(on_traces)
            async with runtime:
                tasks = runtime.load_tasks()
                num_tasks = self.config.eval.num_tasks
                selected = tasks if num_tasks is None else tasks[:num_tasks]
                self.logger.event(
                    "eval_started",
                    step=step,
                    policy_version=version,
                    learner_version=learner_version,
                    num_tasks=len(selected),
                )
                semaphore = asyncio.Semaphore(self.max_concurrent)

                async def run_one(index: int, task: Any) -> RolloutGroup:
                    async with semaphore:
                        return await runtime.rollout_group(
                            task,
                            self.config.rollout.num_generations,
                            group_id=f"eval-v{version}:{index}",
                            policy_version=version,
                        )

                settled = await asyncio.gather(
                    *(run_one(index, task) for index, task in enumerate(selected)),
                    return_exceptions=True,
                )

            groups = [item for item in settled if isinstance(item, RolloutGroup)]
            await asyncio.to_thread(self._write_records, out_dir, groups, trace_records, step)
            failure = next((item for item in settled if isinstance(item, BaseException)), None)
            if failure is not None:
                raise failure

            ended_at = time.time()
            duration_s = time.monotonic() - started_mono
            stats = group_stats(groups)
            summary = {
                "policy_version": version,
                "learner_version_at_schedule": learner_version,
                "step": step,
                "num_tasks": len(selected),
                "num_groups": len(groups),
                "reward_mean": stats.get("reward_mean"),
                "reward_std": stats.get("reward_std"),
                "started_at": started_at,
                "ended_at": ended_at,
                "duration_s": duration_s,
            }
            await asyncio.to_thread(write_json_atomic, out_dir / "summary.json", summary)
            self.logger.event(
                "eval_finished",
                step=step,
                policy_version=version,
                learner_version=learner_version,
                num_tasks=len(selected),
                reward_mean=summary["reward_mean"],
                duration_s=duration_s,
                path=str(out_dir),
            )
        except asyncio.CancelledError:
            self.logger.event(
                "eval_failed",
                step=step,
                policy_version=version,
                learner_version=learner_version,
                error="cancelled",
            )
            raise
        except BaseException as exc:
            self.logger.event(
                "eval_failed",
                step=step,
                policy_version=version,
                learner_version=learner_version,
                error=str(exc),
            )
        finally:
            self.policy.release(lease)

    @staticmethod
    def _write_records(
        out_dir: Path,
        groups: list[RolloutGroup],
        trace_records: list[dict],
        step: int,
    ) -> None:
        AsyncEvaluator._write_jsonl(
            out_dir / "rollouts.jsonl",
            [
                rollout_row(group, trajectory, step=step)
                for group in groups
                for trajectory in group.trajectories
            ],
        )
        AsyncEvaluator._write_jsonl(out_dir / "traces.jsonl", trace_records)

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        """Replace one evaluation artifact so a repeated lease of a version stays coherent."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        temporary.replace(path)


__all__ = ["AsyncEvaluator", "RuntimeFactory", "eval_max_concurrent"]
