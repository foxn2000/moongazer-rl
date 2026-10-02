"""Keep rollout admission, optimizer ownership and shutdown on one event loop."""

from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
import signal
import statistics
import time
import uuid
from collections import deque
from collections.abc import Callable, Coroutine, Sequence
from pathlib import Path
from typing import Any

import yaml

from moongazer_rl.async_rl.evaluator import AsyncEvaluator, eval_max_concurrent
from moongazer_rl.async_rl.publisher import WeightPublisher
from moongazer_rl.async_rl.queue import QueueClosed, RolloutQueue
from moongazer_rl.async_rl.types import (
    AsyncStepStats,
    LeasedPolicy,
    PublishRequest,
    QueuedGroup,
)
from moongazer_rl.config.loader import resolved_dict
from moongazer_rl.config.schema import RunConfig
from moongazer_rl.logging import lineage
from moongazer_rl.logging.records import write_json_atomic
from moongazer_rl.logging.run_dir import RunLogger, runtime_info
from moongazer_rl.orchestrator import (
    RendererFactory,
    TrainOutcome,
    _build_backend,
    _config_hash,
    _config_text,
    _flatten_step_row,
    _latest_checkpoint,
    _prune_checkpoints,
    _read_orchestrator_state,
    _record_effective_config,
    _restore_for_resume,
    _run_dir,
    _wandb_sink,
    default_renderer_factory,
)
from moongazer_rl.policy.vllm_pool import AsyncVllmPolicyPool, LeaseUnavailable
from moongazer_rl.policy.vllm_server_async import AsyncVllmServerPolicy
from moongazer_rl.rollout.convert import fidelity_summary
from moongazer_rl.rollout.runtime import VerifiersRuntime
from moongazer_rl.rollout.schema import RolloutGroup
from moongazer_rl.trainer.base import CheckpointInfo, StepResult, TrainerBackend

PolicyFactory = Callable[
    [TrainerBackend, RunConfig, str, Callable[[PublishRequest], None]], LeasedPolicy
]


class TaskCursor:
    """Rebuild epoch shuffles from the seed so a checkpoint need only store positions."""

    def __init__(
        self,
        tasks: Sequence[Any],
        *,
        seed: int,
        num_epochs: int | None,
        task_cursor: int = 0,
        uncommitted_task_positions: Sequence[int] = (),
    ) -> None:
        if not tasks:
            raise ValueError("taskset loaded no tasks")
        self.tasks = tasks
        self.seed = seed
        self.num_epochs = num_epochs
        self.position = task_cursor
        self.uncommitted = set(uncommitted_task_positions)
        if task_cursor < 0 or any(p < 0 or p >= task_cursor for p in self.uncommitted):
            raise ValueError("invalid checkpoint task positions")
        self._replay = deque(sorted(self.uncommitted))
        self._epoch = -1
        self._order: list[int] = []

    def __iter__(self) -> TaskCursor:
        return self

    def __next__(self) -> tuple[int, Any]:
        if self._replay:
            position = self._replay.popleft()
        else:
            position = self.position
            if self.num_epochs is not None and position >= len(self.tasks) * self.num_epochs:
                raise StopIteration
            self.position += 1
        epoch, offset = divmod(position, len(self.tasks))
        if epoch != self._epoch:
            self._order = list(range(len(self.tasks)))
            random.Random(self.seed + epoch).shuffle(self._order)
            self._epoch = epoch
        self.uncommitted.add(position)
        return position, self.tasks[self._order[offset]]

    def commit(self, position: int) -> None:
        self.uncommitted.discard(position)

    def requeue(self, position: int) -> None:
        """Hand an uncommitted position back so the next ``__next__`` re-samples it first."""
        if position not in self.uncommitted:
            raise ValueError(f"position {position} is not outstanding")
        self._replay.appendleft(position)


def _save_async_checkpoint(
    backend: TrainerBackend, directory: Path, step: int, state: dict[str, Any]
) -> CheckpointInfo:
    """Include sampler progress before rename so weights and replay positions cannot tear."""
    final = directory / f"step-{step}"
    tmp = directory / f"step-{step}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    info = backend.save_checkpoint(tmp)
    state = {
        **state,
        "step": step,
        "learner_global_step": info.global_step,
        "weight_version": info.weight_version,
    }
    (tmp / "orchestrator_state.json").write_text(json.dumps(state), encoding="utf-8")
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    info.path = final
    return info


class _Execution:
    def __init__(
        self, config: RunConfig, backend: TrainerBackend, logger: RunLogger, run_gen: str
    ) -> None:
        if config.async_rl is None:
            raise ValueError("async training requires an async section")
        self.config = config
        self.options = config.async_rl
        self.backend = backend
        self.logger = logger
        self.run_gen = run_gen
        self.loop = asyncio.get_running_loop()
        self.condition = asyncio.Condition()
        self.stopping = asyncio.Event()
        self.queue = RolloutQueue(self.options.queue_capacity)
        self.reserved_slots = 0
        self._slots: set[int] = set()
        self.error: BaseException | None = None
        self.policy: LeasedPolicy | None = None
        self.publisher: WeightPublisher | None = None
        self.publisher_task: asyncio.Task[None] | None = None
        self.evaluator: AsyncEvaluator | None = None
        self.learner_task: asyncio.Task[None] | None = None
        self.train_future: asyncio.Task[StepResult] | None = None
        self.backend_operation: asyncio.Task[Any] | None = None
        self.learner_failed = False
        self.frozen = False
        self.producers: list[asyncio.Task[None]] = []
        self.cursor: TaskCursor | None = None
        self.steps_completed = 0
        self.last_step = 0
        self.last_checkpoint_step = 0
        self.train_id = 0
        self.requested_version = -1
        self.export_durations: dict[int, float] = {}
        self.snapshot_durations: dict[int, float] = {}
        self.consecutive_failures = 0
        self.consecutive_zero_std_drops = 0
        self.zero_std_dropped = 0
        self.zero_std_dropped_producer = 0
        self.zero_std_dropped_learner = 0
        self.zero_std_dropped_producer_since_last_step = 0
        self.zero_std_dropped_learner_since_last_step = 0
        self.rollouts_since_last_step = 0
        self.first_rollout_finished_mono: float | None = None
        self.last_optimizer_step_mono: float | None = None
        self.train_skipped = 0
        self.active_producers = 0
        self.wandb: Any = None
        self._wake_tasks: set[asyncio.Task[None]] = set()

    async def initialize(self, operation: Coroutine[Any, Any, Any]) -> Any:
        # Cancellation of startup must leave a handle to the model-loading/export worker,
        # just as training and saving do, so shutdown cannot close its resources early.
        self.backend_operation = asyncio.create_task(operation)
        return await asyncio.shield(self.backend_operation)

    def notify(self) -> None:
        async def wake() -> None:
            async with self.condition:
                self.condition.notify_all()

        task = self.loop.create_task(wake())
        self._wake_tasks.add(task)
        task.add_done_callback(self._wake_tasks.discard)

    def stop(self, reason: str = "requested") -> None:
        if not self.stopping.is_set():
            self.logger.event("stop_requested", reason=reason, step=self.last_step)
            self.stopping.set()
        self.queue.close()
        self.notify()

    def handle_signal(self, sig: signal.Signals) -> None:
        """Make repeated shutdown signals escalate without racing the first cleanup pass."""
        if self.stopping.is_set():
            self.logger.event("stop_requested", reason=sig.name, step=self.last_step, repeat=True)
            self.loop.remove_signal_handler(sig)
            signal.signal(sig, signal.SIG_DFL)
            return
        self.stop(sig.name)

    def fail(self, exc: BaseException) -> None:
        if self.error is None:
            self.error = exc
            self.logger.event("async_failed", error=str(exc), step=self.last_step)
        self.stop("failure")

    def export_adapter(self, version: int) -> Path:
        export = getattr(self.backend, "export_adapter", None)
        if not callable(export):
            raise TypeError("async backend requires export_adapter")
        started = time.monotonic()
        exported = export(version)
        duration = time.monotonic() - started
        self.export_durations[version] = duration
        if not isinstance(exported, str | Path):
            # AdapterSnapshot (trainer/unsloth_native/adapter_export.py): the policy's publish
            # worker writes it; only the in-lock snapshot time is the learner's cost.
            if not hasattr(exported, "tensors") or not hasattr(exported, "version"):
                raise TypeError("export_adapter must return a path or an AdapterSnapshot")
            data = {
                "version": version,
                "publish_id": version,
                "adapter_snapshot_s": duration,
                "mono": time.monotonic(),
                "ts": time.time(),
            }
            self.loop.call_soon_threadsafe(lambda: self.logger.event("adapter_snapshot", **data))
            return exported
        path = Path(exported)
        data = {
            "version": version,
            "publish_id": version,
            "adapter_export_s": duration,
            "mono": time.monotonic(),
            "ts": time.time(),
        }
        self.loop.call_soon_threadsafe(lambda: self.logger.event("adapter_exported", **data))
        return path

    def publish_sink(self, req: PublishRequest) -> None:
        # Called under the learner's model lock, before train_step returns.
        self.requested_version = max(self.requested_version, req.version)
        if req.snapshot is not None:
            self.snapshot_durations[req.version] = req.snapshot.snapshot_s
        if self.publisher is None:
            raise RuntimeError("publisher is not initialized")
        self.publisher.request(req)

    def publisher_event(self, kind: str, data: dict) -> None:
        self.logger.event(kind, **data)
        if kind in {"publish_finished", "replica_recovered"}:
            self.notify()

    def seconds_since_last_step(self) -> float:
        origin = self.last_optimizer_step_mono or self.first_rollout_finished_mono
        return 0.0 if origin is None else max(0.0, time.monotonic() - origin)

    def record_zero_std_drop(self, *, stage: str) -> None:
        self.zero_std_dropped += 1
        self.consecutive_zero_std_drops += 1
        if stage == "producer":
            self.zero_std_dropped_producer += 1
            self.zero_std_dropped_producer_since_last_step += 1
        else:
            self.zero_std_dropped_learner += 1
            self.zero_std_dropped_learner_since_last_step += 1
        limit = self.options.max_consecutive_zero_std_drops
        if self.consecutive_zero_std_drops % limit != 0:
            return
        action = self.options.zero_std_budget_action
        self.logger.event(
            "zero_std_budget_exhausted",
            count=self.consecutive_zero_std_drops,
            rollouts_since_last_step=self.rollouts_since_last_step,
            seconds_since_last_step=self.seconds_since_last_step(),
            action=action,
        )
        if action == "stop":
            self.fail(
                RuntimeError(
                    "zero-std drop budget exhausted after "
                    f"{self.consecutive_zero_std_drops} consecutive drops"
                )
            )

    def check_no_progress(self) -> bool:
        limit = self.options.max_no_progress_s
        elapsed = self.seconds_since_last_step()
        if limit is None or self.first_rollout_finished_mono is None or elapsed < limit:
            return False
        self.logger.event(
            "no_progress_timeout",
            max_no_progress_s=limit,
            seconds_since_last_step=elapsed,
            rollouts_since_last_step=self.rollouts_since_last_step,
        )
        self.fail(RuntimeError(f"no optimizer step for {elapsed:.3f}s (limit {limit:.3f}s)"))
        return True

    def release_slot(self, position: int, *, commit: bool) -> None:
        if position in self._slots:
            self._slots.remove(position)
            self.reserved_slots -= 1
        if commit:
            if self.cursor is None:
                raise RuntimeError("task cursor is not initialized")
            self.cursor.commit(position)
        self.notify()

    def admission_ready(self) -> bool:
        if self.stopping.is_set():
            return True
        if self.policy is None:
            raise RuntimeError("policy is not initialized")
        # Registry publication follows the learner's version increment. Read it first:
        # reading an old learner version then a newly published version could admit an
        # extra slot during an in-flight step (including K=0).
        published = self.policy.published_version
        learner = self.backend.weight_version
        return (
            learner + self.reserved_slots // self.config.rollout.groups_per_step - published
            <= self.options.max_staleness
        )

    async def producer(self, producer_id: int, runtime: VerifiersRuntime) -> None:
        if self.cursor is None:
            raise RuntimeError("task cursor is not initialized")
        if self.policy is None:
            raise RuntimeError("policy is not initialized")
        position: int | None = None
        policy = self.policy
        try:
            while not self.stopping.is_set():
                entry = next(self.cursor, None)
                if entry is None:
                    return
                position, task = entry
                wait_started = time.monotonic()
                lease = None
                lease_error = None
                try:
                    async with self.condition:
                        try:
                            await self.condition.wait_for(self.admission_ready)
                        finally:
                            self.logger.event(
                                "producer_idle",
                                producer_id=producer_id,
                                task_cursor=position,
                                duration=time.monotonic() - wait_started,
                            )
                        if self.stopping.is_set():
                            return
                        self.reserved_slots += 1
                        self._slots.add(position)
                        try:
                            lease = policy.lease()
                        except LeaseUnavailable as exc:
                            lease_error = exc
                        learner_version = self.backend.weight_version
                    if lease is None:
                        self.release_slot(position, commit=False)
                        self.cursor.requeue(position)
                        self.logger.event(
                            "lease_unavailable",
                            producer_id=producer_id,
                            task_cursor=position,
                            published_version=policy.published_version,
                            error=str(lease_error),
                        )
                        position = None  # another producer may now own the replayed slot
                        # Probe off-loop and retain the worker through cancellation so shutdown
                        # cannot close the policy under its HTTP operation.
                        check_server = getattr(policy, "check_server", None)
                        if not callable(check_server):
                            raise TypeError("lease recovery requires policy.check_server")
                        probe = asyncio.create_task(asyncio.to_thread(check_server))
                        try:
                            try:
                                await asyncio.shield(probe)
                            except asyncio.CancelledError:
                                await asyncio.gather(probe, return_exceptions=True)
                                raise
                        except Exception:
                            # A failed probe consumes this same failure, not an extra one.
                            pass
                        self.consecutive_failures += 1
                        if (
                            self.consecutive_failures
                            >= self.options.max_consecutive_rollout_failures
                        ):
                            raise RuntimeError(
                                "consecutive rollout failures reached limit"
                            ) from lease_error
                        continue
                    group_id = f"{self.run_gen}:{position}"
                    started_at = time.time()
                    try:
                        self.logger.event(
                            "rollout_started",
                            group_id=group_id,
                            task_idx=task.data.idx,
                            task_cursor=position,
                            producer_id=producer_id,
                            behavior_version=lease.version,
                            learner_version=learner_version,
                            step=self.last_step,
                        )
                        group = await runtime.rollout_group(
                            task,
                            self.config.rollout.num_generations,
                            group_id=group_id,
                            policy_version=lease.version,
                        )
                        if group.trajectories and all(t.has_error for t in group.trajectories):
                            errors = [str(t.errors[0]) for t in group.trajectories if t.errors]
                            raise RuntimeError("all rollouts failed: " + "; ".join(errors))
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        self.logger.event(
                            "rollout_failed",
                            group_id=group_id,
                            task_cursor=position,
                            producer_id=producer_id,
                            error=str(exc),
                        )
                        self.release_slot(position, commit=True)
                        self.consecutive_failures += 1
                        if (
                            self.consecutive_failures
                            >= self.options.max_consecutive_rollout_failures
                        ):
                            raise RuntimeError(
                                "consecutive rollout failures reached limit"
                            ) from exc
                        continue
                finally:
                    if lease is not None:
                        policy.release(lease)
                assert lease is not None
                self.consecutive_failures = 0
                ended_at = time.time()
                finished_mono = time.monotonic()
                if self.first_rollout_finished_mono is None:
                    self.first_rollout_finished_mono = finished_mono
                self.rollouts_since_last_step += 1
                self.logger.event(
                    "rollout_finished",
                    group_id=group_id,
                    task_cursor=position,
                    behavior_version=lease.version,
                    duration=ended_at - started_at,
                    num_tokens=sum(t.num_output_tokens for t in group.trajectories),
                    num_prompt_tokens=sum(
                        len(call.prompt_ids)
                        for trajectory in group.trajectories
                        for call in trajectory.model_calls
                    ),
                    tokens_per_s=(
                        sum(t.num_output_tokens for t in group.trajectories)
                        / (ended_at - started_at)
                        if ended_at > started_at
                        else None
                    ),
                )
                fidelity = fidelity_summary(group)
                if fidelity["num_with_problems"]:
                    self.logger.event(
                        "fidelity_violation",
                        group_id=group_id,
                        task_cursor=position,
                        behavior_version=lease.version,
                        num_trajectories=fidelity["num_trajectories"],
                        num_with_problems=fidelity["num_with_problems"],
                        problems={
                            trajectory_id: trajectory_problems[:5]
                            for trajectory_id, trajectory_problems in fidelity["problems"].items()
                        },
                    )
                would_train = getattr(self.backend, "would_train", None)
                if (
                    self.options.drop_zero_std_groups
                    and getattr(self.backend, "early_drop_safe", False)
                    and callable(would_train)
                    and not would_train(group)
                ):
                    trainable = group.trainable_trajectories
                    few_trainable = len(trainable) < 2
                    zero_variance = (
                        not few_trainable
                        and statistics.pstdev(trajectory.reward for trajectory in trainable) == 0.0
                    )
                    self.logger.event(
                        "zero_std_group_dropped",
                        group_id=group_id,
                        task_cursor=position,
                        behavior_version=lease.version,
                        reward=trainable[0].reward if trainable else None,
                        reason=(
                            "few_trainable"
                            if few_trainable
                            else "zero_variance"
                            if zero_variance
                            else "backend_would_not_train"
                        ),
                        num_trainable=len(trainable),
                        stage="producer",
                    )
                    self.record_zero_std_drop(stage="producer")
                    self.check_no_progress()
                    self.release_slot(position, commit=True)
                    position = None
                    continue
                item = QueuedGroup(
                    group,
                    lease.version,
                    learner_version,
                    started_at,
                    ended_at,
                    producer_id,
                    position,
                    slot_id=position,
                )
                put_started = time.monotonic()
                await self.queue.put(item)
                self.logger.event(
                    "group_enqueued",
                    group_id=group_id,
                    task_cursor=position,
                    queue_size=self.queue.qsize(),
                    wait=time.monotonic() - put_started,
                )
                position = None  # queue/learner now owns the reservation
        except QueueClosed:
            if not self.stopping.is_set():
                self.fail(RuntimeError("queue closed while producers were active"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.fail(exc)
        finally:
            if position is not None:
                self.release_slot(position, commit=False)
            self.active_producers -= 1
            if self.active_producers == 0:
                self.queue.close()

    async def checkpoint(self) -> None:
        if self.frozen:
            return
        if self.learner_failed:
            # A failing step may already have mutated the model (notably export failures).
            # Saving it under the last successful step would corrupt resume correspondence.
            self.logger.event(
                "checkpoint_skipped",
                reason="learner step failed; weights may be partially updated",
                step=self.last_step,
                unpersisted_steps=self.last_step - self.last_checkpoint_step,
            )
            return
        if self.cursor is None:
            raise RuntimeError("task cursor is not initialized")
        if self.policy is None:
            raise RuntimeError("policy is not initialized")
        state = {
            "published_version": self.policy.published_version,
            "task_cursor": self.cursor.position,
            "uncommitted_task_positions": sorted(self.cursor.uncommitted),
            "run_gen": self.run_gen,
        }
        directory = self.logger.run_dir / "checkpoints"
        worker = asyncio.create_task(
            asyncio.to_thread(
                _save_async_checkpoint, self.backend, directory, self.last_step, state
            )
        )
        self.backend_operation = worker
        try:
            info = await asyncio.shield(worker)
        except asyncio.CancelledError:
            await asyncio.shield(worker)
            raise
        if self.frozen:
            return
        self.logger.checkpoint_saved(
            {
                "path": str(info.path),
                "step": self.last_step,
                "global_step": info.global_step,
                "weight_version": info.weight_version,
            }
        )
        self.last_checkpoint_step = self.last_step
        _prune_checkpoints(directory, self.config.checkpoint.keep_last)

    async def learner(self) -> None:
        if self.policy is None:
            raise RuntimeError("policy is not initialized")
        while not self.stopping.is_set():
            maximum = self.config.rollout.max_steps
            if maximum is not None and self.last_step >= maximum:
                return
            batch: list[QueuedGroup] = []
            idle_started = time.monotonic()
            dropped = 0
            queue_size = 0
            while len(batch) < self.config.rollout.groups_per_step and not self.stopping.is_set():
                try:
                    item = await self.queue.get()
                except QueueClosed:
                    break
                item.learner_version_at_consume = self.backend.weight_version
                item.stamp_metadata()
                stale = item.staleness
                if stale is None:
                    raise RuntimeError("staleness is undefined before learner_version_at_consume")
                queue_size = self.queue.qsize()
                self.logger.event(
                    "group_dequeued",
                    group_id=item.group.group_id,
                    task_cursor=item.task_cursor,
                    behavior_version=item.behavior_version,
                    learner_version=item.learner_version_at_consume,
                    staleness=stale,
                    queue_wait_s=item.queue_wait_s,
                    queue_size=queue_size,
                    reason="accepted",
                )
                if self.check_no_progress():
                    break
                if stale > self.options.max_staleness:
                    self.logger.event(
                        "stale_group_dropped",
                        group_id=item.group.group_id,
                        staleness=stale,
                        task_cursor=item.task_cursor,
                        reason="max_staleness_exceeded",
                    )
                    dropped += 1
                    self.release_slot(item.task_cursor, commit=True)
                    continue
                if (
                    any(
                        call.policy_version != item.behavior_version
                        for trajectory in item.group.trajectories
                        for call in trajectory.model_calls
                    )
                    or item.group.policy_version != item.behavior_version
                ):
                    raise RuntimeError(f"policy version invariant violated: {item.group.group_id}")
                trainable = item.group.trainable_trajectories
                few_trainable = len(trainable) < 2
                zero_variance = (
                    not few_trainable
                    and statistics.pstdev(trajectory.reward for trajectory in trainable) == 0.0
                )
                backend_would_train = True
                if self.options.drop_zero_std_groups:
                    would_train = getattr(self.backend, "would_train", None)
                    backend_would_train = (
                        bool(would_train(item.group))
                        if callable(would_train)
                        else not (few_trainable or zero_variance)
                    )
                if self.options.drop_zero_std_groups and not backend_would_train:
                    reward = trainable[0].reward if trainable else None
                    if few_trainable:
                        reason = "few_trainable"
                    elif zero_variance:
                        reason = "zero_variance"
                    else:
                        reason = "backend_would_not_train"
                    self.logger.event(
                        "zero_std_group_dropped",
                        group_id=item.group.group_id,
                        task_cursor=item.task_cursor,
                        behavior_version=item.behavior_version,
                        reward=reward,
                        reason=reason,
                        num_trainable=len(trainable),
                        stage="learner",
                    )
                    self.record_zero_std_drop(stage="learner")
                    self.release_slot(item.task_cursor, commit=True)
                    continue
                batch.append(item)
            idle_s = time.monotonic() - idle_started
            self.logger.event("learner_idle", duration=idle_s, step=self.last_step + 1)
            if not batch or self.stopping.is_set():
                return
            before = self.backend.weight_version
            for item in batch:
                stale = before - item.behavior_version
                if not 0 <= stale <= self.options.max_staleness:
                    raise RuntimeError(f"policy version invariant violated: {item.group.group_id}")
            groups = [item.group for item in batch]
            timing: dict[str, float] = {}
            step_before = self.backend.global_step
            self.train_id += 1
            train_id = self.train_id

            def execute(
                batch: list[QueuedGroup],
                groups: list[RolloutGroup],
                before: int,
                timing: dict[str, float],
                train_id: int,
            ) -> StepResult:
                timing.update(started_at=time.time(), started_mono=time.monotonic())
                event = {
                    "train_id": train_id,
                    "step": self.last_step + 1,
                    "learner_version": before,
                    "behavior_versions": [item.behavior_version for item in batch],
                    "group_ids": [g.group_id for g in groups],
                    "ts": timing["started_at"],
                    "mono": timing["started_mono"],
                }
                self.loop.call_soon_threadsafe(lambda: self.logger.event("train_started", **event))
                try:
                    return self.backend.train_step(groups)
                finally:
                    timing.update(ended_at=time.time(), ended_mono=time.monotonic())

            self.train_future = asyncio.create_task(
                asyncio.to_thread(execute, batch, groups, before, timing, train_id)
            )
            self.backend_operation = self.train_future
            try:
                try:
                    result = await asyncio.shield(self.train_future)
                except asyncio.CancelledError as exc:
                    self.fail(exc)
                    result = await asyncio.shield(self.train_future)
            except BaseException as exc:
                self.learner_failed = True
                if self.frozen:
                    return
                self.logger.event(
                    "train_failed",
                    train_id=train_id,
                    step=self.last_step + 1,
                    group_ids=[g.group_id for g in groups],
                    error=str(exc),
                    learner_version=self.backend.weight_version,
                    train_started_mono=timing.get("started_mono"),
                    train_ended_mono=timing.get("ended_mono"),
                )
                raise
            if self.frozen:
                return
            train_s = timing["ended_mono"] - timing["started_mono"]
            if result.global_step == step_before:
                if result.weight_version != before:
                    raise RuntimeError("backend changed weight_version without an optimizer step")
                reason = str(result.extra.get("skipped", "no optimizer step"))
                self.logger.event(
                    "train_skipped",
                    train_id=train_id,
                    step=self.last_step + 1,
                    reason=reason,
                    group_ids=[g.group_id for g in groups],
                    learner_version=before,
                    duration=train_s,
                    train_started_mono=timing["started_mono"],
                    train_ended_mono=timing["ended_mono"],
                    mono=timing["ended_mono"],
                    ts=timing["ended_at"],
                )
                self.train_skipped += 1
                for item in batch:
                    self.release_slot(item.task_cursor, commit=True)
                self.notify()
                continue
            self.logger.event(
                "train_finished",
                train_id=train_id,
                step=self.last_step + 1,
                learner_version=result.weight_version,
                group_ids=[g.group_id for g in groups],
                duration=train_s,
                train_started_mono=timing["started_mono"],
                train_ended_mono=timing["ended_mono"],
                mono=timing["ended_mono"],
                ts=timing["ended_at"],
                **{
                    key: result.metrics.get(key)
                    for key in (
                        "loss",
                        "grad_norm",
                        "importance_ratio_mean",
                        "importance_ratio_p50",
                        "importance_ratio_capped_fraction",
                        "sampling_vs_learner_logprob_diff_mean",
                        "importance_ratio_p95",
                        "importance_ratio_p99",
                        "behavior_log_ratio_mean",
                        "behavior_log_ratio_std",
                        "behavior_log_ratio_abs_p99",
                        "behavior_ratio_raw_p50",
                        "behavior_ratio_raw_p95",
                        "behavior_ratio_raw_p99",
                        "behavior_ratio_raw_max",
                        "behavior_ratio_raw_mean",
                        "behavior_ess_relative",
                        "behavior_ess_relative_used",
                        "proximal_ratio_identity",
                        "num_loss_action_tokens",
                        "num_forward_tokens",
                        "time/score_forward_s",
                        "time/forward_backward_s",
                        "time/optimizer_s",
                    )
                },
                num_action_tokens=result.num_action_tokens,
                num_tokens=result.metrics.get("num_tokens"),
                num_micro_batches=len(result.extra.get("microbatches", [])),
                off_policy_by_staleness=result.extra.get("off_policy_by_staleness"),
            )
            for item in batch:
                self.release_slot(item.task_cursor, commit=True)
            self.last_step = result.global_step
            self.steps_completed += result.global_step - step_before
            stats = AsyncStepStats(
                behavior_versions=[item.behavior_version for item in batch],
                staleness_values=[before - item.behavior_version for item in batch],
                queue_wait_s=[item.queue_wait_s or 0.0 for item in batch],
                queue_size_at_dequeue=queue_size,
                learner_idle_s=idle_s,
                num_stale_dropped=dropped,
                published_version=self.policy.published_version,
                learner_version_before=before,
                learner_version_after=result.weight_version,
                train_started_at=timing["started_at"],
                train_ended_at=timing["ended_at"],
                adapter_export_s=self.export_durations.pop(
                    result.weight_version, result.extra.get("adapter_export_s")
                ),
                num_zero_std_dropped=(
                    self.zero_std_dropped_producer_since_last_step
                    + self.zero_std_dropped_learner_since_last_step
                ),
                num_zero_std_dropped_producer=self.zero_std_dropped_producer_since_last_step,
                num_zero_std_dropped_learner=self.zero_std_dropped_learner_since_last_step,
                rollouts_since_last_step=self.rollouts_since_last_step,
                adapter_snapshot_s=self.snapshot_durations.pop(result.weight_version, None),
            )
            row = self.logger.log_step(
                result,
                groups,
                rollout_wall_time_s=(
                    max(item.rollout_ended_at for item in batch)
                    - min(item.rollout_started_at for item in batch)
                ),
                train_wall_time_s=train_s,
                step=self.last_step,
                extra=stats.to_row(),
            )
            if self.evaluator is not None:
                self.evaluator.maybe_schedule(self.last_step, result.weight_version)
            if self.wandb is not None:
                self.wandb.log(self.last_step, _flatten_step_row(row))
            self.consecutive_zero_std_drops = 0
            self.zero_std_dropped_producer_since_last_step = 0
            self.zero_std_dropped_learner_since_last_step = 0
            self.rollouts_since_last_step = 0
            self.last_optimizer_step_mono = timing["ended_mono"]
            self.notify()
            if result.weight_version > before and self.requested_version < result.weight_version:
                raise RuntimeError(
                    "backend.train_step advanced weight_version without calling the publish sink"
                )
            if (
                self.config.checkpoint.save_steps
                and self.last_step > self.last_checkpoint_step
                and self.last_step % self.config.checkpoint.save_steps == 0
            ):
                await self.checkpoint()

    async def run(self, runtime: VerifiersRuntime) -> None:
        if self.publisher is None:
            raise RuntimeError("publisher is not initialized")
        self.publisher_task = asyncio.create_task(self.publisher.run())
        self.active_producers = self.options.num_producers
        self.producers = [
            asyncio.create_task(self.producer(i, runtime)) for i in range(self.active_producers)
        ]
        self.learner_task = asyncio.create_task(self.learner())
        failure = asyncio.create_task(self.publisher.failure_event.wait())
        stopped = asyncio.create_task(self.stopping.wait())
        try:
            await asyncio.wait(
                [self.learner_task, failure, stopped], return_when=asyncio.FIRST_COMPLETED
            )
            if self.publisher.failed is not None:
                self.fail(self.publisher.failed)
            if self.learner_task.done():
                try:
                    self.learner_task.result()
                except Exception as exc:
                    self.fail(exc)
        finally:
            failure.cancel()
            stopped.cancel()
            await asyncio.gather(failure, stopped, return_exceptions=True)
            await self.shutdown()

    async def shutdown(self) -> None:
        self.stop("shutdown")
        deadline = self.loop.time() + self.options.shutdown_timeout_s

        async def bounded(awaitable: Any) -> Any:
            return await asyncio.wait_for(
                asyncio.shield(awaitable), max(0.0, deadline - self.loop.time())
            )

        try:
            for producer in self.producers:
                producer.cancel()
            await bounded(asyncio.gather(*self.producers, return_exceptions=True))
            if self.learner_task is not None:
                try:
                    await bounded(self.learner_task)
                except Exception as exc:
                    if isinstance(exc, TimeoutError):
                        raise
                    self.fail(exc)
            if self.backend_operation is not None and not self.backend_operation.done():
                try:
                    await bounded(self.backend_operation)
                except TimeoutError:
                    raise
                except Exception as exc:
                    self.fail(exc)
            if self.evaluator is not None:
                await bounded(asyncio.create_task(self.evaluator.stop()))
            if self.config.checkpoint.save_steps and self.last_step > self.last_checkpoint_step:
                try:
                    await bounded(asyncio.create_task(self.checkpoint()))
                except TimeoutError:
                    raise
                except Exception as exc:
                    self.fail(exc)
            for slot in list(self._slots):
                self.release_slot(slot, commit=False)
            if self.publisher is not None:
                await self.publisher.stop(max(0.0, deadline - self.loop.time()))
                if self.publisher.failed is not None:
                    self.fail(self.publisher.failed)
            if self.publisher_task is not None:
                try:
                    await bounded(self.publisher_task)
                except TimeoutError:
                    raise
                except Exception as exc:
                    self.fail(exc)
            if self.policy is not None:
                try:
                    await bounded(asyncio.create_task(self.policy.close()))
                except TimeoutError:
                    raise
                except Exception as exc:
                    self.fail(exc)
        except TimeoutError as exc:
            self.frozen = True
            self.logger.event("shutdown_timeout", timeout_s=self.options.shutdown_timeout_s)
            self.fail(exc)
            # A Python thread cannot be killed safely. Never close model/policy underneath
            # it. The caller gets a failure; process supervisors own hard termination.
            raise
        finally:
            if self._wake_tasks:
                await asyncio.gather(*self._wake_tasks)


async def _train_async(
    config: RunConfig,
    *,
    resume: bool = False,
    backend: TrainerBackend | None = None,
    renderer_factory: RendererFactory = default_renderer_factory,
    policy_factory: PolicyFactory | None = None,
) -> TrainOutcome:
    if config.async_rl is None:
        raise ValueError("async training requires an async section")
    inline_eval_every_steps = config.eval.every_steps
    if inline_eval_every_steps is not None:
        # Persist the behavior the async path actually uses so its resolved config is resumable.
        # Versioned async evaluation is controlled separately by async.eval_every_steps.
        config.eval.every_steps = None
    if config.rollout.num_epochs is None and config.rollout.max_steps is None:
        raise ValueError("set rollout.num_epochs or rollout.max_steps")
    run_dir = _run_dir(config)
    if resume:
        original = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
        if not isinstance(original, dict) or original.get("async") is None:
            raise ValueError("cannot resume a synchronous run with the async orchestrator")
    backend = backend or _build_backend(config)
    run_gen = time.strftime("%Y%m%dT%H%M%S") + uuid.uuid4().hex[:4]

    class AsyncLogger(RunLogger):
        def event(self, kind: str, **data: Any) -> dict[str, Any]:
            return super().event(kind, **{"run_gen": run_gen, "mono": time.monotonic(), **data})

    if resume:
        logger, state = AsyncLogger.open_for_resume(
            run_dir, save_token_ids=config.logging.save_token_ids
        )
        logged_step = state.last_step or 0
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"run directory {run_dir} is not empty; use resume")
        logger = AsyncLogger.create(
            run_dir,
            config_raw_text=_config_text(config),
            resolved=resolved_dict(config),
            runtime_info=runtime_info(CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES")),
            save_token_ids=config.logging.save_token_ids,
        )
        lineage.init_lineage(run_dir, config_hash=_config_hash(config), parent_run=None)
        logged_step = 0
    if inline_eval_every_steps is not None:
        logger.event("inline_eval_disabled", previous_value=inline_eval_every_steps)
    execution = _Execution(config, backend, logger, run_gen)
    installed_signals: dict[signal.Signals, Any] = {}
    status = "failed"
    shutdown_done = False
    try:
        execution.wandb = _wandb_sink(config)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                previous = signal.getsignal(sig)
                execution.loop.add_signal_handler(sig, execution.handle_signal, sig)
            except (NotImplementedError, RuntimeError, ValueError):
                continue
            installed_signals[sig] = previous
        planned = getattr(backend, "set_planned_steps", None)
        if callable(planned):
            planned(config.rollout.max_steps)
        await execution.initialize(asyncio.to_thread(backend.setup))
        checkpoint_state: dict[str, Any] = {}
        if resume:
            execution.last_step = await execution.initialize(
                _restore_for_resume(
                    backend,
                    run_dir,
                    run_dir / "checkpoints",
                    logged_last_step=logged_step,
                    logger=logger,
                )
            )
            execution.last_checkpoint_step = execution.last_step
            latest = _latest_checkpoint(run_dir / "checkpoints")
            if latest is not None:
                checkpoint_state = _read_orchestrator_state(latest)
                if "task_cursor" not in checkpoint_state:
                    raise ValueError("checkpoint has no async task cursor")
                if checkpoint_state.get("weight_version") != backend.weight_version:
                    raise RuntimeError("mismatched checkpoint weight_version")
            _record_effective_config(run_dir, config, logger)
        fingerprint = getattr(backend, "tokenizer_fingerprint", None)
        if callable(fingerprint):
            logger.event("tokenizer_fingerprint", fingerprint=fingerprint())
        optimizer_info = getattr(backend, "optimizer_info", None)
        if callable(optimizer_info):
            logger.event("optimizer", **optimizer_info())
        device_info = getattr(backend, "device_info", None)
        info = json.loads((run_dir / "runtime.json").read_text())
        info.update(device_info() if callable(device_info) else {})
        info["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        write_json_atomic(run_dir / "runtime.json", info)
        if policy_factory is None:
            args = dict(config.policy.args)
            args.setdefault("model", config.model.name)
            is_pool = "base_urls" in args
            policy_cls = AsyncVllmPolicyPool if is_pool else AsyncVllmServerPolicy
            # A backend whose export_adapter returns an AdapterSnapshot (LoRA tensors cloned
            # under the model lock) needs the directory the publisher materializes it into.
            export_root = getattr(backend, "adapter_export_root", None)
            policy = policy_cls(
                export_adapter=execution.export_adapter,
                weight_version=backend.weight_version,
                run_gen=run_gen,
                unload_grace_s=config.async_rl.unload_grace_s,
                publish_sink=execution.publish_sink,
                **({"adapter_export_root": export_root} if export_root is not None else {}),
                **(
                    {
                        "on_event": lambda kind, data: execution.loop.call_soon_threadsafe(
                            execution.publisher_event, kind, data
                        )
                    }
                    if is_pool
                    else {}
                ),
                **args,
            )
            execution.policy = policy
            await execution.initialize(asyncio.to_thread(policy.check_server))
        else:
            execution.policy = policy_factory(backend, config, run_gen, execution.publish_sink)
        policy = execution.policy
        if policy is None:
            raise RuntimeError("policy factory did not initialize a policy")
        execution.publisher = WeightPublisher(
            policy,
            loop=execution.loop,
            on_event=execution.publisher_event,
            retries=config.async_rl.publish_retries,
            retry_backoff_s=config.async_rl.publish_retry_backoff_s,
        )
        attach = getattr(backend, "attach_policy", None)
        initial_publish = getattr(policy, "initial_publish", None)
        if not callable(attach) or not callable(initial_publish):
            raise TypeError("async backend/policy requires attach_policy and initial_publish")
        attach(policy)
        await execution.initialize(asyncio.to_thread(initial_publish, backend.weight_version))
        if policy.published_version != backend.weight_version:
            raise RuntimeError("initial publish did not match restored learner version")
        logger.event(
            "initial_publish",
            version=backend.weight_version,
            publish_id=backend.weight_version,
            published_version=policy.published_version,
            learner_version=backend.weight_version,
        )

        def eval_runtime_factory(on_eval_traces: Any) -> VerifiersRuntime:
            environment = config.environment.to_env_config_dict()
            if config.eval.taskset is not None:
                environment["taskset"] = dict(config.eval.taskset)
            return VerifiersRuntime(
                environment=environment,
                policy=policy,
                renderer=renderer_factory(backend, config),
                model_name=config.model.name,
                sampling=dict(config.rollout.sampling),
                max_concurrent=eval_max_concurrent(config),
                on_traces=on_eval_traces,
            )

        execution.evaluator = AsyncEvaluator(
            policy=policy,
            runtime_factory=eval_runtime_factory,
            logger=logger,
            run_dir=run_dir,
            config=config,
        )

        async def on_traces(group: RolloutGroup, records: list[dict]) -> None:
            # records.model_call_row preserves extra but omits the canonical timestamps.
            # Carry their measured values there until the shared row schema grows them.
            for trajectory in group.trajectories:
                for call in trajectory.model_calls:
                    call.extra.update(
                        started_at=call.started_at, ended_at=call.ended_at, run_gen=run_gen
                    )
            logger.log_group(group, step=backend.weight_version, trace_records=records)

        runtime = VerifiersRuntime(
            environment=config.environment.to_env_config_dict(),
            policy=policy,
            renderer=renderer_factory(backend, config),
            model_name=config.model.name,
            sampling=dict(config.rollout.sampling),
            max_concurrent=config.rollout.max_concurrent,
            on_traces=on_traces,
        )
        async with runtime:
            tasks = runtime.load_tasks()
            logger.event("tasks_loaded", num_tasks=len(tasks), taskset_id=runtime.taskset_id)
            execution.cursor = TaskCursor(
                tasks,
                seed=config.run.seed,
                num_epochs=config.rollout.num_epochs,
                task_cursor=int(checkpoint_state.get("task_cursor", 0)),
                uncommitted_task_positions=checkpoint_state.get("uncommitted_task_positions", []),
            )
            try:
                await execution.run(runtime)
            finally:
                shutdown_done = True
        if execution.error is not None:
            raise execution.error
        status = "completed"
    except BaseException as exc:
        execution.fail(exc)
        raise
    finally:
        try:
            if not shutdown_done:
                await execution.shutdown()
        finally:
            for sig, previous in installed_signals.items():
                execution.loop.remove_signal_handler(sig)
                if previous is not None:
                    signal.signal(sig, previous)
            # A timed-out worker still owns its resources; closing them would race it.
            busy = (
                execution.backend_operation is not None and not execution.backend_operation.done()
            )
            if not busy and not (execution.publisher and execution.publisher.in_progress):
                try:
                    backend.close()
                except Exception as exc:
                    execution.fail(exc)
                    status = "failed"
            if execution.steps_completed == 0 and status == "completed":
                logger.event(
                    "no_optimizer_steps",
                    warning="run completed without an optimizer step",
                )
            logger.finish(
                status,
                steps_completed=execution.steps_completed,
                last_step=execution.last_step,
                reserved_slots=execution.reserved_slots,
                zero_std_dropped=execution.zero_std_dropped,
                zero_std_dropped_producer=execution.zero_std_dropped_producer,
                zero_std_dropped_learner=execution.zero_std_dropped_learner,
                train_skipped=execution.train_skipped,
                queue=execution.queue.stats(),
            )
            if execution.wandb is not None:
                execution.wandb.finish()
    if execution.error is not None:
        raise execution.error
    return TrainOutcome(run_dir, execution.steps_completed, execution.last_step, status)


def train(
    config: RunConfig,
    *,
    resume: bool = False,
    backend: TrainerBackend | None = None,
    renderer_factory: RendererFactory = default_renderer_factory,
    policy_factory: PolicyFactory | None = None,
) -> Path:
    return asyncio.run(
        _train_async(
            config,
            resume=resume,
            backend=backend,
            renderer_factory=renderer_factory,
            policy_factory=policy_factory,
        )
    ).run_dir


__all__ = ["TaskCursor", "train"]
