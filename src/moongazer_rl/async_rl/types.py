"""Frozen interfaces of the asynchronous path.

Everything here is torch-free and importable on CPU. The other ``async_rl`` modules, the
``policy.vllm_server_async`` policy and the tests all code against these definitions; change them
only together with every consumer.

Vocabulary
----------
behavior version
    LoRA adapter version the rollout was sampled with. One per episode (task x G rollouts), so
    one per ``RolloutGroup`` and per trajectory; it never changes inside a trajectory.
published version
    Version the sampler hands out to *new* leases. Switched atomically by the publisher after
    the adapter for that version is loaded in vLLM.
learner version
    ``TrainerBackend.weight_version``: +1 per optimizer step.
staleness
    ``learner_version_at_consume - behavior_version`` of a group when the learner takes it
    off the queue. Groups with ``staleness > max_staleness`` are dropped, never trained.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from moongazer_rl.trainer.unsloth_native.adapter_export import AdapterSnapshot

from moongazer_rl.policy.base import PolicyBackend
from moongazer_rl.rollout.schema import RolloutGroup


@dataclass(frozen=True)
class Lease:
    """A hold on one adapter version for the lifetime of one episode.

    Taken with ``LeasedPolicy.lease()`` before the first model call of the episode and given
    back with ``release()`` after the last one (also on failure). While at least one lease on a
    version exists the policy must keep that adapter loaded and serve requests for it.
    """

    version: int
    lease_id: int
    acquired_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class PublishRequest:
    """One "make version ``version`` live" request from the learner to the publisher.

    Created by the policy's ``set_weight_version`` (learner thread, after the adapter for
    ``version`` was exported to ``path`` or snapshotted in memory) and handed to the publisher with
    ``loop.call_soon_threadsafe``. Pending requests that no worker has picked up yet may be
    coalesced (latest wins); a request a worker owns is carried to completion.
    """

    version: int
    path: Path
    requested_at: float = field(default_factory=time.time)
    requested_mono: float = field(default_factory=time.monotonic)
    snapshot: AdapterSnapshot | None = None


@dataclass
class PublishResult:
    """Outcome of ``LeasedPolicy.publish(version)``."""

    version: int
    adapter_name: str
    load_s: float
    """Wall time of the vLLM ``load_lora_adapter`` call."""
    unloaded_versions: list[int] = field(default_factory=list)
    """Versions retired by this publish (refcount 0 for at least the grace period, older than
    ``version``)."""
    cleanup_errors: list[str] = field(default_factory=list)
    """Post-commit cleanup failures (unload / rmtree). The publish itself succeeded."""
    export_s: float = 0.0
    """Wall time spent materializing an adapter snapshot before the vLLM load."""


@runtime_checkable
class LeasedPolicy(PolicyBackend, Protocol):
    """A ``PolicyBackend`` that serves several adapter versions at once.

    Thread model: ``lease`` / ``release`` / ``published_version`` are called from the event-loop
    thread (producers). ``set_weight_version`` is called from the learner thread inside
    ``train_step`` (it must only export the adapter and hand a ``PublishRequest`` to the
    publisher, never block on vLLM). ``publish`` is called from the publisher's worker thread.
    The implementation guards its registry with a lock whose critical sections contain state
    changes only: no HTTP, no file I/O, no model lock inside it.

    Refcount of a version = live leases + HTTP requests in flight for it (``generate``
    increments/decrements in try/finally, so a cancelled coroutine still releases). A version
    is unloaded only when its refcount has been 0 for at least the grace period and it is older
    than the published version.
    """

    @property
    def published_version(self) -> int:
        """Version new leases receive."""
        ...

    def lease(self) -> Lease:
        """Reserve the published version for one episode (refcount +1)."""
        ...

    def release(self, lease: Lease) -> None:
        """Give a lease back (refcount -1). Idempotent for the same lease."""
        ...

    def publish(self, request: PublishRequest) -> PublishResult:
        """Blocking (publisher worker thread): materialize a snapshot when present, load the
        exported adapter at ``request.path`` under its versioned name (unloading a stale
        same-name adapter first, so a retry is idempotent), then commit
        ``published_version = request.version`` atomically.

        A load failure raises and leaves the previous published version in place (the caller
        retries). After the commit, cleanup (unload + rmtree of retired versions) runs as a
        separate step whose failures are reported in ``PublishResult.cleanup_errors`` and
        retried on the next ``publish`` or ``prune``; they never roll back the commit."""
        ...

    def prune(self) -> list[int]:
        """Unload (and delete the export of) every version whose refcount has been 0 for the
        grace period and that is older than ``published_version``. Called by the publisher
        after a publish and after the last release of a version. Returns the versions unloaded."""
        ...

    def registered_versions(self) -> dict[int, int]:
        """``{version: refcount}`` of every registered version
        (exported, possibly not yet loaded)."""
        ...

    def forget(self, version: int) -> bool:
        """Forget an unused, unloaded export after its directory was discarded."""
        ...


@dataclass
class QueuedGroup:
    """One ``RolloutGroup`` travelling from a producer to the learner."""

    group: RolloutGroup
    behavior_version: int
    learner_version_at_start: int
    rollout_started_at: float
    rollout_ended_at: float
    producer_id: int
    task_cursor: int
    """Position of the task in the shared task order. Committed (trained / dropped / failed)
    positions leave ``uncommitted_task_positions``; the rest are re-sampled on resume."""
    slot_id: int = -1
    """Admission slot held from admission until the group is trained or discarded."""
    enqueued_at: float | None = None
    dequeued_at: float | None = None
    learner_version_at_consume: int | None = None

    @property
    def rollout_s(self) -> float:
        return self.rollout_ended_at - self.rollout_started_at

    @property
    def queue_wait_s(self) -> float | None:
        if self.enqueued_at is None or self.dequeued_at is None:
            return None
        return self.dequeued_at - self.enqueued_at

    @property
    def staleness(self) -> int | None:
        if self.learner_version_at_consume is None:
            return None
        return self.learner_version_at_consume - self.behavior_version

    def stamp_metadata(self) -> None:
        """Expose async bookkeeping to debuggers and other in-memory consumers only.

        ``rollouts.jsonl`` is written before queueing and does not serialize group metadata;
        dequeue fields are persisted by ``group_dequeued`` and the step's ``async`` object.
        """
        self.group.metadata.update(
            {
                "behavior_version": self.behavior_version,
                "learner_version_at_start": self.learner_version_at_start,
                "rollout_started_at": self.rollout_started_at,
                "rollout_ended_at": self.rollout_ended_at,
                "enqueued_at": self.enqueued_at,
                "dequeued_at": self.dequeued_at,
                "learner_version_at_consume": self.learner_version_at_consume,
                "staleness": self.staleness,
                "producer_id": self.producer_id,
            }
        )


@dataclass
class AsyncStepStats:
    """What the learner loop adds to a ``steps.jsonl`` row (``row["async"]``)."""

    behavior_versions: list[int]
    staleness_values: list[int]
    queue_wait_s: list[float]
    queue_size_at_dequeue: int
    learner_idle_s: float
    """Time the learner waited for its batch (queue empty / stale drops)."""
    num_stale_dropped: int
    published_version: int
    learner_version_before: int
    learner_version_after: int
    train_started_at: float
    train_ended_at: float
    adapter_export_s: float | None = None
    num_zero_std_dropped: int = 0
    num_zero_std_dropped_producer: int = 0
    num_zero_std_dropped_learner: int = 0
    rollouts_since_last_step: int = 0
    adapter_snapshot_s: float | None = None

    def to_row(self) -> dict[str, Any]:
        stale = self.staleness_values
        return {
            "behavior_versions": list(self.behavior_versions),
            "staleness_values": list(stale),
            "staleness_min": min(stale) if stale else None,
            "staleness_max": max(stale) if stale else None,
            "staleness_mean": (sum(stale) / len(stale)) if stale else None,
            "queue_wait_s_mean": (
                sum(self.queue_wait_s) / len(self.queue_wait_s) if self.queue_wait_s else None
            ),
            "queue_size_at_dequeue": self.queue_size_at_dequeue,
            "learner_idle_s": self.learner_idle_s,
            "num_stale_dropped": self.num_stale_dropped,
            "num_zero_std_dropped": self.num_zero_std_dropped,
            "num_zero_std_dropped_producer": self.num_zero_std_dropped_producer,
            "num_zero_std_dropped_learner": self.num_zero_std_dropped_learner,
            "rollouts_since_last_step": self.rollouts_since_last_step,
            "published_version": self.published_version,
            "learner_version_before": self.learner_version_before,
            "learner_version_after": self.learner_version_after,
            "train_started_at": self.train_started_at,
            "train_ended_at": self.train_ended_at,
            "adapter_export_s": self.adapter_export_s,
            "adapter_snapshot_s": self.adapter_snapshot_s,
        }


ORCHESTRATOR_STATE_ASYNC_KEYS = (
    "published_version",
    "task_cursor",
    "uncommitted_task_positions",
    "run_gen",
)
"""Extra keys the async path writes into ``checkpoints/step-<n>/orchestrator_state.json``."""

ADAPTER_EXPORT_DIRNAME = ".lora-sync"
"""``<output_dir>/.lora-sync/v<version>``: where the learner exports adapters (same as sync)."""


def adapter_export_dir(output_dir: str | Path, version: int) -> Path:
    return Path(output_dir) / ADAPTER_EXPORT_DIRNAME / f"v{int(version)}"


def adapter_name(run_gen: str, version: int, prefix: str = "policy") -> str:
    """``policy-<gen>-v<version>``: the vLLM LoRA name. ``run_gen`` changes per process start so
    a resumed run never collides with adapters a previous process left registered."""
    return f"{prefix}-{run_gen}-v{int(version)}"


__all__ = [
    "ADAPTER_EXPORT_DIRNAME",
    "ORCHESTRATOR_STATE_ASYNC_KEYS",
    "AsyncStepStats",
    "Lease",
    "LeasedPolicy",
    "PublishRequest",
    "PublishResult",
    "QueuedGroup",
    "adapter_export_dir",
    "adapter_name",
]
