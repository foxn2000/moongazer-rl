"""Publish exported weights without making the learner wait for server administration."""

from __future__ import annotations

import asyncio
import shutil
import time
from collections.abc import Callable
from typing import Any

from moongazer_rl.async_rl.types import LeasedPolicy, PublishRequest, PublishResult


class WeightPublisher:
    def __init__(
        self,
        policy: LeasedPolicy,
        *,
        loop: asyncio.AbstractEventLoop,
        on_event: Callable[[str, dict], None],
        retries: int,
        retry_backoff_s: float,
    ) -> None:
        if retries < 0 or retry_backoff_s < 0:
            raise ValueError("publish retry settings must be non-negative")
        self.policy = policy
        self.loop = loop
        self.on_event = on_event
        self.retries = retries
        self.retry_backoff_s = retry_backoff_s
        self.failed: BaseException | None = None
        self.failure_event = asyncio.Event()
        self.num_published = 0
        self.num_coalesced = 0
        self.last_latency_s: float | None = None
        self.in_progress = False
        self._pending: PublishRequest | None = None
        self._active_version: int | None = None
        self._ready = asyncio.Event()
        self._stopping = asyncio.Event()
        self._done = asyncio.Event()
        self._accepting = True
        self._running = False
        self._worker: asyncio.Task[PublishResult] | None = None

    @property
    def published_version(self) -> int:
        return self.policy.published_version

    def request(self, req: PublishRequest) -> None:
        """Only the loop mutates pending state; the model-lock holder never waits for it."""
        self.loop.call_soon_threadsafe(self._handoff, req)

    def _handoff(self, req: PublishRequest) -> None:
        try:
            self._receive(req)
        except Exception as exc:
            self.failed = exc
            self.failure_event.set()
            self._accepting = False
            self._stopping.set()
            self._ready.set()

    def _emit(self, kind: str, req: PublishRequest, **data: Any) -> None:
        self.on_event(
            kind,
            {"version": req.version, "publish_id": req.version, "mono": time.monotonic(), **data},
        )

    def _discard(self, req: PublishRequest) -> None:
        # A missing registry entry is not evidence of zero refs. Only delete an export
        # whose ownership the policy can confirm, and never the currently served adapter.
        if req.version == self._active_version:
            return  # A duplicate request must not delete the worker's load source.
        # forget() is the authoritative test (unreferenced, never loaded, not published); the
        # directory goes only when the registry agreed to drop the version.
        if self.policy.forget(req.version):
            shutil.rmtree(req.path, ignore_errors=True)

    def _receive(self, req: PublishRequest) -> None:
        if not self._accepting:
            self._emit("publish_discarded", req)
            self._discard(req)
            return
        self._emit(
            "publish_requested",
            req,
            requested_at=req.requested_at,
            adapter_snapshot_s=(req.snapshot.snapshot_s if req.snapshot is not None else None),
        )
        if self._pending is not None:
            skipped = self._pending
            self.num_coalesced += 1
            self._emit("publish_coalesced", skipped, replacement_version=req.version)
            if skipped.version != req.version:
                self._discard(skipped)
        self._pending = req
        self._ready.set()

    async def run(self) -> None:
        if self._running or self._done.is_set():
            raise RuntimeError("publisher.run() can only be called once")
        self._running = True
        try:
            while True:
                await self._ready.wait()
                self._ready.clear()
                req, self._pending = self._pending, None
                if req is None:
                    if self._stopping.is_set():
                        return
                    continue
                self.in_progress = True
                self._active_version = req.version
                started = time.time()
                self._emit("publish_started", req, lag_s=started - req.requested_at)
                for attempt in range(self.retries + 1):
                    self._worker = asyncio.create_task(asyncio.to_thread(self.policy.publish, req))
                    try:
                        result = await asyncio.shield(self._worker)
                    except asyncio.CancelledError:
                        # Task cancellation is not thread termination: preserve ownership
                        # until HTTP and post-commit cleanup have actually returned.
                        await asyncio.shield(self._worker)
                        raise
                    except Exception as exc:
                        self._emit("publish_failed", req, attempt=attempt + 1, error=str(exc))
                        if attempt == self.retries or self._stopping.is_set():
                            self.failed = exc
                            self.failure_event.set()
                            return
                        try:
                            await asyncio.wait_for(
                                self._stopping.wait(), self.retry_backoff_s * (attempt + 1)
                            )
                        except TimeoutError:
                            pass
                        if self._stopping.is_set():
                            self.failed = exc
                            self.failure_event.set()
                            return
                    else:
                        self.num_published += 1
                        self.last_latency_s = time.time() - req.requested_at
                        self._emit(
                            "publish_finished",
                            req,
                            export_s=result.export_s,
                            load_s=result.load_s,
                            latency_s=self.last_latency_s,
                            lag_s=started - req.requested_at,
                            unloaded_versions=result.unloaded_versions,
                            cleanup_errors=result.cleanup_errors,
                        )
                        for version in result.unloaded_versions:
                            self._emit(
                                "adapter_unloaded", req, version=version, unloaded_version=version
                            )
                        break
                self.in_progress = False
                self._active_version = None
                if self._stopping.is_set() and self._pending is None:
                    return
        except Exception as exc:
            self.failed = exc
            self.failure_event.set()
        finally:
            self._accepting = False
            self.in_progress = False
            self._active_version = None
            self._running = False
            self._done.set()

    async def stop(self, timeout_s: float) -> None:
        """Drain the last accepted export and bound waiting without cancelling its thread."""
        # Flush request callbacks already submitted by a just-completed learner thread.
        await asyncio.sleep(0)
        self._accepting = False
        self._stopping.set()
        self._ready.set()
        if self._pending is not None and (self._done.is_set() or not self._running):
            self._discard(self._pending)
            self._pending = None
        if self._done.is_set() or (not self._running and not self._done.is_set()):
            # Already finished (or never started): wait_for(..., 0.0) would raise TimeoutError.
            return
        await asyncio.wait_for(self._done.wait(), timeout_s)


__all__ = ["WeightPublisher"]
