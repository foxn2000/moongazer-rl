"""Bounded FIFO joining asynchronous rollout producers to the learner consumer."""

from __future__ import annotations

import asyncio
import time
from collections import deque

from moongazer_rl.async_rl.types import QueuedGroup


class QueueClosed(Exception):
    """The queue is closed and cannot accept or return another group."""


class RolloutQueue:
    """A bounded FIFO with condition-based backpressure and wait-time diagnostics.

    ``put`` and ``get`` belong to one event-loop thread; ``close`` may be called from that
    thread or another thread, so wakeups are always scheduled onto the remembered loop.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = int(capacity)
        self._items: deque[QueuedGroup] = deque()
        self._condition = asyncio.Condition()
        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pending_tasks: set[asyncio.Task[None]] = set()
        self.max_occupancy = 0
        self.total_enqueue_wait_s = 0.0
        self.total_dequeue_wait_s = 0.0
        self.num_put = 0
        self.num_get = 0

    @property
    def closed(self) -> bool:
        return self._closed

    def qsize(self) -> int:
        return len(self._items)

    def empty(self) -> bool:
        return not self._items

    def full(self) -> bool:
        return len(self._items) >= self.capacity

    async def put(self, item: QueuedGroup) -> None:
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        started = time.monotonic()
        async with self._condition:
            while self.full() and not self._closed:
                await self._condition.wait()
            if self._closed:
                raise QueueClosed("rollout queue is closed")
            item.enqueued_at = time.time()
            self._items.append(item)
            self.num_put += 1
            self.total_enqueue_wait_s += time.monotonic() - started
            self.max_occupancy = max(self.max_occupancy, len(self._items))
            self._condition.notify_all()

    async def get(self) -> QueuedGroup:
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        started = time.monotonic()
        async with self._condition:
            while self.empty() and not self._closed:
                await self._condition.wait()
            if self.empty():
                raise QueueClosed("rollout queue is closed and empty")
            item = self._items.popleft()
            item.dequeued_at = time.time()
            self.num_get += 1
            self.total_dequeue_wait_s += time.monotonic() - started
            self._condition.notify_all()
            return item

    def close(self) -> None:
        """Close immediately and schedule wakeups for blocked getters and putters."""
        if self._closed:
            return
        self._closed = True

        async def wake_waiters() -> None:
            async with self._condition:
                self._condition.notify_all()

        loop = self._loop
        if loop is None:
            return

        def schedule_wakeup() -> None:
            task = loop.create_task(wake_waiters())
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is loop:
            schedule_wakeup()
        else:
            loop.call_soon_threadsafe(schedule_wakeup)

    def stats(self) -> dict[str, int | float | None]:
        now = time.time()
        oldest_age = None
        if self._items and self._items[0].enqueued_at is not None:
            oldest_age = max(0.0, now - self._items[0].enqueued_at)
        return {
            "capacity": self.capacity,
            "occupancy": len(self._items),
            "max_occupancy": self.max_occupancy,
            "num_put": self.num_put,
            "num_get": self.num_get,
            "mean_enqueue_wait_s": (
                self.total_enqueue_wait_s / self.num_put if self.num_put else 0.0
            ),
            "mean_dequeue_wait_s": (
                self.total_dequeue_wait_s / self.num_get if self.num_get else 0.0
            ),
            "oldest_age_s": oldest_age,
        }


__all__ = ["QueueClosed", "RolloutQueue"]
