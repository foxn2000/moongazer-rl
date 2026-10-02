"""Keep ``math_verify`` from stalling the rollout event loop.

Two problems:

1. ``math_verify.utils.timeout`` (used by ``parse`` / ``verify``, hence by verifiers'
   ``verify_boxed_math_answer``) implements its deadline with ``signal.alarm``. Inside a
   long-lived asyncio process a late alarm raises ``TimeoutException`` in whatever frame is
   executing, escapes the caller's ``except`` and kills an unrelated coroutine.
2. Replacing the alarm with a thread deadline stops the stray exception but leaves the
   abandoned sympy thread spinning, and enough of them starve the event loop.

So the symbolic check runs in a **separate worker process** with a hard deadline: on timeout the
worker is killed and replaced, and the caller returns 0.0 exactly as verifiers does for
``MathVerifyTimeout``. The parent never executes ``math_verify`` itself. Workers are started
lazily with the ``spawn`` context (a fresh interpreter, safe next to a CUDA-initialised
learner). ``install_thread_timeout`` is kept as a defence for environments that call
``math_verify`` directly; it never applies inside the workers, which are plain single-threaded
processes where ``signal.alarm`` is fine.
"""

from __future__ import annotations

import atexit
import functools
import multiprocessing as mp
import threading
from collections.abc import Callable
from typing import Any

_INSTALLED_FLAG = "_moongazer_rl_thread_timeout"


def thread_timeout(timeout_seconds: int | None = 10) -> Callable[[Callable], Callable]:
    """Drop-in replacement for ``math_verify.utils.timeout`` without ``signal.alarm``."""
    from math_verify.errors import TimeoutException

    if timeout_seconds is None or timeout_seconds <= 0:
        return lambda func: func

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            outcome: dict[str, Any] = {}

            def run() -> None:
                try:
                    outcome["value"] = func(*args, **kwargs)
                except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
                    outcome["error"] = exc

            worker = threading.Thread(target=run, name="math-verify", daemon=True)
            worker.start()
            worker.join(timeout_seconds)
            if worker.is_alive():
                raise TimeoutException("Operation timed out!")
            if "error" in outcome:
                raise outcome["error"]
            return outcome["value"]

        wrapper.__name__ = getattr(func, "__name__", "wrapper")
        wrapper.__doc__ = getattr(func, "__doc__", None)
        return wrapper

    setattr(decorator, _INSTALLED_FLAG, True)
    return decorator


setattr(thread_timeout, _INSTALLED_FLAG, True)


def install_thread_timeout() -> bool:
    """Rebind ``math_verify``'s alarm-based ``timeout`` to :func:`thread_timeout` (idempotent;
    False when ``math_verify`` is not importable)."""
    try:
        import math_verify.grader as grader
        import math_verify.parser as parser
        import math_verify.utils as utils
    except ImportError:
        return False
    for module in (utils, parser, grader):
        current = getattr(module, "timeout", None)
        if current is None or getattr(current, _INSTALLED_FLAG, False):
            continue
        module.timeout = thread_timeout
    return True


# --- process isolation of verifiers.v1.scoring.verify_boxed_math_answer -----------------------


def _worker_main(conn: Any) -> None:  # pragma: no cover - runs in the spawned process
    from verifiers.v1.scoring import verify_boxed_math_answer

    while True:
        try:
            item = conn.recv()
        except EOFError:
            return
        if item is None:
            return
        response, answer, timeout_seconds = item
        try:
            result = float(
                verify_boxed_math_answer(response, answer, timeout_seconds=timeout_seconds)
            )
        except BaseException:  # noqa: BLE001 - verifiers itself maps failures to 0.0
            result = 0.0
        conn.send(result)


class _WorkerUnavailable(RuntimeError):
    """The worker pipe failed before a result deadline could be enforced."""


class _Worker:
    def __init__(self) -> None:
        context = mp.get_context("spawn")
        self.conn, child = context.Pipe()
        self.process = context.Process(target=_worker_main, args=(child,), daemon=True)
        self.process.start()
        child.close()

    def call(self, args: tuple[Any, ...], deadline_s: float) -> float | None:
        """Result, or ``None`` when the worker exceeded the deadline (it is killed)."""
        try:
            self.conn.send(args)
        except (EOFError, OSError, ValueError) as exc:
            self.kill()
            raise _WorkerUnavailable from exc
        if self.conn.poll(deadline_s):
            try:
                return float(self.conn.recv())
            except (EOFError, OSError, ValueError) as exc:
                self.kill()
                raise _WorkerUnavailable from exc
        self.kill()
        return None

    def kill(self) -> None:
        try:
            if self.process.is_alive():
                self.process.kill()
            self.process.join(5.0)
        finally:
            self.conn.close()

    @property
    def alive(self) -> bool:
        return self.process.is_alive()


class MathVerifyPool:
    """A few spawned workers, each used by one caller at a time, replaced when killed.

    ``verify`` is synchronous because verifiers invokes reward functions on the event-loop
    thread. Its parent-side deadline is therefore deliberately limited to the worker's
    ``timeout_seconds + grace_s``. This may return 0.0 before several sequential internal
    parse/verify deadlines finish, but bounds the default event-loop stall to seven seconds.
    """

    def __init__(self, size: int = 4, *, grace_s: float = 2.0) -> None:
        if size < 1:
            raise ValueError("size must be >= 1")
        self.size = size
        self.grace_s = grace_s
        self._idle: list[_Worker] = []
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(size)
        self.num_calls = 0
        self.num_timeouts = 0

    def verify(self, response: str | None, answer: str, *, timeout_seconds: int = 5) -> float:
        with self._slots:
            with self._lock:
                worker = self._idle.pop() if self._idle else None
            if worker is None or not worker.alive:
                worker = _Worker()
            # math_verify enforces timeout_seconds per parse/verify call inside the worker;
            # the pipe deadline covers the whole helper plus the grace for pathological inputs.
            args = (response, answer, timeout_seconds)
            deadline_s = timeout_seconds + self.grace_s
            try:
                result = worker.call(args, deadline_s)
            except _WorkerUnavailable:
                worker = _Worker()
                try:
                    result = worker.call(args, deadline_s)
                except _WorkerUnavailable:
                    result = None
            with self._lock:
                self.num_calls += 1
                if result is None:
                    self.num_timeouts += 1
                elif worker.alive:
                    self._idle.append(worker)
            return 0.0 if result is None else result

    def close(self) -> None:
        with self._lock:
            workers, self._idle = self._idle, []
        for worker in workers:
            try:
                worker.conn.send(None)
            except (OSError, ValueError):
                pass
            worker.process.join(1.0)
            if worker.alive:
                worker.kill()


_POOL: MathVerifyPool | None = None
_POOL_LOCK = threading.Lock()


def _close_global_pool() -> None:
    with _POOL_LOCK:
        pool = _POOL
    if pool is not None:
        pool.close()


_ATEXIT_CLOSE = atexit.register(_close_global_pool)


def install_process_isolation(size: int = 4) -> bool:
    """Route ``verifiers.v1.verify_boxed_math_answer`` (and the ``scoring`` module attribute)
    through :class:`MathVerifyPool`. Idempotent; False when verifiers has no such helper."""
    global _POOL
    try:
        import verifiers.v1 as vf
        import verifiers.v1.scoring as scoring
    except ImportError:
        return False
    if not hasattr(scoring, "verify_boxed_math_answer"):
        return False
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = MathVerifyPool(size)
    original = scoring.verify_boxed_math_answer

    @functools.wraps(original)
    def isolated(response: str | None, answer: str, *, timeout_seconds: int = 5) -> float:
        with _POOL_LOCK:
            pool = _POOL
        if pool is None:  # defensive: installation always initializes it first
            return 0.0
        return pool.verify(response, answer, timeout_seconds=timeout_seconds)

    isolated._moongazer_rl_isolated = True  # type: ignore[attr-defined]
    for module in (scoring, vf):
        current = getattr(module, "verify_boxed_math_answer", None)
        if current is not None and not getattr(current, "_moongazer_rl_isolated", False):
            module.verify_boxed_math_answer = isolated  # type: ignore[attr-defined]
    return True


def math_verify_pool() -> MathVerifyPool | None:
    return _POOL


__all__ = [
    "MathVerifyPool",
    "install_process_isolation",
    "install_thread_timeout",
    "math_verify_pool",
    "thread_timeout",
]
