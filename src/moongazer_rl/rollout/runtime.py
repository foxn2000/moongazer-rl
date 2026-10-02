"""``VerifiersRuntime``: run verifiers v1 environments and hand back canonical groups.

Verifiers owns the rollout loop — taskset, harness, tools, user simulation, stop
conditions, group rewards. This class owns only the three things it cannot: the model
call (a :class:`~moongazer_rl.rollout.client.PolicyClient` over a local
``PolicyBackend``), the serving lifecycle, and the conversion to
:class:`~moongazer_rl.rollout.schema.RolloutGroup`.

Lifecycle (mirrors the upstream eval runners):

    runtime = VerifiersRuntime({"taskset": {...}, "harness": {...}}, policy=..., renderer=...)
    async with runtime:                       # holds env.serving() open across all groups
        tasks = runtime.load_tasks()
        group = await runtime.rollout_group(tasks[0], 4)

Importing this module must stay torch-free: it pulls in verifiers, renderers and pydantic
only. Trainer backends are never imported here.
"""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from typing import Any

import verifiers.v1 as vf
from renderers.base import Renderer
from verifiers.v1.clients import ModelContext
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace

from moongazer_rl.policy.base import PolicyBackend
from moongazer_rl.rollout.client import CallStore, PolicyClient
from moongazer_rl.rollout.convert import traces_to_group, verify_fidelity
from moongazer_rl.rollout.math_verify_compat import (
    install_process_isolation,
    install_thread_timeout,
)
from moongazer_rl.rollout.schema import RolloutGroup

__all__ = ["RuntimeLifecycleError", "VerifiersRuntime"]

TraceRecordsCallback = Callable[[RolloutGroup, list[dict]], Awaitable[None] | None]
"""Called once per converted group with the raw ``trace.to_record()`` dicts, so a logger
can write ``traces.jsonl`` without the records riding inside ``RolloutGroup``."""


class RuntimeLifecycleError(RuntimeError):
    """The runtime was used outside its ``async with`` block, or entered twice."""


class VerifiersRuntime:
    """One verifiers ``Environment`` plus the policy that answers its model calls.

    ``environment`` is the plain ``EnvConfig`` mapping (``taskset``, ``harness``,
    ``timeout``, ``retries``, ``max_turns``, ``max_input_tokens``, ``max_output_tokens``,
    ``max_total_tokens``, ``multiplex``). It is validated through
    ``vf.EnvConfig.model_validate``, which resolves ``taskset.id`` / ``harness.id`` to
    each plugin's own typed config — so plugin-specific fields are checked by verifiers,
    not by us.
    """

    def __init__(
        self,
        environment: Mapping[str, Any],
        *,
        policy: PolicyBackend,
        renderer: Renderer,
        model_name: str = "moongazer-rl-policy",
        sampling: Mapping[str, Any] | None = None,
        max_concurrent: int | None = 32,
        require_logprobs: bool = True,
        on_traces: TraceRecordsCallback | None = None,
    ) -> None:
        # math_verify's signal.alarm deadline is unsafe on an asyncio loop and a runaway
        # sympy call starves it: score boxed answers in killable worker processes.
        install_thread_timeout()
        install_process_isolation()
        self.config = vf.EnvConfig.model_validate(dict(environment))
        self.environment = vf.Environment(self.config)
        self.policy = policy
        self.renderer = renderer
        self.model_name = model_name
        self.sampling = dict(sampling or {})
        self.sampling_config = vf.Sampling.model_validate(self.sampling)
        self.max_concurrent = max_concurrent
        self.on_traces = on_traces
        self.store = CallStore()
        self.client = PolicyClient(
            policy=policy,
            renderer=renderer,
            store=self.store,
            require_logprobs=require_logprobs,
        )
        self._tasks: list[Task] | None = None
        self._serving: Any = None
        self._active = False
        self._semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent is not None else None

    # --- identity ---------------------------------------------------------------------

    @property
    def taskset_id(self) -> str:
        return self.config.taskset.id

    @property
    def harness_id(self) -> str:
        return self.config.harness.id

    @property
    def is_active(self) -> bool:
        return self._active

    # --- lifecycle --------------------------------------------------------------------

    async def __aenter__(self) -> VerifiersRuntime:
        if self._active:
            raise RuntimeLifecycleError("VerifiersRuntime is already serving")
        self._serving = self.environment.serving()
        try:
            await self._serving.__aenter__()
        except BaseException:
            self._serving = None
            raise
        self._active = True
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        try:
            if self._serving is not None:
                await self._serving.__aexit__(exc_type, exc, traceback)
        finally:
            self._serving = None
            self._active = False
            self.store.clear()

    def load_tasks(self, *, reload: bool = False) -> list[Task]:
        """The taskset's typed tasks, in taskset order. Cached after the first call."""
        if self._tasks is None or reload:
            self._tasks = list(self.environment.taskset.load())
        return list(self._tasks)

    # --- rollouts ---------------------------------------------------------------------

    async def rollout_group(
        self,
        task: Task,
        num_rollouts: int,
        *,
        group_id: str | None = None,
        learner_step: int | None = None,
        policy_version: int | None = None,
    ) -> RolloutGroup:
        """Run ONE verifiers episode (``task`` x ``num_rollouts``) and convert it.

        Rollout-level failures are already data: verifiers records them on the trace and
        the group still contains one trajectory per rollout. An exception escaping here
        is a framework or programming failure; the episode's store entries are discarded
        first so nothing leaks into the next step.
        """
        if not self._active:
            raise RuntimeLifecycleError(
                "use 'async with VerifiersRuntime(...)' before running rollouts"
            )
        if num_rollouts < 1:
            raise ValueError("num_rollouts must be at least 1")

        episode_client = self.client
        if policy_version is not None:
            episode_client = PolicyClient(
                policy=self.policy,
                renderer=self.renderer,
                store=self.store,
                require_logprobs=self.client.require_logprobs,
                policy_version=policy_version,
            )
        context = ModelContext(
            model=self.model_name, client=episode_client, sampling=self.sampling_config
        )
        # Sampling happens with the weights served now; a mid-episode sync would be a bug.
        group_policy_version = (
            self.policy.weight_version if policy_version is None else policy_version
        )
        created_at = time.time()
        episode = self.environment.episode(task, context, n=num_rollouts)
        try:
            traces = await episode.run(self._semaphore)
        except BaseException:
            for rollout in episode.rollouts:
                if rollout.trace is not None:
                    self.store.discard(rollout.trace.id)
            raise

        group = traces_to_group(
            traces,
            self.store,
            group_id=group_id or self._group_id(task, learner_step),
            taskset_id=self.taskset_id,
            harness_id=self.harness_id,
            policy_version=group_policy_version,
            sampling=self.sampling,
            learner_step=learner_step,
            created_at=created_at,
            metadata={
                "model_name": self.model_name,
                "policy_name": self.policy.name,
                "wall_time_s": time.time() - created_at,
            },
        )
        for trajectory in group.trajectories:
            trajectory.metadata["fidelity_problems"] = verify_fidelity(trajectory)
        await self._emit_traces(group, traces)
        return group

    async def rollout_groups(
        self,
        tasks: Iterable[Task],
        num_rollouts: int,
        *,
        learner_step: int | None = None,
        policy_version: int | None = None,
    ) -> list[RolloutGroup]:
        """Run several episodes concurrently; groups come back in ``tasks`` order.

        Failure policy: every task is allowed to settle, then the **first** exception is
        re-raised. Nothing is swallowed and no partial list is returned silently — a
        caller that wants the successful groups alongside the failures uses
        :meth:`rollout_groups_settled` instead.
        """
        settled = await self.rollout_groups_settled(
            tasks,
            num_rollouts,
            learner_step=learner_step,
            policy_version=policy_version,
        )
        for item in settled:
            if isinstance(item, BaseException):
                raise item
        return [item for item in settled if isinstance(item, RolloutGroup)]

    async def rollout_groups_settled(
        self,
        tasks: Iterable[Task],
        num_rollouts: int,
        *,
        learner_step: int | None = None,
        policy_version: int | None = None,
    ) -> list[RolloutGroup | BaseException]:
        """Like :meth:`rollout_groups`, but returns failures in place, in ``tasks`` order.

        Concurrency across episodes is bounded by the same semaphore verifiers applies
        per rollout (``max_concurrent``), so a caller can hand it every task at once.
        """
        task_list = list(tasks)
        results = await asyncio.gather(
            *(
                self.rollout_group(
                    task,
                    num_rollouts,
                    learner_step=learner_step,
                    policy_version=policy_version,
                )
                for task in task_list
            ),
            return_exceptions=True,
        )
        return list(results)

    # --- internals --------------------------------------------------------------------

    def _group_id(self, task: Task, learner_step: int | None) -> str:
        step = "-" if learner_step is None else str(learner_step)
        return f"{self.taskset_id}:{task.data.idx}:step{step}:{uuid.uuid4().hex[:8]}"

    async def _emit_traces(self, group: RolloutGroup, traces: Sequence[Trace]) -> None:
        if self.on_traces is None:
            return
        result = self.on_traces(group, [trace.to_record() for trace in traces])
        if inspect.isawaitable(result):
            await result
