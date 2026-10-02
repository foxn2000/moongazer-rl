"""verifiers ``Trace`` -> canonical ``Trajectory`` / ``RolloutGroup``.

Two rules govern everything
here:

* **Nothing is re-composed.** ``trace.rewards`` are copied as-is (never re-weighted or
  re-summed), metrics and info are copied verbatim, and token IDs come straight off the
  graph — nothing is re-tokenized.
* **A failed rollout is data.** A trace that errored before its first model call still
  becomes a ``Trajectory`` (with ``model_calls=[]``), so a run can account for every
  rollout it launched.

The two canonical views of one rollout come from different places: ``model_calls`` are
what ``PolicyClient`` recorded per turn, and ``sequences`` are the verifiers graph's
root-to-leaf ``Branch``es. :func:`verify_fidelity` cross-checks them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from verifiers.v1.graph import MessageNode
from verifiers.v1.trace import Branch, Trace

from moongazer_rl.policy.base import CallUnit
from moongazer_rl.rollout.schema import (
    ErrorInfo,
    ModelCall,
    RolloutGroup,
    TrainingSequence,
    Trajectory,
)

if TYPE_CHECKING:
    from moongazer_rl.rollout.client import CallStore

__all__ = [
    "fidelity_summary",
    "trace_to_trajectory",
    "traces_to_group",
    "verify_fidelity",
    "verify_training_fidelity",
]


def _verifiers_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("verifiers")
    except PackageNotFoundError:  # pragma: no cover - verifiers is a hard dependency
        return "unknown"


def _timing(trace: Trace) -> dict[str, float]:
    timing = trace.timing
    spans = (timing.setup, timing.generation, timing.finalize, timing.scoring)
    ends = [span.end for span in spans if span.end]
    total = max(ends) - timing.start if ends and timing.start else 0.0
    return {
        "setup_s": timing.setup.duration,
        "generation_s": timing.generation.duration,
        "finalize_s": timing.finalize.duration,
        "scoring_s": timing.scoring.duration,
        "total_s": max(0.0, total),
    }


def _sampled_completion(node: MessageNode) -> list[int]:
    """The model-sampled suffix of a graph node (``[]`` for input nodes).

    ``graph._commit_turn`` always writes an assistant node as
    ``generation-prompt scaffold (mask False) + sampled completion (mask True)``, so the
    sampled tokens are the suffix beginning at the first ``True``.
    """
    first = next((i for i, sampled in enumerate(node.mask) if sampled), None)
    return [] if first is None else list(node.token_ids[first:])


def _thinking_leak_metadata(calls: Sequence[ModelCall]) -> dict[str, bool | str]:
    """Collapse per-call detection recorded by ``PolicyClient`` to one trajectory flag."""
    detected = False
    reasons: list[str] = []
    for call in calls:
        leak = call.extra.get("thinking_leak")
        if not isinstance(leak, Mapping) or leak.get("detected") is not True:
            continue
        detected = True
        reason = leak.get("reason")
        if isinstance(reason, str) and reason not in reasons:
            reasons.append(reason)
    reason = ",".join(reasons)
    if not reason:
        reason = "unknown" if detected else "none"
    return {"detected": detected, "reason": reason}


def _branch_to_sequence(branch: Branch, index: int, calls: Sequence[ModelCall]) -> TrainingSequence:
    """One verifiers ``Branch`` -> one ``TrainingSequence``.

    ``model_call_indices`` links the branch's sampled spans back to the recorded calls:
    each sampled node is matched, in branch order, against the next recorded call whose
    ``completion_ids`` are identical. A forked trajectory re-walks the call list from the
    start for each branch, since branches share their common prefix's calls.
    """
    call_indices: list[int] = []
    cursor = 0
    for node in branch.nodes:
        if not node.sampled:
            continue
        completion = _sampled_completion(node)
        if not completion:
            continue
        match = next(
            (i for i in range(cursor, len(calls)) if calls[i].completion_ids == completion),
            None,
        )
        if match is None:
            continue
        call_indices.append(calls[match].index)
        cursor = match + 1
    action_mask = branch.sampled_mask
    sampling_logprobs = [
        logprob if position >= len(action_mask) or action_mask[position] else 0.0
        for position, logprob in enumerate(branch.logprobs)
    ]
    return TrainingSequence(
        index=index,
        token_ids=branch.token_ids,
        action_mask=action_mask,
        sampling_logprobs=sampling_logprobs,
        num_turns=branch.num_turns,
        model_call_indices=call_indices,
    )


def trace_to_trajectory(
    trace: Trace,
    calls: Sequence[ModelCall],
    rollout_index: int,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Trajectory:
    """Convert one finished verifiers trace into a canonical ``Trajectory``.

    ``calls`` are the ``ModelCall``s ``PolicyClient`` recorded for ``trace.id``; an empty
    list is legitimate (the rollout failed before its first model call) and the trajectory
    is kept, not dropped.
    """
    model_calls = list(calls)
    sequences = [
        _branch_to_sequence(branch, i, model_calls) for i, branch in enumerate(trace.branches)
    ]
    trajectory_metadata: dict[str, Any] = {
        "task_idx": trace.task.data.idx,
        "task_name": trace.task.data.name,
        "task_type": trace.task.type,
        "num_branches": len(sequences),
        "thinking_leak": _thinking_leak_metadata(model_calls),
    }
    if metadata:
        trajectory_metadata.update(metadata)
    # A trace whose scoring raised (verifiers wraps it as a TaskError and leaves ``rewards``
    # empty) has no reward, not a reward of 0.0: ``Trace.reward`` sums an empty dict. Training
    # on it would teach "this answer was wrong" for a rollout nobody graded. Mark it
    # non-finite so ``Trajectory.is_trainable`` is False and it is counted as dropped.
    reward = float(trace.reward)
    scoring_failed = bool(trace.errors) and not trace.rewards
    if scoring_failed:
        reward = math.nan
        trajectory_metadata["scoring_failed"] = True
    return Trajectory(
        trajectory_id=trace.id,
        rollout_index=rollout_index,
        reward=reward,
        reward_components={name: float(value) for name, value in trace.rewards.items()},
        metrics={name: float(value) for name, value in trace.metrics.items()},
        info=dict(trace.info),
        is_completed=bool(trace.is_completed),
        stop_condition=trace.stop_condition,
        is_truncated=bool(trace.is_truncated),
        has_error=bool(trace.has_error),
        errors=[
            ErrorInfo(type=error.type, message=error.message, traceback=error.traceback)
            for error in trace.errors
        ],
        num_turns=trace.num_turns,
        num_input_tokens=trace.num_input_tokens,
        num_output_tokens=trace.num_output_tokens,
        timing=_timing(trace),
        extra_usage=[usage.model_dump(mode="json") for usage in trace.extra_usage],
        model_calls=model_calls,
        sequences=sequences,
        metadata=trajectory_metadata,
    )


def traces_to_group(
    traces: Sequence[Trace],
    store: CallStore,
    *,
    group_id: str,
    taskset_id: str,
    policy_version: int,
    harness_id: str | None = None,
    sampling: Mapping[str, Any] | None = None,
    learner_step: int | None = None,
    created_at: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RolloutGroup:
    """Convert one episode's traces (``1 Task x G rollouts``) into a ``RolloutGroup``.

    Each trace's calls are **popped** from ``store``, so a converted group leaves nothing
    behind; the runtime asserts ``store.pending_ids`` is empty afterwards.
    """
    trace_list = list(traces)
    if not trace_list:
        raise ValueError("an episode produced no traces; cannot build a RolloutGroup")

    first = trace_list[0]
    shared: dict[str, Any] = {
        "taskset_id": taskset_id,
        "harness_id": harness_id,
        "sampling": dict(sampling or {}),
        "learner_step": learner_step,
        "verifiers_version": _verifiers_version(),
    }
    trajectories = [
        trace_to_trajectory(trace, store.pop(trace.id), index, metadata=shared)
        for index, trace in enumerate(trace_list)
    ]
    group_metadata: dict[str, Any] = {
        **shared,
        "trace_ids": [trace.id for trace in trace_list],
        "num_rollouts": len(trace_list),
    }
    if metadata:
        group_metadata.update(metadata)
    return RolloutGroup(
        group_id=group_id,
        task_idx=first.task.data.idx,
        task_name=first.task.data.name,
        task_type=first.task.type,
        task_id=f"{taskset_id}:{first.task.data.idx}",
        policy_version=policy_version,
        trajectories=trajectories,
        metadata=group_metadata,
        created_at=created_at,
    )


def _find_span(
    sequence: TrainingSequence,
    completion_ids: list[int],
    *,
    start: int = 0,
    require_action: bool = True,
) -> int | None:
    """Start index of ``completion_ids`` inside ``sequence`` at or after ``start``.

    The start must always be actionable. By default every later position must be actionable
    too; training-view validation can disable that full-span condition because a sequence row
    may mask positions inside a recorded call. Calls are matched in order with a moving cursor
    so repeated answers resolve to their own occurrences.
    """
    n = len(completion_ids)
    if n == 0:
        return None
    tokens = sequence.token_ids
    mask = sequence.action_mask
    while True:
        try:
            start = tokens.index(completion_ids[0], start)
        except ValueError:
            return None
        if (
            tokens[start : start + n] == completion_ids
            and start < len(mask)
            and mask[start]
            and (not require_action or (start + n <= len(mask) and all(mask[start : start + n])))
        ):
            return start
        start += 1


def verify_fidelity(trajectory: Trajectory) -> list[str]:
    """Cross-check the two canonical views of one rollout; return human-readable problems.

    For every recorded ``ModelCall`` this asserts that its ``completion_ids`` appear
    verbatim in at least one ``TrainingSequence``, marked as actions, and that logprob
    lengths line up on both sides. Never raises: the runtime records the findings in
    ``trajectory.metadata["fidelity_problems"]`` so a run can measure drift instead of
    dying on it.
    """
    problems: list[str] = []
    if trajectory.model_calls and not trajectory.sequences:
        problems.append(
            f"{len(trajectory.model_calls)} model call(s) recorded but the trace has no "
            "branches (token sequences)"
        )
    cursors: dict[int, int] = {}  # sequence index -> end of the previous matched call
    for call in trajectory.model_calls:
        if call.sampling_logprobs and len(call.sampling_logprobs) != len(call.completion_ids):
            problems.append(
                f"call {call.index}: {len(call.sampling_logprobs)} sampling logprobs for "
                f"{len(call.completion_ids)} completion tokens"
            )
        if not call.completion_ids:
            continue
        located = False
        for sequence in trajectory.sequences:
            start = _find_span(sequence, call.completion_ids, start=cursors.get(sequence.index, 0))
            if start is None:
                continue
            located = True
            cursors[sequence.index] = start + len(call.completion_ids)
            if sequence.sampling_logprobs and call.sampling_logprobs:
                span = sequence.sampling_logprobs[start : start + len(call.completion_ids)]
                if span != call.sampling_logprobs:
                    problems.append(
                        f"call {call.index}: sampling logprobs differ from sequence "
                        f"{sequence.index} at [{start}, {start + len(call.completion_ids)})"
                    )
            break
        if not located:
            problems.append(
                f"call {call.index}: its {len(call.completion_ids)} completion tokens do not "
                "appear as an action span in any sequence"
            )
    for sequence in trajectory.sequences:
        if sequence.sampling_logprobs and len(sequence.sampling_logprobs) != len(
            sequence.token_ids
        ):
            problems.append(
                f"sequence {sequence.index}: {len(sequence.sampling_logprobs)} logprobs for "
                f"{len(sequence.token_ids)} tokens"
            )
    return problems


def _action_spans(sequence: TrainingSequence) -> list[tuple[int, int]]:
    """Half-open maximal runs of actionable positions in one sequence."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for position, actionable in enumerate(sequence.action_mask):
        if actionable and start is None:
            start = position
        elif not actionable and start is not None:
            spans.append((start, position))
            start = None
    if start is not None:
        spans.append((start, len(sequence.action_mask)))
    return spans


def _uncovered_runs(covered_by: list[int | None], start: int, end: int) -> list[tuple[int, int]]:
    """Half-open runs inside ``[start, end)`` that no model call covers."""
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for position in range(start, end):
        if covered_by[position] is None:
            if run_start is None:
                run_start = position
        elif run_start is not None:
            runs.append((run_start, position))
            run_start = None
    if run_start is not None:
        runs.append((run_start, end))
    return runs


def verify_training_fidelity(
    trajectory: Trajectory,
    *,
    unit: CallUnit,
    group_policy_version: int | None = None,
    sequence_indices: set[int] | None = None,
) -> list[str]:
    """Validate only the canonical view that will become loss input.

    Unlike :func:`verify_fidelity`, this is a training gate. Call mode deliberately ignores
    sequence defects. Sequence mode validates every selected actionable branch span against the
    recorded calls, while using an independent cursor for each branch so forked branches may
    share calls. ``sequence_indices=None`` selects every sequence with action tokens.
    """
    problems: list[str] = []

    if trajectory.model_calls:
        expected_version = trajectory.model_calls[0].policy_version
        for call in trajectory.model_calls:
            if call.policy_version != expected_version:
                problems.append(
                    f"call {call.index}: policy_version {call.policy_version} differs from "
                    f"call {trajectory.model_calls[0].index} policy_version {expected_version}"
                )
            if group_policy_version is not None and call.policy_version != group_policy_version:
                problems.append(
                    f"call {call.index}: policy_version {call.policy_version} differs from "
                    f"group policy_version {group_policy_version}"
                )

    if unit == "call":
        for call in trajectory.model_calls:
            if not call.completion_ids:
                continue
            if len(call.sampling_logprobs) != len(call.completion_ids):
                problems.append(
                    f"call {call.index}: {len(call.sampling_logprobs)} sampling logprobs for "
                    f"{len(call.completion_ids)} completion tokens"
                )
            non_finite = [
                position
                for position, logprob in enumerate(call.sampling_logprobs)
                if not math.isfinite(logprob)
            ]
            if non_finite:
                problems.append(
                    f"call {call.index}: non-finite sampling logprobs at completion positions "
                    f"{non_finite}"
                )
            if not call.has_logprobs and not (
                len(call.sampling_logprobs) != len(call.completion_ids) or non_finite
            ):
                problems.append(f"call {call.index}: completion has no sampling logprobs")
        return problems

    if unit != "sequence":  # defensive for callers outside the typed API
        raise ValueError(f"unknown call unit: {unit!r}")

    calls = [(position, call) for position, call in enumerate(trajectory.model_calls)]
    calls_by_index = {call.index: (position, call) for position, call in calls}
    for sequence in trajectory.sequences:
        if sequence.num_action_tokens <= 0 or (
            sequence_indices is not None and sequence.index not in sequence_indices
        ):
            continue
        lengths = (
            len(sequence.sampling_logprobs),
            len(sequence.token_ids),
            len(sequence.action_mask),
        )
        if len(set(lengths)) != 1:
            problems.append(
                f"sequence {sequence.index}: {lengths[0]} sampling logprobs, "
                f"{lengths[1]} tokens, and {lengths[2]} action-mask positions"
            )
        non_finite = [
            position
            for position, actionable in enumerate(sequence.action_mask)
            if actionable
            and (
                position >= len(sequence.sampling_logprobs)
                or not math.isfinite(sequence.sampling_logprobs[position])
            )
        ]
        if non_finite:
            problems.append(
                f"sequence {sequence.index}: non-finite or missing sampling logprobs at "
                f"action positions {non_finite}"
            )

        sequence_cursor = 0
        covered_by: list[int | None] = [None] * len(sequence.action_mask)
        if sequence.model_call_indices:
            sequence_calls = [
                calls_by_index[index]
                for index in sequence.model_call_indices
                if index in calls_by_index
            ]
        else:
            sequence_calls = calls
        for call_position, call in sequence_calls:
            if not call.completion_ids:
                continue
            start = _find_span(
                sequence,
                call.completion_ids,
                start=sequence_cursor,
                require_action=False,
            )
            if start is None:
                continue
            end = start + len(call.completion_ids)
            sequence_cursor = end
            for position in range(start, min(end, len(covered_by))):
                covered_by[position] = call_position
            mismatched = [
                position
                for position in range(start, min(end, len(sequence.action_mask)))
                if sequence.action_mask[position]
                and (
                    position >= len(sequence.sampling_logprobs)
                    or position - start >= len(call.sampling_logprobs)
                    or sequence.sampling_logprobs[position]
                    != call.sampling_logprobs[position - start]
                )
            ]
            if mismatched:
                problems.append(
                    f"call {call.index}: sampling logprobs differ from sequence "
                    f"{sequence.index} at action positions {mismatched}"
                )

        # Every sampled token must come from exactly one recorded call. Two calls may sit back
        # to back inside one maximal action span (a continued turn with no scaffold between
        # them); the moving cursor already prevents two calls from claiming the same position.
        uncovered = [
            (start, end)
            for start, end in _action_spans(sequence)
            for start, end in _uncovered_runs(covered_by, start, end)
        ]
        if uncovered:
            problems.append(
                f"sequence {sequence.index}: action positions {uncovered} are not covered by "
                "any model-call completion"
            )

    return problems


def fidelity_summary(group: RolloutGroup) -> dict[str, Any]:
    """Summarise the runtime's both-view fidelity audit for logging/orchestration."""
    found: dict[str, list[str]] = {}
    for trajectory in group.trajectories:
        raw = trajectory.metadata.get("fidelity_problems", [])
        problems = [str(problem) for problem in raw] if isinstance(raw, list) else []
        if problems:
            found[trajectory.trajectory_id] = problems
    return {
        "num_trajectories": len(group.trajectories),
        "num_with_problems": len(found),
        "problems": found,
    }
