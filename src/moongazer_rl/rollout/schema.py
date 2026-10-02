"""Canonical rollout schema shared by the Verifiers runtime and every trainer backend.

Design rules:

* ``1 verifiers Task x G rollouts == 1 RolloutGroup``.
* Every model call keeps the token IDs and sampling logprobs returned by the policy at
  sampling time. Nothing here is re-tokenized from text.
* ``ModelCall`` is the lossless per-call unit ("1 model call = 1 call segment").
  ``TrainingSequence`` is the verifiers ``Branch`` view: one root-to-leaf token path with a
  per-token action mask. When every turn's prompt was an exact token prefix of the previous
  turn, a trajectory has exactly one sequence; otherwise verifiers forks and there are several.
  Trainers choose which unit to consume; both are always present.
* Nothing in this module imports torch, verifiers, unsloth, or ART.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "abort"] | None
"""finish_reason as reported by the policy backend; ``None`` when unknown."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorInfo(_Strict):
    """A verifiers ``Error`` (rollout, model, judge, ...) copied verbatim."""

    type: str
    message: str
    traceback: str | None = None


class ModelCall(_Strict):
    """One policy call inside a rollout, exactly as sampled.

    ``prompt_ids`` are the token IDs the policy actually consumed for this call (rendered
    chat template, or the previous turn's tokens plus a bridged tail). ``completion_ids`` and
    ``sampling_logprobs`` come from the sampling engine, aligned 1:1.
    """

    index: int = Field(ge=0, description="0-based order of this call inside its trajectory")
    prompt_ids: list[int]
    completion_ids: list[int]
    sampling_logprobs: list[float] = Field(
        default_factory=list,
        description="Per completion token; empty only when the policy could not report them",
    )
    finish_reason: FinishReason = None
    policy_version: int = Field(ge=0, description="Weight version the policy served this call with")
    started_at: float | None = None
    ended_at: float | None = None
    request_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_alignment(self) -> ModelCall:
        if self.sampling_logprobs and len(self.sampling_logprobs) != len(self.completion_ids):
            raise ValueError(
                "sampling_logprobs must align with completion_ids: "
                f"{len(self.sampling_logprobs)} != {len(self.completion_ids)}"
            )
        return self

    @property
    def has_logprobs(self) -> bool:
        return (
            len(self.sampling_logprobs) == len(self.completion_ids)
            and bool(self.completion_ids)
            and all(math.isfinite(logprob) for logprob in self.sampling_logprobs)
        )

    @property
    def action_mask(self) -> list[bool]:
        """Call-segment view: every completion token is an action."""
        return [True] * len(self.completion_ids)

    @property
    def duration_s(self) -> float | None:
        if self.started_at is None or self.ended_at is None:
            return None
        return self.ended_at - self.started_at


class TrainingSequence(_Strict):
    """One lossless token path (a verifiers ``Branch``) with a per-token action mask.

    ``token_ids[:prompt_len]`` is the untrainable prefix (everything before the first sampled
    token). Inside ``token_ids[prompt_len:]`` the ``action_mask`` marks sampled tokens; tool
    results, user-simulator turns and template scaffold are ``False``.
    """

    index: int = Field(ge=0)
    token_ids: list[int]
    action_mask: list[bool]
    sampling_logprobs: list[float] = Field(
        default_factory=list,
        description="Aligned to token_ids; 0.0 at non-action positions",
    )
    num_turns: int = Field(ge=0, description="Sampled model turns on this path")
    model_call_indices: list[int] = Field(
        default_factory=list,
        description="ModelCall.index values whose completions appear on this path, in order",
    )

    @model_validator(mode="after")
    def _check_alignment(self) -> TrainingSequence:
        if len(self.action_mask) != len(self.token_ids):
            raise ValueError("action_mask must align with token_ids")
        if self.sampling_logprobs and len(self.sampling_logprobs) != len(self.token_ids):
            raise ValueError("sampling_logprobs must align with token_ids")
        return self

    @property
    def prompt_len(self) -> int:
        for i, flag in enumerate(self.action_mask):
            if flag:
                return i
        return len(self.token_ids)

    @property
    def num_action_tokens(self) -> int:
        return sum(1 for flag in self.action_mask if flag)

    @property
    def completion_ids(self) -> list[int]:
        return self.token_ids[self.prompt_len :]

    @property
    def completion_mask(self) -> list[bool]:
        return self.action_mask[self.prompt_len :]

    @property
    def completion_logprobs(self) -> list[float]:
        return self.sampling_logprobs[self.prompt_len :] if self.sampling_logprobs else []


class Trajectory(_Strict):
    """One verifiers rollout (one ``Trace``) with everything a trainer or an inspector needs."""

    trajectory_id: str = Field(description="verifiers trace id; also the interception session id")
    rollout_index: int = Field(ge=0, description="Position inside the group")

    reward: float = Field(description="trace.reward == sum(reward_components.values())")
    reward_components: dict[str, float] = Field(
        default_factory=dict, description="trace.rewards: weighted contributions per @reward"
    )
    metrics: dict[str, float] = Field(default_factory=dict, description="trace.metrics verbatim")
    info: dict[str, Any] = Field(default_factory=dict, description="trace.info verbatim (JSON)")

    is_completed: bool = False
    stop_condition: str | None = None
    is_truncated: bool = False
    has_error: bool = False
    errors: list[ErrorInfo] = Field(default_factory=list)

    num_turns: int = Field(ge=0)
    num_input_tokens: int = Field(ge=0)
    num_output_tokens: int = Field(ge=0)
    timing: dict[str, float] = Field(
        default_factory=dict,
        description="setup_s, generation_s, finalize_s, scoring_s, total_s",
    )
    extra_usage: list[dict[str, Any]] = Field(
        default_factory=list, description="Judge and other off-graph model usage"
    )

    model_calls: list[ModelCall] = Field(default_factory=list)
    sequences: list[TrainingSequence] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def has_model_calls(self) -> bool:
        return bool(self.model_calls)

    @property
    def has_complete_logprobs(self) -> bool:
        return bool(self.model_calls) and all(c.has_logprobs for c in self.model_calls)

    @property
    def num_action_tokens(self) -> int:
        return sum(len(c.completion_ids) for c in self.model_calls)

    @property
    def is_trainable(self) -> bool:
        """A trajectory the trainer may use: at least one sampled token with logprobs and a
        finite reward. Errors do not disqualify (partial trajectories are data); a trainer may
        apply stricter filters on top."""
        return self.has_complete_logprobs and math.isfinite(self.reward)


class RolloutGroup(_Strict):
    """``1 Task x G rollouts``: the GRPO unit."""

    group_id: str
    task_idx: int
    task_name: str | None = None
    task_type: str
    task_id: str = Field(description="Stable task identity: '<taskset id>:<task idx>'")
    policy_version: int = Field(ge=0, description="Weight version when the episode started")
    trajectories: list[Trajectory]
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="taskset id/args, harness id, sampling, learner step, timings",
    )
    created_at: float | None = None

    @model_validator(mode="after")
    def _check_indices(self) -> RolloutGroup:
        indices = [t.rollout_index for t in self.trajectories]
        if sorted(indices) != list(range(len(indices))):
            raise ValueError("trajectory rollout_index values must be 0..G-1")
        return self

    @property
    def size(self) -> int:
        return len(self.trajectories)

    @property
    def rewards(self) -> list[float]:
        return [t.reward for t in self.trajectories]

    @property
    def finite_rewards(self) -> list[float]:
        """Rewards of trajectories whose reward is finite (NaN/inf rewards are recorded but
        never enter statistics; they also make the trajectory untrainable)."""
        return [r for r in self.rewards if math.isfinite(r)]

    @property
    def reward_mean(self) -> float:
        finite = self.finite_rewards
        return statistics.fmean(finite) if finite else math.nan

    @property
    def reward_std(self) -> float:
        """Population std over finite rewards (matches TRL's ``scale_rewards='group'``)."""
        finite = self.finite_rewards
        if len(finite) < 2:
            return 0.0
        return statistics.pstdev(finite)

    @property
    def is_zero_std(self) -> bool:
        return self.reward_std == 0.0

    @property
    def trainable_trajectories(self) -> list[Trajectory]:
        return [t for t in self.trajectories if t.is_trainable]

    @property
    def num_errors(self) -> int:
        return sum(1 for t in self.trajectories if t.has_error)

    @property
    def num_truncated(self) -> int:
        return sum(1 for t in self.trajectories if t.is_truncated)


__all__ = [
    "ErrorInfo",
    "FinishReason",
    "ModelCall",
    "RolloutGroup",
    "Trajectory",
    "TrainingSequence",
]
