"""TrainerBackend contract.

A trainer backend receives canonical ``RolloutGroup``s and performs optimizer steps. It also
owns the policy weights, so it hands out the ``PolicyBackend`` used for rollouts. The
orchestrator never touches model objects.

Backends:

* ``unsloth-native``: TRL ``GRPOTrainer`` (patched by Unsloth when installed) with generation
  replaced by the canonical rollouts. ``trainer.args`` are passed through to ``GRPOConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from moongazer_rl.policy.base import PolicyBackend
from moongazer_rl.rollout.schema import RolloutGroup


@dataclass
class StepResult:
    """Outcome of one optimizer step. Metric names below are the canonical keys the logger
    expects; backends fill what they can measure and leave the rest absent (never fake 0.0)."""

    global_step: int
    weight_version: int
    metrics: dict[str, float] = field(default_factory=dict)
    """Canonical keys (all optional):
    loss, policy_loss, grad_norm, entropy, kl, learning_rate,
    importance_ratio_mean, importance_ratio_p95, importance_ratio_p99,
    clipped_token_fraction, sampling_vs_learner_logprob_diff_mean,
    sampling_vs_learner_logprob_diff_abs_max, num_tokens, num_action_tokens
    """
    num_groups_received: int = 0
    num_groups_used: int = 0
    num_trajectories_used: int = 0
    num_sequences_used: int = 0
    num_action_tokens: int = 0
    wall_time_s: float = 0.0
    trajectory_ids_used: list[str] = field(default_factory=list)
    """Which trajectories contributed loss, so a run can prove rollout/learner correspondence."""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointInfo:
    path: Path
    global_step: int
    weight_version: int
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TrainerBackend(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def global_step(self) -> int: ...

    @property
    def weight_version(self) -> int: ...

    def validate_args(self, args: dict[str, Any]) -> list[str]:
        """Return human-readable problems with ``trainer.args`` for this backend (empty = OK).
        Must not require a GPU or load a model; used by ``validate`` and at startup."""
        ...

    def setup(self) -> None:
        """Load model/tokenizer, build the upstream trainer. May require a GPU."""
        ...

    def policy(self) -> PolicyBackend:
        """Inference backend sharing this trainer's weights. Valid after ``setup``."""
        ...

    def train_step(self, groups: list[RolloutGroup]) -> StepResult:
        """One optimizer step over ``groups``. Synchronous; the orchestrator runs it off the
        event loop when needed."""
        ...

    def save_checkpoint(self, directory: Path) -> CheckpointInfo: ...

    def load_checkpoint(self, directory: Path) -> CheckpointInfo: ...

    def close(self) -> None: ...


@runtime_checkable
class TrainabilityBackend(Protocol):
    """Optional cheap capability for matching a backend's exact row-building semantics.

    Keeping this separate from ``TrainerBackend`` preserves compatibility with backends that
    predate dynamic sampling; orchestrators must retain a generic fallback for those backends.
    ``early_drop_safe`` additionally says whether this decision is group-local, so a producer
    may apply it before queueing without changing a batch-scoped objective.
    """

    @property
    def early_drop_safe(self) -> bool:
        """Whether ``would_train`` is final before other groups in the step are known."""
        ...

    def would_train(self, group: RolloutGroup) -> bool:
        """Whether ``train_step([group])`` would contain a non-zero-advantage row."""
        ...


__all__ = ["CheckpointInfo", "StepResult", "TrainabilityBackend", "TrainerBackend"]
