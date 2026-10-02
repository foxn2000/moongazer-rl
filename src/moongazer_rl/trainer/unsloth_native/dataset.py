"""``RolloutGroup`` -> flat training rows for the TRL/Unsloth GRPO shim.

One row is one (prompt, completion) pair with a per-completion-token **action mask**. The
shim feeds ``prompt_ids`` / ``completion_ids`` / ``sampling_logprobs`` straight into
``GRPOTrainer._generate_single_turn`` and puts the action mask into the batch as
``tool_mask`` (Unsloth's own name for it).

Two units, both lossless (``policy.base.CallUnit``):

``"sequence"`` (default)
    One row per ``TrainingSequence`` (a verifiers ``Branch``): the whole root-to-leaf token
    path, with tool results / user-simulator turns masked out. Multi-turn credit assignment
    is exact and the prefix is shared, so this is the cheap and correct default.
``"call"``
    One row per ``ModelCall``: prompt = the tokens the policy consumed for that call,
    completion = what it sampled, mask = all ones. Lossless too, but the shared prefix is
    recomputed once per call.

A trajectory with several sequences (verifiers forked because a prior turn re-tokenised)
produces several rows in ``"sequence"`` mode. Nothing is dropped for that reason.

Advantages are computed here, per canonical ``RolloutGroup``, with TRL's exact formula
(``(r - mean) / (std + 1e-4)`` over the group, sample std, ``scale_rewards`` honoured). The
baseline is always a **per-trajectory** statistic — one reward per trajectory, then broadcast
to that trajectory's rows — so a trajectory that yields several rows is not double-counted. The
the shim overwrites TRL's own ``advantages`` tensor with them. This decouples the GRPO group
from TRL's rigid "``num_generations`` consecutive rows" layout, so a group may contribute a
variable number of rows (fork, ``"call"`` mode, a filtered trajectory) without silently
reshaping anything.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Literal

from moongazer_rl.policy.base import CallUnit
from moongazer_rl.rollout.convert import verify_training_fidelity
from moongazer_rl.rollout.schema import RolloutGroup, TrainingSequence, Trajectory

ScaleRewards = Literal["group", "batch", "none"]
ForkPolicy = Literal["error", "all", "longest"]

MIN_GROUP_SIZE = 2
"""A GRPO group needs at least two trainable trajectories to have a baseline."""

_STD_EPS = 1e-4
"""TRL: ``advantages / (std_rewards + 1e-4)`` (trl/trainer/grpo_trainer.py:1507)."""


@dataclass(frozen=True)
class AdvantageSpec:
    advantage_estimator: str = "group_zscore"
    std_scope: str = "group"


@dataclass(frozen=True)
class TrainRow:
    """One training example: a prompt, a completion, and which completion tokens get loss."""

    trajectory_id: str
    group_id: str
    group_index: int
    """Index of the canonical ``RolloutGroup`` inside the batch (advantage grouping)."""
    prompt_ids: list[int]
    completion_ids: list[int]
    sampling_logprobs: list[float]
    action_mask: list[bool]
    """Aligned to ``completion_ids``. ``False`` = tool/user/scaffold token, no loss."""
    reward: float
    advantage: float
    unit: CallUnit
    sub_index: int
    """``TrainingSequence.index`` in ``"sequence"`` mode, ``ModelCall.index`` in ``"call"``."""
    behavior_version: int = 0
    """Adapter version that sampled the containing episode."""
    truncated: bool = False
    """The sampled completion hit ``max_tokens`` (``ModelCall.finish_reason == "length"``; for
    a sequence, its last call). This is the *renderer-aware* truncation signal: TRL's own test
    (last token not in ``{eos, pad}``) misfires when the stop token is a chat-template token
    such as Gemma's ``<turn|>``, so the shim uses this flag instead."""

    def __post_init__(self) -> None:
        n = len(self.completion_ids)
        if len(self.action_mask) != n:
            raise ValueError(
                f"action_mask/completion_ids misaligned for {self.trajectory_id}: "
                f"{len(self.action_mask)} != {n}"
            )
        if len(self.sampling_logprobs) != n:
            raise ValueError(
                f"sampling_logprobs/completion_ids misaligned for {self.trajectory_id}: "
                f"{len(self.sampling_logprobs)} != {n}"
            )
        if n == 0:
            raise ValueError(f"empty completion for {self.trajectory_id}")

    @property
    def num_action_tokens(self) -> int:
        return sum(1 for flag in self.action_mask if flag)

    @property
    def total_len(self) -> int:
        return len(self.prompt_ids) + len(self.completion_ids)


@dataclass
class RolloutBatch:
    """What one ``train_step`` feeds the trainer, plus the bookkeeping for ``StepResult``."""

    estimator: str = "group_zscore"
    num_zero_advantage_groups: int = 0
    rows: list[TrainRow] = field(default_factory=list)
    num_groups_received: int = 0
    num_groups_used: int = 0
    num_trajectories_received: int = 0
    num_trajectories_used: int = 0
    num_dropped_untrainable: int = 0
    num_dropped_fidelity: int = 0
    fidelity_dropped: list[tuple[str, list[str]]] = field(default_factory=list)
    num_dropped_small_group: int = 0
    num_zero_std_groups: int = 0
    num_rows_over_max_len: int = 0
    num_dropped_over_max_len: int = 0
    num_forked_trajectories: int = 0
    num_dropped_forked_branches: int = 0
    group_reward_mean: float = math.nan
    group_reward_std_mean: float = math.nan

    @property
    def size(self) -> int:
        return len(self.rows)

    @property
    def trajectory_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for row in self.rows:
            seen.setdefault(row.trajectory_id, None)
        return list(seen)

    @property
    def num_action_tokens(self) -> int:
        return sum(row.num_action_tokens for row in self.rows)

    @property
    def num_truncated_rows(self) -> int:
        return sum(1 for row in self.rows if row.truncated)

    @property
    def num_tokens(self) -> int:
        return sum(row.total_len for row in self.rows)

    @property
    def one_row_per_trajectory(self) -> bool:
        """True when every used trajectory produced exactly one row. Only then does TRL's own
        row-position grouping agree with our per-trajectory advantage baseline."""
        return len(self.trajectory_ids) == len(self.rows)

    @property
    def rows_per_group(self) -> int | None:
        """Uniform rows/group when every used group contributed the same count, else ``None``."""
        counts = {}
        for row in self.rows:
            counts[row.group_index] = counts.get(row.group_index, 0) + 1
        values = set(counts.values())
        return values.pop() if len(values) == 1 else None

    def to_trl_inputs(self) -> list[dict[str, object]]:
        """The ``inputs`` list ``_generate_and_score_completions`` expects.

        Deliberately minimal: ``prompt`` is an empty string so ``is_conversational`` is False
        (TRL then skips the ``prompt.pop()`` bootstrap path), and ``reward`` is the only extra
        column, which ``_calculate_rewards`` forwards to the reward function as a kwarg
        (trl/trainer/grpo_trainer.py:1022-1023). Token IDs never travel through here; the shim
        reads them from this batch object directly.
        """
        return [{"prompt": "", "reward": float(row.reward)} for row in self.rows]


def build_batch(
    groups: list[RolloutGroup],
    *,
    unit: CallUnit = "sequence",
    scale_rewards: ScaleRewards = "group",
    fork_policy: ForkPolicy = "error",
    max_seq_length: int | None = None,
    sequence_overflow: Literal["error", "drop"] = "error",
    min_group_size: int = MIN_GROUP_SIZE,
    advantage: AdvantageSpec | None = None,
) -> RolloutBatch:
    """Flatten canonical groups into training rows.

    Filtering (all counted, never silent):

    * the selected loss-input view fails ``verify_training_fidelity`` -> fidelity-dropped.
    * a non-finite reward, or a selected unit that produces no row -> dropped as untrainable.
    * a group left with fewer than ``min_group_size`` trainable trajectories -> dropped whole
      (no GRPO baseline is definable).
    * a trajectory whose selected unit produces no sampled token -> dropped as untrainable.

    ``fork_policy`` decides what happens when a trajectory has several ``TrainingSequence``s
    in ``"sequence"`` mode: ``"error"`` (default, safest: a fork means the rollout re-tokenised
    and the operator should look at it), ``"all"`` (one row per branch; the trajectory's reward
    is broadcast to each), ``"longest"`` (keep the branch with the most action tokens; dropped
    branches are counted in ``num_dropped_forked_branches``).
    """
    if advantage is None:
        advantage = AdvantageSpec(
            advantage_estimator="group_center"
            if scale_rewards in ("none", False)
            else "group_zscore",
            std_scope="batch" if scale_rewards == "batch" else "group",
        )
    estimator = advantage.advantage_estimator
    if estimator not in {"group_zscore", "group_center", "rloo"}:
        raise ValueError(f"unknown advantage estimator: {estimator}")
    min_group_size = max(MIN_GROUP_SIZE, min_group_size)
    batch = RolloutBatch(num_groups_received=len(groups), estimator=estimator)
    group_reward_means: list[float] = []
    group_reward_stds: list[float] = []

    # (group_index, one reward per USED trajectory, rows). The advantage baseline is a
    # per-trajectory statistic: a trajectory that produced several rows (unit="call", or
    # fork_policy="all") must not be counted several times in its group's mean/std.
    pending: list[tuple[int, list[float], list[TrainRow]]] = []
    trajectory_rewards: list[float] = []

    for group in groups:
        batch.num_trajectories_received += len(group.trajectories)
        faithful: list[Trajectory] = []
        for traj in group.trajectories:
            sequence_indices: set[int] | None = None
            if unit == "sequence":
                usable_sequences = _usable_sequences(traj)
                if fork_policy == "longest" and usable_sequences:
                    longest = max(usable_sequences, key=lambda sequence: sequence.num_action_tokens)
                    sequence_indices = {longest.index}
                else:
                    sequence_indices = {sequence.index for sequence in usable_sequences}
            problems = verify_training_fidelity(
                traj,
                unit=unit,
                group_policy_version=group.policy_version,
                sequence_indices=sequence_indices,
            )
            if problems:
                batch.num_dropped_fidelity += 1
                batch.fidelity_dropped.append((traj.trajectory_id, problems))
            else:
                faithful.append(traj)
        # ``Trajectory.is_trainable`` is intentionally call-view conservative. Once the selected
        # view has passed its fidelity gate, only reward validity is common to both units; e.g. a
        # non-finite call logprob at a sequence-masked position must not discard a sound sequence.
        trainable = [t for t in faithful if math.isfinite(t.reward)]
        batch.num_dropped_untrainable += len(faithful) - len(trainable)

        rows: list[TrainRow] = []
        used: list[Trajectory] = []
        for traj in trainable:
            traj_rows = _rows_for_trajectory(
                traj, group, unit=unit, fork_policy=fork_policy, batch=batch
            )
            if not traj_rows:
                batch.num_dropped_untrainable += 1
                continue
            over_max = [
                row
                for row in traj_rows
                if max_seq_length is not None and row.total_len > max_seq_length
            ]
            batch.num_rows_over_max_len += len(over_max)
            if over_max:
                if sequence_overflow == "error":
                    row = over_max[0]
                    raise ValueError(
                        f"trajectory {traj.trajectory_id} has total_len {row.total_len} "
                        f"exceeding max_seq_length {max_seq_length}"
                    )
                batch.num_dropped_over_max_len += 1
                continue
            rows.extend(traj_rows)
            used.append(traj)

        if len(used) < min_group_size:
            batch.num_dropped_small_group += len(used)
            continue

        rewards = [t.reward for t in used]
        group_reward_means.append(statistics.fmean(rewards))
        group_reward_stds.append(statistics.stdev(rewards) if len(rewards) > 1 else 0.0)
        if len(set(rewards)) == 1:
            batch.num_zero_std_groups += 1

        pending.append((len(pending), rewards, rows))
        batch.num_groups_used += 1
        batch.num_trajectories_used += len(used)
        trajectory_rewards.extend(rewards)

    batch_std = statistics.stdev(trajectory_rewards) if len(trajectory_rewards) > 1 else 0.0
    final_rows: list[TrainRow] = []
    for group_index, traj_rewards, rows in pending:
        # One entry per trajectory, never per row: with several rows per trajectory a
        # row-based mean/std would weight long trajectories more (rewards 0 and 1 with 1 and 3
        # rows gave -0.75/+0.25 instead of -0.5/+0.5). Each trajectory's advantage is then
        # broadcast to all of its rows, because row.reward is its trajectory's reward.
        mean = statistics.fmean(traj_rewards)
        std = None
        if estimator == "group_zscore":
            std = (
                batch_std
                if advantage.std_scope == "batch"
                else (statistics.stdev(traj_rewards) if len(traj_rewards) > 1 else 0.0)
            )
        group_advantages = []
        for row in rows:
            adv = row.reward - mean
            if std is not None:
                adv = adv / (std + _STD_EPS)
            if estimator == "rloo":
                adv *= len(traj_rewards) / (len(traj_rewards) - 1)
            group_advantages.append(adv)
            final_rows.append(
                TrainRow(
                    trajectory_id=row.trajectory_id,
                    group_id=row.group_id,
                    group_index=group_index,
                    prompt_ids=row.prompt_ids,
                    completion_ids=row.completion_ids,
                    sampling_logprobs=row.sampling_logprobs,
                    action_mask=row.action_mask,
                    reward=row.reward,
                    advantage=float(adv),
                    unit=row.unit,
                    sub_index=row.sub_index,
                    behavior_version=row.behavior_version,
                    truncated=row.truncated,
                )
            )

        batch.num_zero_advantage_groups += int(all(a == 0 for a in group_advantages))

    batch.rows = final_rows
    if group_reward_means:
        batch.group_reward_mean = statistics.fmean(group_reward_means)
        batch.group_reward_std_mean = statistics.fmean(group_reward_stds)
    return batch


def would_train_group(
    group: RolloutGroup,
    *,
    unit: CallUnit,
    fork_policy: ForkPolicy,
    max_seq_length: int | None,
    advantage: AdvantageSpec,
    mask_truncated: bool = False,
    kl_active: bool = False,
) -> bool:
    """Would ``train_step([group])`` produce a row that carries loss?

    Mirrors the row builder, then applies the same truncation rule the shim applies through
    ``tool_mask``: with ``mask_truncated`` a row whose completion hit ``max_tokens`` gets no
    loss. A group whose
    only non-zero advantages sit on masked rows is not trainable.
    When ``kl_active`` is true, any live row remains trainable because KL can supply a gradient.
    ``max_seq_length`` is diagnostic, as in ``build_batch``."""
    batch = build_batch(
        [group],
        unit=unit,
        fork_policy=fork_policy,
        max_seq_length=max_seq_length,
        advantage=advantage,
    )
    live = [row for row in batch.rows if not (mask_truncated and row.truncated)]
    if not live:
        return False
    if kl_active:
        return True
    return any(row.advantage != 0.0 for row in live)


def _rows_for_trajectory(
    traj: Trajectory,
    group: RolloutGroup,
    *,
    unit: CallUnit,
    fork_policy: ForkPolicy,
    batch: RolloutBatch,
) -> list[TrainRow]:
    if unit == "call":
        return _rows_from_calls(traj, group)
    return _rows_from_sequences(traj, group, fork_policy=fork_policy, batch=batch)


def _sequence_truncated(traj: Trajectory, seq: TrainingSequence) -> bool:
    """A sequence is truncated when the model call that ended *it* hit ``max_tokens``. With
    forks (``fork_policy`` ``all``/``longest``) each branch has its own last call, so look it
    up through ``seq.model_call_indices`` rather than the trajectory's last call."""
    wanted = set(seq.model_call_indices)
    calls = [c for c in traj.model_calls if c.completion_ids and c.index in wanted]
    if not calls:
        calls = [c for c in traj.model_calls if c.completion_ids]
    return bool(calls) and calls[-1].finish_reason == "length"


def _rows_from_calls(traj: Trajectory, group: RolloutGroup) -> list[TrainRow]:
    rows: list[TrainRow] = []
    for call in traj.model_calls:
        if not call.completion_ids or not call.has_logprobs:
            continue
        rows.append(
            TrainRow(
                trajectory_id=traj.trajectory_id,
                group_id=group.group_id,
                group_index=0,
                prompt_ids=list(call.prompt_ids),
                completion_ids=list(call.completion_ids),
                sampling_logprobs=list(call.sampling_logprobs),
                action_mask=[True] * len(call.completion_ids),
                reward=float(traj.reward),
                advantage=0.0,
                unit="call",
                sub_index=call.index,
                behavior_version=group.policy_version,
                truncated=call.finish_reason == "length",
            )
        )
    return rows


def _rows_from_sequences(
    traj: Trajectory,
    group: RolloutGroup,
    *,
    fork_policy: ForkPolicy,
    batch: RolloutBatch,
) -> list[TrainRow]:
    usable = _usable_sequences(traj)
    if not usable:
        return []
    if len(usable) > 1:
        batch.num_forked_trajectories += 1
        if fork_policy == "error":
            raise ValueError(
                f"trajectory {traj.trajectory_id!r} has {len(usable)} training sequences "
                "(verifiers forked because a turn re-tokenised). Unit 'sequence' cannot map it "
                "to a single row. Choose call_unit='call' (lossless, one row per model call), "
                "or set fork_policy='all' (one row per branch, reward broadcast) or "
                "fork_policy='longest' (keep the branch with the most action tokens)."
            )
        if fork_policy == "longest":
            best = max(usable, key=lambda s: s.num_action_tokens)
            batch.num_dropped_forked_branches += len(usable) - 1
            usable = [best]

    rows: list[TrainRow] = []
    for seq in usable:
        start = seq.prompt_len
        action_mask = list(seq.action_mask[start:])
        sampling_logprobs = [
            logprob if action_mask[position] else 0.0
            for position, logprob in enumerate(seq.sampling_logprobs[start:])
        ]
        rows.append(
            TrainRow(
                trajectory_id=traj.trajectory_id,
                group_id=group.group_id,
                group_index=0,
                prompt_ids=list(seq.token_ids[:start]),
                completion_ids=list(seq.token_ids[start:]),
                sampling_logprobs=sampling_logprobs,
                action_mask=action_mask,
                reward=float(traj.reward),
                advantage=0.0,
                unit="sequence",
                sub_index=seq.index,
                behavior_version=group.policy_version,
                truncated=_sequence_truncated(traj, seq),
            )
        )
    return rows


def _usable_sequences(traj: Trajectory) -> list[TrainingSequence]:
    """Sequences eligible for row construction, before applying the fork policy."""
    return [
        sequence
        for sequence in traj.sequences
        if sequence.num_action_tokens > 0
        and len(sequence.sampling_logprobs) == len(sequence.token_ids)
    ]


__all__ = [
    "AdvantageSpec",
    "would_train_group",
    "MIN_GROUP_SIZE",
    "ForkPolicy",
    "RolloutBatch",
    "ScaleRewards",
    "TrainRow",
    "build_batch",
]
