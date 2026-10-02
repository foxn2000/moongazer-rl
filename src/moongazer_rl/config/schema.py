"""The run config: a THIN launcher config, not a copy of Unsloth/TRL/verifiers settings.

Design rule (no YAML for the sake of YAML): our own sections
(``run``, ``model`` minus ``renderer``, ``rollout`` minus ``sampling``, ``logging``,
``checkpoint``, ``eval``) are small and ``extra="forbid"`` — every key is one we actually
read. Everything that belongs to an upstream library is an opaque passthrough dict, typed
``dict[str, Any]``, validated lazily against the installed upstream API by
``config.validate``:

* ``environment`` -> ``verifiers.v1.EnvConfig`` (only ``taskset.id`` / ``harness.id`` are
  structurally required here; everything else, including env-specific taskset fields such
  as ``split``, passes through untouched).
* ``rollout.sampling`` -> ``verifiers.v1.Sampling``.
* ``model.renderer`` -> the ``renderers`` package's per-model config.
* ``trainer.args`` -> the resolved ``TrainerBackend``'s own args (e.g. TRL ``GRPOConfig``
  for ``unsloth-native``).

Nothing here imports torch, verifiers, trl, or unsloth.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Section(BaseModel):
    """Our own config sections: every key is declared, unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid")


class RunSection(_Section):
    name: str = Field(min_length=1, description="Run name; also the default runs/<name> dir")
    seed: int = 42


class LoraSection(_Section):
    """LoRA adapter shape. It lives in our schema rather than inside the opaque
    ``trainer.args``."""

    r: int = Field(ge=1)
    alpha: float = Field(gt=0)
    dropout: float = Field(ge=0.0, le=1.0, default=0.0)
    target_modules: list[str] | None = None


class ModelSection(_Section):
    name: str = Field(min_length=1, description="HF hub id or local path")
    load_in_4bit: bool = False
    text_only_offload: bool = Field(
        default=True,
        description=(
            "Move Gemma 4 E-series per-layer embeddings and vision/audio towers to CPU for "
            "text-only training; set false when training image or audio inputs"
        ),
    )
    max_seq_length: int | None = Field(default=None, ge=1)
    gradient_checkpointing: Literal["unsloth", "torch", "off", "auto"] = Field(
        default="unsloth",
        description=(
            "unsloth = Unsloth's offloaded checkpointing, torch = standard recompute-only "
            "checkpointing, off = no checkpointing, auto = checkpoint by micro-batch size"
        ),
    )
    gradient_checkpointing_auto_max_tokens: int = Field(
        default=4096,
        ge=1,
        description=(
            "auto: micro-batches whose padded tokens (rows x widest row) are at most this "
            "many run without checkpointing; longer ones checkpoint"
        ),
    )
    lora: LoraSection | None = None
    renderer: dict[str, Any] = Field(
        default_factory=dict, description="Opaque passthrough to the renderers package config"
    )


class TasksetSection(BaseModel):
    """``environment.taskset``. Accepts verifiers' native flat shape (env-specific fields,
    e.g. ``split``, sit directly on the taskset dict) *and* a nested ``args`` mapping for
    convenience; ``EnvironmentSection.to_env_config_dict`` normalizes both to verifiers'
    flat form (verifiers 0.2.0's ``TasksetConfig`` has no ``args`` field of its own —
    env-specific ``Taskset`` subclasses add their fields flat on the taskset config)."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _id_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("environment.taskset.id must not be empty")
        return value


class HarnessSection(BaseModel):
    """``environment.harness``. ``runtime`` is opaque (subprocess/docker/prime/modal each
    have their own fields in verifiers)."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(default="default", min_length=1)
    runtime: dict[str, Any] = Field(default_factory=dict)


class EnvironmentSection(BaseModel):
    """Passthrough to ``verifiers.v1.EnvConfig``. Only ``taskset.id`` / ``harness.id`` are
    checked here; everything else (including unknown top-level keys) passes through as-is
    and is validated lazily by ``config.validate`` against the installed ``verifiers``."""

    model_config = ConfigDict(extra="allow")

    taskset: TasksetSection
    harness: HarnessSection = Field(default_factory=HarnessSection)
    max_turns: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    timeout: dict[str, Any] = Field(default_factory=dict)
    retries: dict[str, Any] = Field(default_factory=dict)
    multiplex: int = Field(default=32, ge=1)

    def to_env_config_dict(self) -> dict[str, Any]:
        """The verifiers-shaped dict: ``taskset`` flattened (``args`` merged onto it, flat
        keys already there win), everything else passed through as given."""
        taskset_dict = self.taskset.model_dump(exclude={"args"})
        for key, value in self.taskset.args.items():
            taskset_dict.setdefault(key, value)
        result: dict[str, Any] = {
            "taskset": taskset_dict,
            "harness": self.harness.model_dump(),
            "max_turns": self.max_turns,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_total_tokens": self.max_total_tokens,
            "timeout": self.timeout,
            "retries": self.retries,
            "multiplex": self.multiplex,
        }
        if self.model_extra:
            result.update(self.model_extra)
        return result


class RolloutSection(_Section):
    num_generations: int = Field(ge=1, description="G rollouts per task = GRPO group size")
    groups_per_step: int = Field(ge=1, description="Tasks per optimizer step")
    max_concurrent: int = Field(default=32, ge=1)
    num_epochs: int | None = 1
    max_steps: int | None = Field(default=None, ge=1)
    sampling: dict[str, Any] = Field(
        default_factory=dict, description="Opaque passthrough to verifiers.v1.Sampling"
    )


class MemorySection(_Section):
    """Memory-efficiency switches of the ``unsloth-native`` learner. All of them are exact
    (same loss, same gradients up to floating-point rounding); they only change peak memory.
    ``token_chunked_logprobs`` and ``score_rows_one_at_a_time`` must be switched together: both
    compute positions on padding-trimmed rows, and mixing one with Unsloth's padded fallback would
    pair old and new log-probabilities from two different position conventions."""

    blocked_global_attention: bool = Field(
        default=True,
        description=(
            "Route Gemma 4 global (head_dim > 256) causal attention through an exact "
            "flex or query-blocked implementation instead of math SDPA (sliding layers follow sliding_attention)"
        ),
    )
    global_attention: Literal["flex", "blocked"] = Field(
        default="flex", description="Global attention implementation; flex falls back to blocked"
    )
    sliding_attention: Literal["flex", "unsloth"] = Field(
        default="flex", description="Sliding attention implementation; flex falls back to Unsloth"
    )
    materialize_sliding_mask: bool = Field(
        default=False,
        description=(
            "Build the dense Gemma 4 sliding-window mask; skipped only when sliding_attention "
            "is flex (Unsloth's own sliding path needs the mask)"
        ),
    )
    query_chunk_size: int = Field(
        default=1024,
        ge=1,
        description=(
            "Query block of the blocked attention"
        ),
    )
    min_global_q_len: int = Field(
        default=1024, ge=1, description="Sequences shorter than this keep the stock SDPA"
    )
    token_chunked_logprobs: bool = Field(
        default=True,
        description=(
            "Compute per-token log-probabilities from the final hidden states in token chunks "
            "(never a full [S, vocab] tensor), with per-chunk recompute in backward"
        ),
    )
    logprob_chunk_size: int = Field(default=64, ge=1)
    score_rows_one_at_a_time: bool = Field(
        default=True,
        description="Old/ref log-probability scoring runs one row per forward",
    )


AdvantageEstimator = Literal["group_zscore", "group_center", "rloo"]
PolicyObjective = Literal["grpo_clip", "cispo", "reinforce"]
RatioReference = Literal["proximal", "behavior"]
BehaviorCorrection = Literal["none", "truncate", "mask"]
LossNormalization = Literal["sequence", "action_token", "fixed_length"]


class AlgorithmSection(_Section):
    """The RL objective, decomposed (docs/algorithms.md). Every field is optional; the defaults
    reproduce the GRPO path this repository shipped with (``loss_type: dapo``). A legacy
    ``trainer.args`` key (``loss_type``, ``scale_rewards``, ``epsilon*``,
    ``vllm_importance_sampling_cap``) that disagrees with an explicit value here is an error."""

    advantage_estimator: AdvantageEstimator | None = Field(
        default=None,
        description=(
            "group_zscore: (r - group mean) / (std + 1e-4) [GRPO]; group_center: r - group mean; "
            "rloo: r_i - mean of the other G-1 rewards"
        ),
    )
    std_scope: Literal["group", "batch"] | None = Field(
        default=None,
        description="group_zscore only: which std divides (legacy scale_rewards=batch)",
    )
    policy_objective: PolicyObjective | None = Field(
        default=None,
        description=(
            "grpo_clip: PPO-style clipped ratio; cispo: -sg(min(r, eps_high)) A logp; "
            "reinforce: -A logp"
        ),
    )
    ratio_reference: RatioReference | None = Field(
        default=None,
        description=(
            "proximal: ratio against the learner before this optimizer step (scoring forward); "
            "behavior: ratio against the sampling (vLLM) logprobs, no scoring forward"
        ),
    )
    behavior_correction: BehaviorCorrection | None = Field(
        default=None,
        description=(
            "Off-policy correction against the sampling logprobs: none; truncate (weight "
            "min(exp(learner - sampling), behavior_cap)); mask (zero tokens whose weight leaves "
            "[behavior_clip_min, behavior_clip_max])"
        ),
    )
    behavior_cap: float | None = Field(default=None, gt=0.0)
    behavior_clip_min: float | None = Field(default=None, ge=0.0)
    behavior_clip_max: float | None = Field(default=None, gt=0.0)
    loss_normalization: LossNormalization | None = Field(
        default=None,
        description=(
            "Global denominator shared by every micro-batch of a step: sequence (mean over rows "
            "of the per-row token mean), action_token (all action tokens of the step), "
            "fixed_length (rows x fixed_length)"
        ),
    )
    fixed_length: int | None = Field(default=None, ge=1)
    skip_old_logprob_forward: Literal["auto"] | bool = Field(
        default="auto",
        description=(
            "Skip the no-grad scoring forward that recomputes the learner's pre-update logprobs "
            "when they are provably identical to the training forward (one optimizer step per "
            "batch, LoRA dropout 0, ratio_reference behavior or proximal)"
        ),
    )


class BatchSection(_Section):
    """Token-budget micro-batch planner (trainer/unsloth_native/planner.py). Every limit is
    optional; ``max_rows`` defaults to ``trainer.args.per_device_train_batch_size``. A
    ``RolloutGroup`` is never split across micro-batches while ``keep_groups_whole`` is set."""

    max_model_tokens: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Cap on the padded token volume of one micro-batch: rows x the widest row "
            "(prompt + completion), which is what the forward actually processes"
        ),
    )
    max_action_tokens: int | None = Field(default=None, ge=1)
    max_rows: int | None = Field(default=None, ge=1)
    max_groups: int | None = Field(default=None, ge=1)
    keep_groups_whole: bool = True
    length_bucketing: bool = True


class TrainerSection(_Section):
    backend: str = Field(min_length=1, description="Name in trainer.registry._BACKENDS")
    args: dict[str, Any] = Field(
        default_factory=dict, description="Opaque passthrough to the backend's own args"
    )
    memory: MemorySection = Field(default_factory=MemorySection)
    algorithm: AlgorithmSection = Field(default_factory=AlgorithmSection)
    batch: BatchSection = Field(default_factory=BatchSection)
    call_unit: Literal["sequence", "call"] = Field(
        default="sequence",
        description=(
            "How a trajectory becomes training rows: 'sequence' = one row per verifiers branch "
            "(tool/user tokens masked), 'call' = one row per model call (exactly the sampled ids)"
        ),
    )
    fork_policy: Literal["error", "all", "longest"] = Field(
        default="error",
        description=(
            "What to do when a trajectory has several branches (a turn re-tokenised): error, "
            "all (one row per branch), longest (keep the branch with the most action tokens)"
        ),
    )
    sequence_overflow: Literal["error", "drop"] = Field(
        default="error",
        description=(
            "What to do when a training row exceeds model.max_seq_length: error or drop the "
            "containing trajectory before advantage calculation"
        ),
    )


class PolicySection(_Section):
    """Where rollouts are sampled. ``auto`` (default) lets the trainer backend pick its own
    in-process policy (HF ``model.generate`` on the learner's weights). ``vllm-server``
    samples from an external ``vllm serve`` process and hot-loads the exported LoRA adapter
    after every optimizer step; ``args`` are the ``policy.vllm_server.VllmServerPolicy``
    keyword arguments (``base_url``, ``model``, ``logprobs_mode``, ``timeout_s``, ...)."""

    backend: str = Field(default="auto", min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)


class LoggingSection(_Section):
    output_dir: str | None = Field(default=None, description="Defaults to runs/<run.name>")
    wandb: bool = False
    save_token_ids: bool = True


class CheckpointSection(_Section):
    save_steps: int | None = Field(default=10, ge=1)
    keep_last: int | None = Field(default=3, ge=1)


class EvalSection(_Section):
    every_steps: int | None = Field(default=10, ge=1)
    num_tasks: int | None = Field(default=None, ge=1)
    taskset: dict[str, Any] | None = Field(
        default=None, description="Overrides environment.taskset for eval; same shape"
    )
    checkpoint: str | None = Field(
        default="latest",
        description=(
            "Standalone `mg-rl eval`: checkpoint directory to restore before evaluating; "
            "'latest' picks the newest under <output_dir>/checkpoints, null evaluates the "
            "freshly initialised model"
        ),
    )


class AsyncSection(_Section):
    """Controls the fully asynchronous producer/learner execution path.

    ``max_staleness`` bounds the accepted learner/behavior version gap;
    ``queue_capacity`` is the bounded queue size in rollout groups; and
    ``num_producers`` is the number of concurrent episodes. ``drop_zero_std_groups``
    avoids optimizer steps with fewer than two usable trajectories or only equal rewards.
    ``max_consecutive_zero_std_drops`` controls periodic warnings or stopping for an
    unbounded task stream with no trainable groups. ``max_no_progress_s`` optionally bounds
    wall-clock time without an optimizer step. ``publish_retries`` and
    ``publish_retry_backoff_s`` control adapter publish recovery, while
    ``max_consecutive_rollout_failures`` stops a persistently failing producer.
    ``shutdown_timeout_s`` bounds graceful shutdown and ``unload_grace_s`` keeps retired
    adapters available while cancelled requests finish disconnecting.
    """

    max_staleness: int = Field(default=1, ge=0)
    queue_capacity: int = Field(default=4, ge=1)
    num_producers: int = Field(default=2, ge=1)
    drop_zero_std_groups: bool = Field(
        default=True,
        description=(
            "DAPO-style dynamic sampling: drop a group with fewer than two trainable "
            "trajectories or all-equal rewards, either of which cannot produce non-zero GRPO "
            "advantages, and keep collecting"
        ),
    )
    max_consecutive_zero_std_drops: int = Field(
        default=16,
        ge=1,
        description=(
            "Emit zero_std_budget_exhausted after this many consecutive dynamic-sampling "
            "drops, and again at each multiple; zero-advantage groups are never force-trained"
        ),
    )
    zero_std_budget_action: Literal["warn", "stop"] = Field(
        default="warn",
        description=(
            "On each zero-std drop-budget boundary, keep dropping with a warning or stop the run"
        ),
    )
    max_no_progress_s: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Stop after this many wall-clock seconds without an optimizer step, measured from "
            "the first completed rollout until the first step"
        ),
    )
    publish_retries: int = Field(default=3, ge=0)
    publish_retry_backoff_s: float = Field(default=2.0, ge=0)
    max_consecutive_rollout_failures: int = Field(default=5, ge=1)
    shutdown_timeout_s: float = Field(default=600.0, gt=0)
    unload_grace_s: float = Field(default=2.0, ge=0)
    eval_every_steps: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Versioned asynchronous evaluation: every N learner steps, lease the published policy "
            "version and evaluate eval.num_tasks tasks on it without pausing the learner; results "
            "are written under eval-v<version>/ and tagged with that version"
        ),
    )


class RunConfig(BaseModel):
    """The full run config. Section shape is documented at the top of this module."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    run: RunSection
    model: ModelSection
    environment: EnvironmentSection
    rollout: RolloutSection
    trainer: TrainerSection
    policy: PolicySection = Field(default_factory=PolicySection)
    logging: LoggingSection = Field(default_factory=LoggingSection)
    checkpoint: CheckpointSection = Field(default_factory=CheckpointSection)
    eval: EvalSection = Field(default_factory=EvalSection)
    async_rl: AsyncSection | None = Field(
        default=None,
        alias="async",
        description="Enable and configure the fully asynchronous rollout/training path",
    )

    @model_validator(mode="after")
    def _default_output_dir(self) -> RunConfig:
        if self.logging.output_dir is None:
            self.logging.output_dir = f"runs/{self.run.name}"
        return self


__all__ = [
    "AdvantageEstimator",
    "AlgorithmSection",
    "AsyncSection",
    "BatchSection",
    "BehaviorCorrection",
    "LossNormalization",
    "PolicyObjective",
    "RatioReference",
    "CheckpointSection",
    "EnvironmentSection",
    "EvalSection",
    "HarnessSection",
    "LoggingSection",
    "LoraSection",
    "MemorySection",
    "ModelSection",
    "PolicySection",
    "RolloutSection",
    "RunConfig",
    "RunSection",
    "TasksetSection",
    "TrainerSection",
]
