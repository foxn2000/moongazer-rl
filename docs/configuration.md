# Configuration and CLI

[日本語](configuration.ja.md)

## Pinned upstream versions

| package | version | note |
|---|---|---|
| verifiers | 0.2.0 | native v1 API |
| renderers | 0.1.11 | verifiers dependency; chat-template render + completion parse per model family |
| trl | 0.24.0 | `unsloth 2026.9.2` pins `trl>=0.18.2,<=0.24.0` |
| transformers | 5.5.0 | unsloth pins `<=5.5.0` |
| torch | 2.11.0 | unsloth pins `<2.12` |
| datasets | 4.3.0 | verifiers needs `<4.7`, unsloth needs `<4.4` |
| peft | 0.20.0 | |
| mcp | 1.29.1 | pinned `<2` — verifiers 0.2.0 needs `mcp.server.fastmcp`, removed in mcp 2.x |
| unsloth / unsloth_zoo | 2026.9.2 / 2026.9.1 | optional `unsloth` extra, Linux only |

## Configuration

`RunConfig` (`src/moongazer_rl/config/schema.py`) is a thin launcher config. Its own
sections are small and `extra="forbid"`; anything that belongs to an upstream library is an
opaque passthrough dict, validated lazily against the installed upstream package by
`mg-rl validate`:

| our key | passes through to |
|---|---|
| `environment` | `verifiers.v1.EnvConfig` (only `taskset.id` / `harness.id` are structurally required; env-specific taskset fields such as `split` pass through untouched) |
| `rollout.sampling` | `verifiers.v1.Sampling` |
| `trainer.args` | the resolved `TrainerBackend`'s own args (TRL `GRPOConfig` for `unsloth-native`) |
| `model.renderer` | the `renderers` package's per-model config |
| `policy.args` | `policy.vllm_server.VllmServerPolicy` keyword arguments when `policy.backend: vllm-server` |

For Gemma 4 E2B text-only training, `model.text_only_offload: true` moves the 4.37 GiB
per-layer embedding and 0.87 GiB vision/audio towers to CPU (RTX 5090, bf16, 2026-09-19).

Full YAML shape:

```yaml
run:
  name: <str>                 # also the default runs/<name> directory
  seed: 42

model:
  name: <str>                 # HF hub id or local path
  load_in_4bit: false
  text_only_offload: true     # Gemma 4 E-series PLE and vision/audio modules stay on CPU
  max_seq_length: <int | null>
  gradient_checkpointing: unsloth  # unsloth (CPU offload), torch (recompute only), off, or auto
  gradient_checkpointing_auto_max_tokens: 4096  # auto disables GC at or below this padded size
  lora:                       # optional
    r: <int>
    alpha: <float>
    dropout: 0.0
    target_modules: <list[str] | null>
  renderer: {}                 # opaque passthrough -> renderers.RendererConfig

environment:
  taskset:
    id: <str>                  # verifiers taskset plugin id
    args: {}                   # merged flat onto the taskset dict (env-specific fields, e.g. split)
  harness:
    id: default
    runtime: {}                 # e.g. {type: subprocess} / {type: docker}
  max_turns: null
  max_input_tokens: null
  max_output_tokens: null
  max_total_tokens: null
  timeout: {}
  retries: {}
  multiplex: 32

rollout:
  num_generations: <int>       # G rollouts per task = GRPO group size
  groups_per_step: <int>       # tasks per optimizer step
  max_concurrent: 32
  num_epochs: 1                # null = "until max_steps"
  max_steps: null
  sampling: {}                  # opaque passthrough -> verifiers.v1.Sampling (temperature, max_tokens, ...)

trainer:
  backend: unsloth-native
  args: {}                       # opaque passthrough -> the backend's own args
  sequence_overflow: error       # error or drop trajectories with rows over model.max_seq_length
  memory:                        # unsloth-native learner memory switches; see "Learner memory settings"
    blocked_global_attention: true
    materialize_sliding_mask: false
    query_chunk_size: 1024
    min_global_q_len: 1024
    token_chunked_logprobs: true
    logprob_chunk_size: 64
    score_rows_one_at_a_time: true

policy:
  backend: auto                  # auto = the trainer backend's in-process sampler; vllm-server = external `vllm serve`
  args: {}                       # vllm-server: base_url (required), model, logprobs_mode, timeout_s, admin_timeout_s, keep_adapters, ...

logging:
  output_dir: null              # defaults to runs/<run.name>
  wandb: false                   # true -> also logs flattened steps.jsonl rows to Weights & Biases (no-op if wandb isn't installed)
  save_token_ids: true

checkpoint:
  save_steps: 10
  keep_last: 3

eval:
  every_steps: 10               # ignored on the async path (use standalone `mg-rl eval`)
  num_tasks: null
  taskset: null                 # declared but not wired yet: eval reuses environment.taskset and slices num_tasks
  checkpoint: latest            # standalone `mg-rl eval`: "latest", a path, or null for the fresh model

async:                          # optional; omit for the synchronous path. See "Fully asynchronous training (2 GPUs)"
  max_staleness: 1
  queue_capacity: 4
  num_producers: 2
  drop_zero_std_groups: true
  max_consecutive_zero_std_drops: 16
  zero_std_budget_action: warn     # or stop
  max_no_progress_s: null
  publish_retries: 3
  publish_retry_backoff_s: 2.0
  max_consecutive_rollout_failures: 5
  shutdown_timeout_s: 600.0
  unload_grace_s: 2.0
```

`--set dotted.key=value` (repeatable) overrides any key before validation; the value is
parsed as YAML (`4` -> int, `1e-5` -> float, `[1, 2]` -> a list).

## Sampling from an external vLLM server (`policy.backend: vllm-server`)

`unsloth-native` samples in-process by default (`policy.backend: auto`): HF
`model.generate` on the learner's own weights (serial; fine for a smoke test, not for training). No vllm release resolves
inside unsloth 2026.9.x's `torch<2.12, transformers<=5.5` pins and supports Gemma 4, so the
production sampler is a separate `vllm serve` process talked to over HTTP:

```yaml
policy:
  backend: vllm-server
  args:
    base_url: http://127.0.0.1:8000
    logprobs_mode: processed_logprobs   # must match the server's --logprobs-mode
    timeout_s: 1800
```

```bash
GPU_UTIL=0.5 examples/rtx5090-math-env/serve-vllm.sh   # --enable-lora, runtime LoRA updating, processed_logprobs
```

Contract (`policy/vllm_server.py`): the renderer's prompt
token ids go to `POST /v1/completions` as `prompt: list[int]`, `return_token_ids` brings the
sampled ids back verbatim, `logprobs: 0` brings the sampled token's logprob. After every
optimizer step the learner writes its PEFT adapter to `<output_dir>/.lora-sync/v<version>/` and
the policy calls `/v1/load_lora_adapter`; requests then name `policy-v<version>` as the model.
The previous adapter stays registered for in-flight requests (`keep_adapters`, default 2). The
server must see the same filesystem as the trainer.

## Fully asynchronous training (2 GPUs)

The `async:` section switches `mg-rl train` to
`async_rl/orchestrator.py`: producer coroutines keep generating episodes (`1 task x G rollouts`
each) into a bounded queue while a learner loop consumes and trains; a publisher task exports
and hot-loads the learner's LoRA adapter without blocking either side.

```text
GPU 0: vllm serve --enable-lora        GPU 1: mg-rl train (async: section)
  producer x N --lease--> generate       learner: train_step -> export -> publish (non-blocking)
```

| key | default | meaning |
|---|---|---|
| `max_staleness` | 1 | max accepted `learner_version - behavior_version` gap; 0 removes rollout/train overlap (barrier-equivalent) |
| `queue_capacity` | 4 | bounded queue size in `RolloutGroup`s; `put()` blocks past this (backpressure) |
| `num_producers` | 2 | concurrent episodes |
| `drop_zero_std_groups` | true | DAPO-style dynamic sampling: replace groups with fewer than 2 trainable trajectories or all-equal rewards |
| `max_consecutive_zero_std_drops` | 16 | after this many consecutive dynamic-sampling drops (and at every further multiple) log `zero_std_budget_exhausted`; zero-advantage groups are never force-trained |
| `zero_std_budget_action` | warn | `warn` keeps dropping, `stop` fails the run when the budget is exhausted |
| `max_no_progress_s` | null | stop the run after this many wall-clock seconds without an optimizer step (measured from the first finished rollout) |
| `publish_retries` | 3 | adapter-publish retry attempts after a load failure |
| `publish_retry_backoff_s` | 2.0 | backoff between publish retries |
| `max_consecutive_rollout_failures` | 5 | stop the run after this many consecutive producer failures |
| `shutdown_timeout_s` | 600.0 | graceful-shutdown deadline |
| `unload_grace_s` | 2.0 | keep a retired adapter resident this long after its refcount hits 0 (lets in-flight HTTP requests finish disconnecting) |

Semantics:

* **Staleness** is `learner_version_at_consume - behavior_version`. Producers reserve an
  admission slot before starting an episode so that
  `learner_version + reserved_slots // groups_per_step - published_version <= max_staleness`;
  a dequeued group whose actual staleness exceeds `max_staleness` is dropped at the gate
  (`stale_group_dropped`). The gate, not the admission estimate, guarantees the bound.
* **Lease**: acquired once per episode and held for its whole lifetime, so every rollout and
  every turn in one episode is generated by the same adapter version. A trajectory never
  straddles two policy versions.
* **Dynamic sampling**: with `drop_zero_std_groups: true` a group with fewer than two trainable
  trajectories or all-equal rewards is committed and dropped instead of producing a no-gradient
  step. When the advantage is group-local (`group_zscore` / `group_center` / `rloo`,
  `std_scope: group`, `beta: 0`) the producer drops it right after the rollout and returns the
  admission slot at once (`zero_std_group_dropped.stage: producer`); otherwise the learner
  decides at dequeue. A batch whose advantages are all exactly zero (with `beta: 0`) is refused
  before any forward (`train_skipped`): no optimizer step, no `weight_version` bump, no publish.
  `max_consecutive_zero_std_drops` only warns (or stops with `zero_std_budget_action: stop`);
  `max_no_progress_s` bounds the wall clock without an optimizer step.
* **Fidelity gate**: `build_batch` verifies the loss-input view of every trajectory
  (`verify_training_fidelity`: token/logprob alignment, finite logprobs, every sampled token
  covered by a recorded call, one policy version per group) and drops violators before the group
  baseline is computed; `fidelity_violation` events and `num_dropped_fidelity` record it.
* **Off-policy correction** is truncated importance sampling only: the token weight
  `min(exp(learner_old - sampling), cap)` (default cap 2.0). It cannot restore support removed
  at sampling time, so the async path requires `temperature > 0` and rejects `top_k`,
  `top_p != 1`, `min_p`, `logit_bias`, `bad_words` and `allowed_token_ids`
  (`config/validate.py::_validate_async`). `policy.backend` must be `vllm-server`.
  `eval.every_steps` is ignored (the run records
  `inline_eval_disabled`).
* **Checkpoint / resume**: checkpoints are taken between learner steps. `orchestrator_state.json`
  records `task_cursor` and `uncommitted_task_positions`; on resume the queue starts empty,
  producers re-sample the uncommitted positions first, the restored adapter is published under a
  new `<gen>` name (stale `policy-*` adapters are unloaded) and confirmed before producers start.

Launch: `examples/rtx5090-math-env/` (Prime Intellect `math-env`, sampler and learner on one RTX 5090).

`events.jsonl` gains `rollout_started`/`rollout_finished`, `group_enqueued`/`group_dequeued`,
`stale_group_dropped`, `train_started`/`train_finished`, `publish_*`, `adapter_unloaded`,
`producer_idle`/`learner_idle`; `steps.jsonl` rows gain an `async` block (`behavior_versions`,
`staleness_min/max/mean`, `queue_wait_s_mean`, `queue_size_at_dequeue`, `learner_idle_s`,
`published_version`, `adapter_export_s`). `mg-rl async-report <run_dir> [--json]` summarizes
a run from these events: rollout/train wall-clock overlap, staleness histogram, publish latency,
throughput and queue occupancy.

## RL objective, batch planner, replicas, versioned evaluation

[algorithms.md](algorithms.md) documents the decomposition; every key is optional and the defaults are the
GRPO path this repository shipped with (`trainer.args.loss_type: dapo`).

```yaml
trainer:
  algorithm:
    advantage_estimator: group_zscore   # group_zscore | group_center | rloo
    policy_objective: grpo_clip         # grpo_clip | cispo | reinforce
    ratio_reference: proximal           # proximal (pre-update learner) | behavior (sampling logprobs)
    behavior_correction: truncate       # none | truncate | mask   (weight min(exp(learner - sampling), behavior_cap))
    loss_normalization: action_token    # sequence | action_token | fixed_length (global denominator, ragged-safe)
    skip_old_logprob_forward: auto      # skip the no-grad scoring forward when provably identical
  batch:
    max_model_tokens: 32768             # token-budget micro-batches; a RolloutGroup is never split
    max_rows: 8
policy:
  args:
    base_urls: [http://127.0.0.1:8101, http://127.0.0.1:8102]   # several vLLM replicas (or base_url)
async:
  eval_every_steps: 6                   # evaluate the published version under a lease; eval-v<N>/
```

Legacy `trainer.args` (`loss_type`, `scale_rewards`, `epsilon*`, `vllm_importance_sampling_cap`)
keep their meaning and are mapped onto the same spec; a disagreement with an explicit
`trainer.algorithm` value is a validation error. All objectives run through unsloth_zoo's fused
loss on GPU; only the advantage computation and the loss kwargs change.

## Learner memory settings (`trainer.memory`, `unsloth-native`)

All switches are exact (same loss and gradients up to floating-point rounding); they change peak
memory only.

| key | default | effect |
|---|---|---|
| `blocked_global_attention` | true | enable exact flex/blocked Gemma 4 global causal attention; false disables both |
| `global_attention` | flex | `flex` (fallback to `blocked`) or `blocked`; supplied RTX 5090 bf16 S=16384 forward+backward: 147.7 ms / +0.13 GB vs blocked(1024) 216.6 ms / +2.38 GB |
| `sliding_attention` | flex | `flex` (32×32 on GPUs below 128 KiB shared memory), or `unsloth`; supplied RTX 5090 B=1 bf16 D=256 forward+backward at S=4096/8192: 0.9/1.7 ms vs dense-mask memory-efficient SDPA 15.1/51.4 ms; failures fall back to Unsloth |
| `materialize_sliding_mask` | false | skip the unused dense sliding mask (862 MiB at S=30,000 on Gemma 4 E2B, RTX 5090, 2026-09-19); true restores stock materialization |
| `query_chunk_size` | 1024 | query block size for `blocked` |
| `min_global_q_len` | 1024 | sequences shorter than this keep the stock SDPA |
| `token_chunked_logprobs` | true | per-token log-probabilities from the final hidden states in token chunks (never a full `[S, vocab]` tensor), recomputed per chunk in backward |
| `logprob_chunk_size` | 64 | tokens per chunk |
| `score_rows_one_at_a_time` | true | old/ref log-probability scoring runs one row per forward; must equal `token_chunked_logprobs` |

Requires a dedicated pad token distinct from eos.

## Optimizer (`unsloth-native`)

The optimizer is whatever `trainer.args` asks TRL for: `optim` / `optim_args` / `learning_rate` /
`weight_decay` pass through to `GRPOConfig` and `Trainer.create_optimizer()`. The default under
Unsloth is `adamw_8bit`; vanilla SGD is `optim: sgd`, `optim_args: "momentum=0"`,
`weight_decay: 0.0` (the LR scale differs by two to three orders of magnitude).
Every run logs an `optimizer` event (name, torch class, hyper-parameters, trainable-parameter
count) and stores the same block in each checkpoint's `trainer_state.json`; every step reports
`learner_step_peak_memory_gb` and `learner_peak_memory_gb`.

## Controlled `trainer.args` keys (`unsloth-native`)

`trainer/unsloth_native/compat/trl_0_24.py::CONTROLLED_ARGS`. A `reject`ed key raises at
`validate`/`setup` time; a `warn`ed key is kept but prints a warning.

| key | policy | why |
|---|---|---|
| `num_generations` | reject | set from `rollout.num_generations` |
| `num_iterations` | reject | must be 1: one optimizer step per `train_step` |
| `gradient_accumulation_steps` | reject | must be 1; accumulate by sending more groups |
| `steps_per_generation` | reject | must be 1, same reason |
| `generation_batch_size` | reject | derived internally |
| `use_vllm`, `vllm_mode` | reject | generation is done by the rollout policy, never by the trainer |
| `use_liger_loss` | reject | bypasses `_compute_loss`, so the action mask would be ignored |
| `remove_unused_columns` | reject | the shim bypasses the dataloader entirely |
| `output_dir` | reject | set from `logging.output_dir` |
| `report_to` | reject | the run logger owns metric reporting |
| `seed` | reject | set from `run.seed` |
| `temperature` | match | must equal `rollout.sampling.temperature` |
| `max_completion_length` | warn | defaults to `model.max_seq_length`; does not truncate completions, only normalises the `dr_grpo` loss |
| `mask_truncated_completions` | warn | re-implemented from the renderer-aware `finish_reason == "length"` |

Unknown keys are rejected with a "did you mean" suggestion. Everything else passes straight to
`trl.GRPOConfig`, validated against the fields the installed TRL/Unsloth version declares.
For Gemma 4 E-series on transformers 5.5.0, `per_device_train_batch_size` must be 1.

## CLI

`mg-rl {validate,train,eval,resume,inspect,async-report}`. `--help` and `validate` work
without torch or verifiers plugins installed.

```bash
uv run mg-rl validate <config.yaml>
uv run mg-rl train <config.yaml> --set rollout.max_steps=2
uv run mg-rl eval <config.yaml>                    # standalone eval, never writes steps.jsonl
uv run mg-rl resume <run_dir> --set rollout.max_steps=4

uv run mg-rl inspect <run_dir>                     # summary.json + last 5 steps
uv run mg-rl inspect <run_dir> --steps --json
uv run mg-rl inspect <run_dir> --rollouts --last 3
uv run mg-rl inspect <run_dir> --events --last 3

uv run mg-rl async-report <run_dir> [--json]       # overlap / staleness / publish summary
```

`resume` restores from the latest checkpoint's batch position, never from `steps.jsonl`'s last
row; a run with logged steps but no checkpoint refuses to resume.
