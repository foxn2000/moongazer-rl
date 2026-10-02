# Run directory, logs and metrics

Every file `mg-rl train` writes, the row keys of the jsonl logs, the resume policy and the optional Weights & Biases sink.

```text
runs/<run-name>/
  config.yaml            # as given
  config.resolved.yaml   # after defaults + CLI overrides
  runtime.json           # versions, platform, git, tokenizer/template fingerprint when known
  rollouts.jsonl         # 1 line / trajectory
  traces.jsonl           # trace.to_record() (raw verifiers trace: nodes[].token_ids/mask, rewards, ...)
  model_calls.jsonl      # 1 line / ModelCall
  steps.jsonl            # 1 line / train step
  events.jsonl           # lifecycle, errors, resume, checkpoint
  summary.json           # aggregate (rewritten atomically after every step)
  lineage.json           # source run / resume history
  checkpoints/step-<n>/   # backend's own checkpoint files + orchestrator_state.json (step, learner_global_step, weight_version)
  eval.jsonl, eval_rollouts.jsonl, eval_traces.jsonl   # inline eval (config.eval.every_steps); kept out of rollouts.jsonl/traces.jsonl
```

The fully asynchronous path (`async_rl/orchestrator.py`) writes the same files, plus an
`async` block on each `steps.jsonl` row and extra `orchestrator_state.json` keys
(`published_version`, `run_gen`, `task_cursor`, `uncommitted_task_positions`) — see "Fully
asynchronous training (2 GPUs)" in [configuration.md](configuration.md).

`RunLogger.open_for_resume` repairs at most a truncated **last** line of each jsonl file (an
interrupted append): the damaged tail is quarantined to `<name>.partial.<UTC>.json` next to
the file and an `resume_repair` event is logged; a bad line anywhere *before* the last one
raises instead of silently truncating real data.

## Resume policy

The resume point is the **latest checkpoint's** batch position, never the log's last step.
`orchestrator._restore_for_resume` reads `step` out of
`checkpoints/step-<n>/orchestrator_state.json` (falling back to parsing the `step-<n>`
directory name for a checkpoint written before this file existed) and resumes from there —
never from `steps.jsonl`'s last row. `steps.jsonl` can run ahead of the checkpoint (a step
trained but never saved, between two `checkpoint.save_steps` intervals); those batches are
trained again on resume, since their weight updates were never persisted, and an
`events.jsonl` row of kind `resume_rewind` records `logged_last_step`/`checkpoint_step`
when that happens. This only occurs after a crash: a run that finishes normally always
saves its final step (when `checkpoint.save_steps` is set), even if that step is not a
multiple of the interval. Checkpoints are written to `step-<n>.tmp` and renamed on
completion, so a half-written directory is never picked up as the latest checkpoint. A run that has logged steps but has **no checkpoint at all** refuses to
resume — `train(resume=True)`/`mg-rl resume` raises `FileNotFoundError` rather than
silently trusting the log as ground truth (the trained weights are gone; start a new run
and set `checkpoint.save_steps`).

## Weights & Biases (optional)

`logging.wandb: true` builds a `WandbSink` (`logging/wandb_sink.py`) that logs a flattened
version of every `steps.jsonl` row to Weights & Biases as it is written: the scalar
top-level fields (`weight_version`, `learner_global_step`, wall times, the `num_*` counts)
plus every numeric `metrics.*` and `group_stats.*` entry, flattened to `metrics/<key>` and
`group_stats/<key>`. `wandb` is never a project dependency — it is imported lazily inside
`WandbSink.__init__`; if the package is not installed, or `wandb.init` raises (no network,
no login), the sink silently disables itself (`.enabled = False`) rather than failing the
run, so `logging.wandb: true` is safe to leave on in an environment without wandb access.

## `rollouts.jsonl` row keys (`logging/records.py::rollout_row`)

`step, group_id, task_id, task_idx, task_name, trajectory_id, rollout_index, reward,
reward_components, metrics, has_error, error_type, error_message, is_truncated,
stop_condition, is_completed, num_turns, num_input_tokens, num_output_tokens,
num_model_calls, num_sequences, num_action_tokens, policy_version, fidelity_problems,
thinking_leak_detected, timing, has_complete_logprobs, is_trainable`

`thinking_leak_detected` is true when a thinking-channel opener reappears after the first
closer, or when the sampled completion contains at least two closers. In that malformed
case the full parsed answer is moved to `reasoning_content` before scoring because the
first boundary is no longer trustworthy; sampled token IDs, logprobs, and training masks
remain unchanged. Marker-free prose that merely sounds like reasoning is not classified.

## `model_calls.jsonl` row keys (`model_call_row`)

`step, group_id, trajectory_id, call_index, num_prompt_tokens, num_completion_tokens,
finish_reason, policy_version, duration_s, logprob_mean, logprob_min, has_logprobs,
request_id`, plus `prompt_ids, completion_ids, sampling_logprobs` when
`logging.save_token_ids` (default `true`).

## `steps.jsonl` row keys (`step_row`)

`step, learner_global_step, weight_version, wall_time_s, rollout_wall_time_s,
train_wall_time_s, num_groups_received, num_groups_used, num_trajectories_used,
num_sequences_used, num_action_tokens, metrics, group_stats, trajectory_ids_used, extra`

On the fully asynchronous path, an additional `async` key carries
`behavior_versions, staleness_min/max/mean, queue_wait_s_mean, queue_size_at_dequeue,
learner_idle_s, published_version, adapter_export_s` (see "Fully asynchronous training (2
GPUs)" in [configuration.md](configuration.md)).

`extra` is the backend's `StepResult.extra` made JSON-safe: drop counts
(`num_dropped_untrainable`, `num_dropped_small_group`, `num_zero_std_groups`, ...), the raw
TRL metrics (`trl_metrics`), the LR-schedule horizon, and any skip reason.

`step` is the orchestrator's batch position — the same number stamped on this step's
`rollouts.jsonl`/`model_calls.jsonl` rows. `learner_global_step` is the trainer's own
optimizer-step counter (`StepResult.global_step`). The two are normally equal but diverge
when a batch had nothing trainable: the orchestrator still advances `step` (the batch was
consumed), while the backend skips the optimizer step, so `learner_global_step` does not
advance.

`metrics` is `StepResult.metrics` (canonical keys, all optional — a backend leaves what it
cannot measure absent, never a fake `0.0`): `loss, policy_loss, grad_norm, entropy, kl,
learning_rate, importance_ratio_mean, importance_ratio_p95, importance_ratio_p99,
clipped_token_fraction, sampling_vs_learner_logprob_diff_mean,
sampling_vs_learner_logprob_diff_abs_max, num_tokens, num_action_tokens`. `policy_loss` is
declared here but not currently emitted by any backend. `unsloth-native` additionally puts
every averaged TRL `_metrics["train"]` key verbatim into `extra["trl_metrics"]`.

`group_stats` is `logging/metrics.py::group_stats(groups)` — never invents a value; a
quantity with no supporting data is absent, not `0.0` (the exception is counts):
`num_groups, num_trajectories, reward_mean, reward_std, reward_min, reward_max,
zero_std_group_fraction, reward_component_mean.<name>, error_rate, truncation_rate,
completed_rate, error_type_counts.<type>, judge_error_rate, judge_calls, turns_mean,
turns_max, input_tokens_mean, output_tokens_mean, output_tokens_p95, action_tokens_total,
model_calls_total, calls_per_trajectory_mean, sampling_logprob_mean,
missing_logprob_fraction, fidelity_problem_fraction, policy_version_min,
policy_version_max, time_setup_s_mean, time_generation_s_mean, time_finalize_s_mean,
time_scoring_s_mean`.

`RunLogger.log_step` merges two more helpers into the same `group_stats` dict before
writing the `steps.jsonl` row (previously these were only computed for `eval.jsonl`):
`throughput()` adds `rollout_tokens_per_s, trajectories_per_s` (rollout-phase, over the
step's groups) and `train_throughput()` adds `train_trajectories_per_s`, plus
`train_tokens_per_s` when `StepResult.num_action_tokens` is nonzero (train-phase, over the
trajectories the learner actually used), and `train_loss_tokens_per_s` when the learner reports
loss-contributing action tokens — every one of these is absent, not `0.0`, when the
relevant wall time isn't known.

`time/optimizer_s` covers only `optimizer.step()`, `lr_scheduler.step()`, and gradient clearing.
