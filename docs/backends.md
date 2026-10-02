# Trainer backends

How each `trainer.backend` value drives the model. `unsloth-native` is the production path.

## `unsloth-native`

TRL `GRPOTrainer` (regenerated/patched in place by Unsloth's `PatchFastRL` when the
`unsloth` package is installed) driven one optimizer step at a time by canonical
`RolloutGroup`s. Summary:

* **Injection point**: `_generate_single_turn(prompts, images)` is overridden to return the
  already-sampled `(prompt_ids, completion_ids, logprobs, {})`. This is the lowest point in
  TRL's call chain, so `_generate`'s length metrics, padding, `old_per_token_logps` /
  `ref_per_token_logps`, rewards and advantages all keep working unmodified. The same
  subclass works against plain TRL and against Unsloth's regenerated trainer because
  Unsloth preserves this method's name and signature.
* **Action mask**: carried as a separate `tool_mask` key (Unsloth's own name for exactly
  this concept) rather than folded into `completion_mask`, because `completion_mask` also
  drives the attention mask — multi-turn tool/user-simulator tokens must be attended to but
  must not receive loss. On the plain-TRL path (CPU, no Unsloth) a small `_compute_loss`
  override applies `tool_mask` to the loss while keeping the model's attention mask
  complete; on the Unsloth path this override is dead code, since Unsloth replaces
  `compute_loss` wholesale and reads `tool_mask` itself.
* **Sampling logprobs -> importance sampling**: `self.use_vllm` and
  `self.vllm_importance_sampling_correction` are set to `True` *after* `__init__` (so no
  vLLM engine is built) purely to turn on TRL's existing `old_per_token_logps` /
  `importance_sampling_ratio` code paths, since the sampled completions come from a policy
  snapshot, not the learner's current forward pass.
* **One optimizer step per `train_step`**: calls `trainer.training_step(...)` (which still
  runs TRL's/Unsloth's real `_prepare_inputs` -> `compute_loss` -> `accelerator.backward`)
  and then replays the ~8-line sync-step block (`_clip_grad_norm`, `optimizer.step()`,
  `lr_scheduler.step()`, `zero_grad`, `global_step += 1`) that
  `Trainer._inner_training_loop` normally runs — `trainer.train()`'s own dataloader/epoch
  loop is never entered. `TrainerCallback`s and `trainer.log()` never fire; the run logger
  owns metrics/checkpointing instead.
* **Checkpoint format**: `checkpoints/step-<n>/model/` — a PEFT adapter (`trainer.save_model()`),
  directly loadable by `PeftModel.from_pretrained`; `optimizer.pt` (optimizer + LR-scheduler
  state); `trainer_state.json` (`global_step`, `weight_version`, tokenizer fingerprint). The
  orchestrator writes one more file next to these, common to every backend:
  `orchestrator_state.json` (`step`, `learner_global_step`, `weight_version`) — see the resume
  policy in [run-directory.md](run-directory.md).
* **Controlled `trainer.args` keys**: see "Controlled `trainer.args` keys" in [configuration.md](configuration.md).
* **Row units** (`trainer/unsloth_native/dataset.py`): `"sequence"` (default) — one row per
  `TrainingSequence`/`Branch`, the cheap and exact default; `"call"` — one row per
  `ModelCall`, lossless but recomputes the shared prefix per call. A forked trajectory (a
  turn re-tokenised) is handled per `fork_policy`: `"error"` (default, safest), `"all"`
  (broadcast reward across branches), `"longest"`. Not yet exposed as `RunConfig` fields
  (`trainer.call_unit`/`trainer.fork_policy`); currently always `"sequence"`/`"error"`
  unless a caller constructs the backend directly.
* **Sequence overflow**: `trainer.sequence_overflow` is `"error"` by default; `"drop"`
  removes the containing trajectory before group advantages are calculated.
