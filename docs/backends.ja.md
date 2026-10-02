# Trainer backend

`trainer.backend` の各値がモデルをどう駆動するか。`unsloth-native` が本番経路である。

## `unsloth-native`

TRL の `GRPOTrainer`(`unsloth` パッケージが入っていれば Unsloth の `PatchFastRL` がその場で
再生成・パッチする)を、正規の `RolloutGroup` で 1 optimizer step ずつ駆動する。要約:

* **注入点**: `_generate_single_turn(prompts, images)` を上書きし、既にサンプル済みの
  `(prompt_ids, completion_ids, logprobs, {})` を返す。TRL の呼び出し連鎖の中で最も低い
  位置であるため、`_generate` の長さメトリクス、パディング、`old_per_token_logps` /
  `ref_per_token_logps`、reward、advantage はすべて無改造のまま動き続ける。Unsloth は
  このメソッドの名前とシグネチャを保存するため、同じサブクラスが素の TRL と Unsloth が
  再生成した trainer の両方に対して動作する。
* **action mask**: `completion_mask` に畳み込むのではなく、別キー `tool_mask`(まさに
  この概念に対する Unsloth 自身の呼び名)として運ぶ。`completion_mask` は attention mask
  も駆動するためであり、マルチターンのツール/ユーザーシミュレータのトークンは attend
  されなければならないが loss を受け取ってはならない。素の TRL パス(CPU、Unsloth 無し)
  では小さな `_compute_loss` の上書きが `tool_mask` を loss に適用しつつモデルの
  attention mask は完全なまま保つ。Unsloth パスではこの上書きは死んだコードになる ──
  Unsloth が `compute_loss` を丸ごと置き換え、`tool_mask` を自ら読むためである。
* **sampling logprobs → importance sampling**: `self.use_vllm` と
  `self.vllm_importance_sampling_correction` を `__init__` の**後**に `True` に設定する
  (vLLM エンジンは構築されない)。これは純粋に TRL 既存の `old_per_token_logps` /
  `importance_sampling_ratio` の経路を有効化するためであり、サンプル済みの completion は
  policy のスナップショットから来ており、learner の現在の forward pass からではないから
  である。
* **`train_step` ごとに厳密に1 optimizer step**: `trainer.training_step(...)` を呼び
  (これは TRL/Unsloth の実際の `_prepare_inputs` -> `compute_loss` ->
  `accelerator.backward` を走らせる)、その後 `Trainer._inner_training_loop` が通常行う
  同期 step のブロック(`_clip_grad_norm`、`optimizer.step()`、`lr_scheduler.step()`、
  `zero_grad`、`global_step += 1`)を約 8 行で再現する ── `trainer.train()` 自身の
  dataloader/epoch ループには一度も入らない。`TrainerCallback` と `trainer.log()` は
  発火せず、メトリクス/チェックポイントは run logger が代わりに担う。
* **チェックポイント形式**: `checkpoints/step-<n>/model/` ── PEFT アダプタ
  (`trainer.save_model()`)。`PeftModel.from_pretrained` で直接読める。`optimizer.pt`
  (optimizer + LR スケジューラの状態)。`trainer_state.json`(`global_step`、
  `weight_version`、トークナイザ指紋)。orchestrator がこれらと並べてもう1つ、
  どの backend でも共通のファイルを書く: `orchestrator_state.json`(`step`、
  `learner_global_step`、`weight_version`)── [run-directory.ja.md](run-directory.ja.md) の「Resume の方針」を参照。
* **制御対象の `trainer.args` キー**: [設定と CLI](configuration.ja.md) の「`trainer.args` の制御対象キー」を参照。
* **行の単位**(`trainer/unsloth_native/dataset.py`): `"sequence"`(既定)── `TrainingSequence`/`Branch`
  1本につき1行。安価で厳密な既定値。`"call"` ── `ModelCall` 1件につき1行。損失はないが
  共有 prefix を呼び出しごとに再計算する。フォークした trajectory(あるターンが再トークン化
  された場合)は `fork_policy` で処理する: `"error"`(既定、最も安全)、`"all"`(reward を
  ブランチ全体にブロードキャスト)、`"longest"`。現時点では `RunConfig` のフィールドとして
  公開されていない(`trainer.call_unit`/`trainer.fork_policy`)。backend を直接構築しない限り
  常に `"sequence"`/`"error"` になる。
