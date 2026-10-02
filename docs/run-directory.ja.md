# Run ディレクトリ、ログ、メトリクス

`mg-rl train` が書き出すファイル、jsonl ログの行キー、resume の方針、任意の Weights & Biases sink をまとめる。

```text
runs/<run-name>/
  config.yaml            # 与えられたまま
  config.resolved.yaml   # デフォルト + CLI 上書き適用後
  runtime.json           # バージョン、プラットフォーム、git、判明していればトークナイザ/テンプレートの指紋
  rollouts.jsonl         # trajectory 1件につき1行
  traces.jsonl           # trace.to_record()(生の verifiers trace: nodes[].token_ids/mask, rewards, ...)
  model_calls.jsonl      # ModelCall 1件につき1行
  steps.jsonl            # 学習 step 1回につき1行
  events.jsonl           # ライフサイクル、エラー、resume、チェックポイント
  summary.json           # 集計(各 step の後に原子的に書き直される)
  lineage.json           # ランの出自 / resume 履歴
  checkpoints/step-<n>/   # backend 自身のチェックポイントファイル + orchestrator_state.json(step, learner_global_step, weight_version)
  eval.jsonl, eval_rollouts.jsonl, eval_traces.jsonl   # インライン eval(config.eval.every_steps)。rollouts.jsonl/traces.jsonl には混ぜない
```

完全非同期経路(`async_rl/orchestrator.py`)も同じファイルを書き出す。加えて
`steps.jsonl` の各行に `async` ブロックが増え、`orchestrator_state.json` にも
`published_version`、`run_gen`、`task_cursor`、`uncommitted_task_positions` が増える ──
[設定と CLI](configuration.ja.md) の「完全非同期学習(2 GPU)」を参照。

`RunLogger.open_for_resume` は各 jsonl ファイルの末尾 1 行だけが壊れている場合(append の
中断)を復旧する: 壊れた末尾は同じディレクトリの `<name>.partial.<UTC>.json` へ退避され、
`resume_repair` イベントが記録される。末尾より前のどこかの行が壊れている場合は、実データを
黙って切り詰めるのではなく例外を送出する。

## Resume の方針

Resume の再開位置は**最新チェックポイントの**バッチ位置であり、ログの最終行では決してない。
`orchestrator._restore_for_resume` は `checkpoints/step-<n>/orchestrator_state.json` の
`step` を読んで再開位置とする(このファイルが存在しない古いチェックポイントに対しては
`step-<n>` というディレクトリ名から数値を読み取る)── `steps.jsonl` の最終行からは
一度も読まない。`steps.jsonl` はチェックポイントより先へ進んでいることがある
(`checkpoint.save_steps` の間隔の間に学習だけ済んで保存されなかった step)。その場合、
それらのバッチは resume 時に再度学習される(重み更新が永続化されていないため)。これが
起きたとき `events.jsonl` に種類 `resume_rewind` の行が `logged_last_step`/`checkpoint_step`
とともに記録される。これはクラッシュ後にだけ起こる。正常終了したランは
(`checkpoint.save_steps` が設定されていれば)最終 step が間隔の倍数でなくても必ず保存する。
チェックポイントは `step-<n>.tmp` に書いてから rename するので、書きかけのディレクトリが
最新として拾われることはない。ログに step はあるがチェックポイントが**一つもない**ランは resume を
拒否する ── `train(resume=True)`/`mg-rl resume` はログを正としてこっそり信用するのでは
なく `FileNotFoundError` を送出する(学習済みの重みは失われている。新しいランを開始し、
`checkpoint.save_steps` を設定すること)。

## Weights & Biases(任意)

`logging.wandb: true` にすると `WandbSink`(`logging/wandb_sink.py`)が構築され、
書き込まれるたびに `steps.jsonl` の各行を平坦化して Weights & Biases へ送る: トップレベルの
スカラー項目(`weight_version`、`learner_global_step`、各種 wall time、`num_*` の件数)に加え、
`metrics.*`・`group_stats.*` の数値項目すべてを `metrics/<key>`・`group_stats/<key>` として
展開する。`wandb` は本プロジェクトの依存関係には一切加えていない ──
`WandbSink.__init__` の中で遅延 import され、パッケージが無い場合や `wandb.init` が
例外を出した場合(ネットワーク無し、未ログインなど)は、ランを失敗させるのではなく
サイレントに無効化される(`.enabled = False`)。そのため `logging.wandb: true` は wandb に
アクセスできない環境でも安全に付けたままにできる。

## `rollouts.jsonl` の行キー(`logging/records.py::rollout_row`)

`step, group_id, task_id, task_idx, task_name, trajectory_id, rollout_index, reward,
reward_components, metrics, has_error, error_type, error_message, is_truncated,
stop_condition, is_completed, num_turns, num_input_tokens, num_output_tokens,
num_model_calls, num_sequences, num_action_tokens, policy_version, fidelity_problems,
thinking_leak_detected, timing, has_complete_logprobs, is_trainable`

`thinking_leak_detected` は、最初の thinking 終端より後に開始マーカーが再出現した場合、
または sampled completion に終端マーカーが2個以上ある場合に true となる。この構造異常では
最初の境界を信用できないため、採点前に回答全文を `reasoning_content` へ移す。sampled token
IDs、logprobs、学習 mask は変更しない。マーカーを伴わず思考調に見える文章は判定しない。

## `model_calls.jsonl` の行キー(`model_call_row`)

`step, group_id, trajectory_id, call_index, num_prompt_tokens, num_completion_tokens,
finish_reason, policy_version, duration_s, logprob_mean, logprob_min, has_logprobs,
request_id`、加えて `logging.save_token_ids`(既定 `true`)のとき
`prompt_ids, completion_ids, sampling_logprobs`。

## `steps.jsonl` の行キー(`step_row`)

`step, learner_global_step, weight_version, wall_time_s, rollout_wall_time_s,
train_wall_time_s, num_groups_received, num_groups_used, num_trajectories_used,
num_sequences_used, num_action_tokens, metrics, group_stats, trajectory_ids_used, extra`

完全非同期経路ではさらに `async` キーが増え、`behavior_versions, staleness_min/max/mean,
queue_wait_s_mean, queue_size_at_dequeue, learner_idle_s, published_version,
adapter_export_s` を持つ([設定と CLI](configuration.ja.md) の「完全非同期学習(2 GPU)」を参照)。

`extra` は backend の `StepResult.extra` を JSON 化したもの。除外件数
(`num_dropped_untrainable`、`num_dropped_small_group`、`num_zero_std_groups` など)、
TRL の生メトリクス(`trl_metrics`)、LR スケジュールの horizon、スキップ理由が入る。

`step` は orchestrator のバッチ位置であり、同じ step の `rollouts.jsonl`/`model_calls.jsonl`
の行に刻まれているのと同じ番号である。`learner_global_step` は trainer 自身の optimizer
step カウンタ(`StepResult.global_step`)である。両者は通常一致するが、バッチに学習可能な
ものが何もなかったとき乖離する: orchestrator はそれでも `step` を進める(バッチは消費
された)が、backend は optimizer step を省略するので `learner_global_step` は進まない。

`metrics` は `StepResult.metrics`(すべて任意の正規キー ── backend は測れないものを
偽の `0.0` にせず単に省く): `loss, policy_loss, grad_norm, entropy, kl,
learning_rate, importance_ratio_mean, importance_ratio_p95, importance_ratio_p99,
clipped_token_fraction, sampling_vs_learner_logprob_diff_mean,
sampling_vs_learner_logprob_diff_abs_max, num_tokens, num_action_tokens`。`policy_loss` は
ここで宣言されているが、現時点でどの backend も出力していない。`unsloth-native` はさらに
TRL の `_metrics["train"]` の平均済みキーをすべてそのまま `extra["trl_metrics"]` に入れる。

`group_stats` は `logging/metrics.py::group_stats(groups)` ── 値を捏造しない。裏付けとなる
データがない量は `0.0` ではなく省かれる(件数だけは例外): `num_groups, num_trajectories,
reward_mean, reward_std, reward_min, reward_max, zero_std_group_fraction,
reward_component_mean.<name>, error_rate, truncation_rate, completed_rate,
error_type_counts.<type>, judge_error_rate, judge_calls, turns_mean, turns_max,
input_tokens_mean, output_tokens_mean, output_tokens_p95, action_tokens_total,
model_calls_total, calls_per_trajectory_mean, sampling_logprob_mean,
missing_logprob_fraction, fidelity_problem_fraction, policy_version_min,
policy_version_max, time_setup_s_mean, time_generation_s_mean, time_finalize_s_mean,
time_scoring_s_mean`。

`RunLogger.log_step` は `steps.jsonl` の行を書く前に、さらに2つのヘルパをこの
`group_stats` 辞書へ統合する(以前は `eval.jsonl` のためだけに計算していたもの):
`throughput()` は `rollout_tokens_per_s, trajectories_per_s`(rollout フェーズ。その step の
groups について)を、`train_throughput()` は `train_trajectories_per_s`、および
`StepResult.num_action_tokens` が非ゼロのときは `train_tokens_per_s`(train フェーズ。
learner が実際に使った trajectory について)、loss に寄与した action token が報告された
ときは `train_loss_tokens_per_s` を追加する ── 該当する wall time が
分からないときはどちらも `0.0` ではなく省かれる。

`time/optimizer_s` は `optimizer.step()`、`lr_scheduler.step()`、勾配消去だけを含む。
