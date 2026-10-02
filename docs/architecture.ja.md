# アーキテクチャ

[English](architecture.md)

```text
YAML / CLI
    ↓
Orchestrator (orchestrator.py | async_rl/orchestrator.py)
    ├── VerifiersRuntime      rollout/runtime.py   (Environment, serving(), episode(task, ctx, n))
    │     └── PolicyClient    rollout/client.py    (vf.Client: renderer -> PolicyBackend -> Response)
    ├── PolicyBackend         policy/base.py       (token ids in, token ids + logprobs out)
    ├── RolloutGroup          rollout/schema.py    (canonical; no torch/verifiers imports)
    ├── RunLogger             logging/             (run directory, jsonl, summary, lineage)
    └── TrainerBackend        trainer/base.py
          └── unsloth-native  trainer/unsloth_native/   TRL GRPOTrainer (+Unsloth patches) shim
```

orchestrator が持つのは順序制御・ロギング・チェックポイントだけである。ロールアウトは
verifiers が、モデル・損失・optimizer は trainer backend が、生成は policy backend が所有する。
`verifiers`/`renderers` に触れるのは `VerifiersRuntime` だけであり、trainer backend からは
一切 import されない。`rollout/schema.py` も torch・verifiers・unsloth のいずれも import
しない ── プロジェクトの両半分をつなぐ縫い目である。backend の呼び出し(`setup`、
`train_step`、`save_checkpoint`、`load_checkpoint`)は `asyncio.to_thread` を通すため、自前の
イベントループを持つ backend も orchestrator と衝突しない。

## 正規ロールアウトスキーマ

`src/moongazer_rl/rollout/schema.py` で定義。

| 型 | 内容 |
|---|---|
| `ModelCall` | ロールアウト内の 1 回の policy 呼び出しを、サンプリング時のまま保持する: `prompt_ids`、`completion_ids`、`sampling_logprobs`(`completion_ids` と 1:1 で整列)、`finish_reason`、`policy_version`、タイミング。 |
| `TrainingSequence` | 1 本の損失のないトークン経路(verifiers の `Branch`): `token_ids` とトークン単位の `action_mask`(ツール結果・ユーザーシミュレータのターン・テンプレートの scaffold は `False`、サンプルされたトークンは `True`)、`token_ids` に整列した `sampling_logprobs`。すべてのターンのプロンプトが直前ターンの厳密なトークン prefix なら trajectory は 1 本の sequence を持つ。そうでなければ verifiers がグラフをフォークし、複数本になる。 |
| `Trajectory` | verifiers の 1 ロールアウト(1 つの `Trace`): reward、`reward_components`(そのまま、再合算しない)、metrics、info、errors、ターン・トークン数、タイミング、`model_calls[]` と `sequences[]` の両方。`is_trainable` は logprobs が揃っており reward が有限であること。採点自体が例外で終わった trace は `reward=NaN`、`metadata.scoring_failed=true` になり、ログには残るが 0.0 として学習には使わない。 |
| `RolloutGroup` | **`1 Task x G rollouts = 1 RolloutGroup`** ── GRPO の単位。`trajectories`(インデックス `0..G-1`)、`task_id`(`"<taskset id>:<task idx>"`)、`policy_version`(episode 開始時の weight version)、`reward_mean` / `reward_std`(母集団標準偏差。TRL の `scale_rewards="group"` と一致)。 |

トークン忠実性の規則: いかなる場合も再トークン化しない。`ModelCall` は policy backend が
サンプリング時に返したトークン ID と logprobs をそのまま保持し、`TrainingSequence` は
verifiers グラフ自身のトークン経路を保持する。`rollout/convert.py::verify_fidelity` が
trajectory ごとにこの 2 つを相互検証し、不一致は `trajectory.metadata["fidelity_problems"]`
に記録して `rollouts.jsonl` の監査情報とする。row 構築前に
`rollout/convert.py::verify_training_fidelity` が `trainer.call_unit` で選択した view を gate
する。call mode では空でない全 `ModelCall` の logprobs が有限かつ token と整列していること、
sequence mode では sequence の各配列が整列し、action 位置の logprobs が有限であり、全 action
span を覆う call と token/logprob が順序どおり一致することを要求する。両 mode とも trajectory
内の policy version が 1 つで group と一致することを要求する。違反した trajectory は group
baseline の計算前に除外し、残った trajectory が estimator の最小 group size 未満になった場合
だけ group 全体を除外する。除外数と ID は `StepResult.extra` に記録し、step ごとに 1 回警告する。
