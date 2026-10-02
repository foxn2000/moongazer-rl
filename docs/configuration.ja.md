# 設定と CLI

[English](configuration.md)

## 固定している上流バージョン

| package | version | 備考 |
|---|---|---|
| verifiers | 0.2.0 | ネイティブ v1 API |
| renderers | 0.1.11 | verifiers の依存。モデルファミリごとの chat-template レンダリングと completion のパース |
| trl | 0.24.0 | `unsloth 2026.9.2` が `trl>=0.18.2,<=0.24.0` を要求 |
| transformers | 5.5.0 | unsloth が `<=5.5.0` を要求 |
| torch | 2.11.0 | unsloth が `<2.12` を要求 |
| datasets | 4.3.0 | verifiers が `<4.7`、unsloth が `<4.4` を要求 |
| peft | 0.20.0 | |
| mcp | 1.29.1 | `<2` に固定 ── verifiers 0.2.0 が使う `mcp.server.fastmcp` は mcp 2.x で削除された |
| unsloth / unsloth_zoo | 2026.9.2 / 2026.9.1 | 任意の `unsloth` extra、Linux 専用 |

## 設定

`RunConfig`(`src/moongazer_rl/config/schema.py`)は薄いランチャー設定である。自前の
セクションは小さく `extra="forbid"`。上流ライブラリに属するものは不透明な passthrough の
辞書であり、`mg-rl validate` がインストール済みの上流パッケージに対して遅延検証する。

| 自前のキー | passthrough 先 |
|---|---|
| `environment` | `verifiers.v1.EnvConfig`(構造的に必須なのは `taskset.id` / `harness.id` のみ。`split` などの環境固有な taskset フィールドはそのまま通す) |
| `rollout.sampling` | `verifiers.v1.Sampling` |
| `trainer.args` | 解決された `TrainerBackend` 自身の引数(`unsloth-native` なら TRL `GRPOConfig`) |
| `model.renderer` | `renderers` パッケージのモデルごとの設定 |
| `policy.args` | `policy.backend: vllm-server` のときの `policy.vllm_server.VllmServerPolicy` のキーワード引数 |

Gemma 4 E2B の text-only 学習では、`model.text_only_offload: true` により per-layer
embedding 4.37 GiB と vision/audio tower 0.87 GiB を CPU に置く(RTX 5090、bf16、2026-09-19)。

完全な YAML の形:

```yaml
run:
  name: <str>                 # runs/<name> ディレクトリの既定名でもある
  seed: 42

model:
  name: <str>                 # HF hub id またはローカルパス
  load_in_4bit: false
  text_only_offload: true     # Gemma 4 E-series の PLE と vision/audio module を CPU に置く
  max_seq_length: <int | null>
  gradient_checkpointing: unsloth  # unsloth (CPU offload)、torch (再計算のみ)、off、auto
  gradient_checkpointing_auto_max_tokens: 4096  # auto はこの padded size 以下で GC を無効化
  lora:                       # 任意
    r: <int>
    alpha: <float>
    dropout: 0.0
    target_modules: <list[str] | null>
  renderer: {}                 # renderers.RendererConfig への不透明な passthrough

environment:
  taskset:
    id: <str>                  # verifiers taskset プラグインの id
    args: {}                   # taskset の辞書へフラットにマージされる(split など環境固有のフィールド)
  harness:
    id: default
    runtime: {}                 # 例: {type: subprocess} / {type: docker}
  max_turns: null
  max_input_tokens: null
  max_output_tokens: null
  max_total_tokens: null
  timeout: {}
  retries: {}
  multiplex: 32

rollout:
  num_generations: <int>       # 1 タスクあたりの G 本 = GRPO のグループサイズ
  groups_per_step: <int>       # 1 optimizer step あたりのタスク数
  max_concurrent: 32
  num_epochs: 1                # null は「max_steps まで」
  max_steps: null
  sampling: {}                  # verifiers.v1.Sampling への不透明な passthrough(temperature, max_tokens, ...)

trainer:
  backend: unsloth-native
  args: {}                       # backend 自身の引数への不透明な passthrough
  sequence_overflow: error       # model.max_seq_length 超過 row の trajectory を error または drop
  memory:                        # unsloth-native learner のメモリ設定。「learner のメモリ設定」参照
    blocked_global_attention: true
    materialize_sliding_mask: false
    query_chunk_size: 1024
    min_global_q_len: 1024
    token_chunked_logprobs: true
    logprob_chunk_size: 64
    score_rows_one_at_a_time: true

policy:
  backend: auto                  # auto = trainer backend の in-process sampler。vllm-server = 外部の `vllm serve`
  args: {}                       # vllm-server: base_url(必須)、model、logprobs_mode、timeout_s、admin_timeout_s、keep_adapters など

logging:
  output_dir: null              # 既定は runs/<run.name>
  wandb: false                   # true にすると steps.jsonl の平坦化行も Weights & Biases へ送る(wandb 未インストールなら no-op)
  save_token_ids: true

checkpoint:
  save_steps: 10
  keep_last: 3

eval:
  every_steps: 10               # 非同期経路では無視される(評価は単独の `mg-rl eval` で行う)
  num_tasks: null
  taskset: null                 # 宣言のみで未配線。eval は environment.taskset を使い先頭 num_tasks 件を切り出す
  checkpoint: latest            # 単独 `mg-rl eval` が復元するチェックポイント("latest"、パス、null なら初期化直後のモデル)

async:                          # 任意。同期経路のみなら省略する。「完全非同期学習(2 GPU)」参照
  max_staleness: 1
  queue_capacity: 4
  num_producers: 2
  drop_zero_std_groups: true
  max_consecutive_zero_std_drops: 16
  zero_std_budget_action: warn     # または stop
  max_no_progress_s: null
  publish_retries: 3
  publish_retry_backoff_s: 2.0
  max_consecutive_rollout_failures: 5
  shutdown_timeout_s: 600.0
  unload_grace_s: 2.0
```

`--set dotted.key=value`(繰り返し指定可)は検証前に任意のキーを上書きする。値は YAML として
パースされる(`4` -> int、`1e-5` -> float、`[1, 2]` -> list)。

## 外部 vLLM サーバーからサンプルする(`policy.backend: vllm-server`)

`unsloth-native` の既定(`policy.backend: auto`)は in-process サンプリングである。learner と同じ重みで HF
`model.generate` を使う(逐次生成。smoke test には十分だが学習には遅い)。unsloth 2026.9.x の
`torch<2.12, transformers<=5.5` の範囲で解決でき Gemma 4 を扱える vllm は無いため、本番の
sampler は別プロセスの `vllm serve` とし、HTTP で会話する。

```yaml
policy:
  backend: vllm-server
  args:
    base_url: http://127.0.0.1:8000
    logprobs_mode: processed_logprobs   # サーバーの --logprobs-mode と一致させる
    timeout_s: 1800
```

```bash
GPU_UTIL=0.5 examples/rtx5090-math-env/serve-vllm.sh   # --enable-lora、実行時 LoRA 更新、processed_logprobs
```

契約(`policy/vllm_server.py`): renderer の prompt token ID を
`POST /v1/completions` の `prompt: list[int]` にそのまま渡し、`return_token_ids` でサンプル
された ID をそのまま受け取り、`logprobs: 0` でサンプル token の logprob を受け取る。optimizer
step ごとに learner が PEFT adapter を `<output_dir>/.lora-sync/v<version>/` に書き出し、policy
が `/v1/load_lora_adapter` を呼ぶ。以後のリクエストは `policy-v<version>` をモデル名にする。
直前の adapter は in-flight リクエストのために残す(`keep_adapters`、既定 2)。サーバーと
trainer は同じファイルシステムを見ている必要がある。

## 完全非同期学習(2 GPU)

`async:` セクションがあると `mg-rl train` は
`async_rl/orchestrator.py` を使う。producer コルーチンが episode(`1 task × G rollouts` 単位)
を生成し続けて bounded queue に入れ、learner loop が取り出して学習する。publisher task が
learner の LoRA adapter を export・hot-load し、どちらの側もブロックしない。

```text
GPU 0: vllm serve --enable-lora        GPU 1: mg-rl train(async: セクション付き)
  producer x N --lease--> generate       learner: train_step -> export -> publish(非ブロック)
```

| キー | 既定値 | 意味 |
|---|---|---|
| `max_staleness` | 1 | 許容する `learner_version - behavior_version` の最大差。0 で rollout/train の重なりが無くなる(barrier 相当) |
| `queue_capacity` | 4 | bounded queue のサイズ(`RolloutGroup` 単位)。超えると `put()` がブロックする(backpressure) |
| `num_producers` | 2 | 並行 episode 数 |
| `drop_zero_std_groups` | true | DAPO 型 dynamic sampling。学習可能 trajectory が 2 本未満、または reward がすべて等しい group を代替する |
| `max_consecutive_zero_std_drops` | 16 | 連続してこの数(以降はその倍数ごと)を捨てたら `zero_std_budget_exhausted` を記録する。zero-advantage の group を強制学習することは無い |
| `zero_std_budget_action` | warn | `warn` は drop を続け、`stop` は予算到達で run を失敗させる |
| `max_no_progress_s` | null | optimizer step の無い状態がこの秒数(最初の rollout 完了から計測)続いたら run を止める |
| `publish_retries` | 3 | adapter publish が load に失敗したときの再試行回数 |
| `publish_retry_backoff_s` | 2.0 | publish 再試行間のバックオフ |
| `max_consecutive_rollout_failures` | 5 | producer が連続でこの回数失敗したらランを止める |
| `shutdown_timeout_s` | 600.0 | graceful shutdown の締切 |
| `unload_grace_s` | 2.0 | 退役した adapter の refcount が 0 になってからも残す秒数(進行中の HTTP request の切断を待つ猶予) |

意味論:

* **staleness** は `learner_version_at_consume - behavior_version`。producer は episode 開始前に
  admission slot を確保し、`learner_version + reserved_slots // groups_per_step -
  published_version <= max_staleness` を満たす。dequeue 時に実際の staleness が
  `max_staleness` を超えている group は gate で捨てる(`stale_group_dropped`)。上限を保証する
  のは admission の見積もりではなくこの gate である。
* **lease** は episode 単位で 1 回だけ取得し、episode が終わるまで保持する。episode 内の
  すべての rollout・すべての turn は同じ adapter version で生成され、1 trajectory が 2 つの
  policy version にまたがることは無い。
* **dynamic sampling**: `drop_zero_std_groups: true` のとき、学習可能 trajectory が 2 本未満、
  または reward がすべて等しい group は commit して捨て、無勾配の step を進めない。advantage が
  group 内で決まる設定(`group_zscore` / `group_center` / `rloo`、`std_scope: group`、`beta: 0`)
  では producer が rollout 完了直後に捨てて admission slot を即返す
  (`zero_std_group_dropped.stage: producer`)。それ以外は learner が dequeue 時に判定する。
  advantage が全て厳密に 0 の batch(`beta: 0`)は forward の前に拒否する(`train_skipped`):
  optimizer step も `weight_version` も publish も進まない。`max_consecutive_zero_std_drops` は
  警告のみ(`zero_std_budget_action: stop` で停止)、`max_no_progress_s` は optimizer step の無い
  wall-clock を制限する。
* **fidelity gate**: `build_batch` は各 trajectory の loss 入力 view を検査し
  (`verify_training_fidelity`: token/logprob の対応、有限な logprob、全 sampled token が記録済み
  call に覆われること、group 内の policy version 一致)、違反を group の baseline 計算前に外す。
  `fidelity_violation` イベントと `num_dropped_fidelity` に残る。
* **off-policy 補正**は truncated importance sampling だけである。token 単位の重み
  `min(exp(learner_old - sampling), cap)`(既定 cap 2.0)。sampling 時に削られた support は
  補正できないため、非同期経路は `temperature > 0` を必須とし、`top_k`、1.0 以外の `top_p`、
  `min_p`、`logit_bias`、`bad_words`、`allowed_token_ids` を拒否する
  (`config/validate.py::_validate_async`)。`policy.backend` は `vllm-server` であること。
  `eval.every_steps` は無視される(run は
  `inline_eval_disabled` を記録する)。
* **checkpoint / resume**: チェックポイントは learner の step の合間に取る。
  `orchestrator_state.json` に `task_cursor` と `uncommitted_task_positions` を記録する。resume
  時は queue が空から始まり、producer は未決着の task 位置を先に再サンプルする。復元した
  adapter は新しい `<gen>` 名で publish し(残っていた `policy-*` adapter は unload)、確認後に
  producer を開始する。

起動: `examples/rtx5090-math-env/`(Prime Intellect の `math-env`、RTX 5090 1 枚に sampler と learner を同居)。

`events.jsonl` には `rollout_started`/`rollout_finished`、`group_enqueued`/`group_dequeued`、
`stale_group_dropped`、`train_started`/`train_finished`、`publish_*`、`adapter_unloaded`、
`producer_idle`/`learner_idle` が増え、`steps.jsonl` の行には `async` ブロック
(`behavior_versions`、`staleness_min/max/mean`、`queue_wait_s_mean`、`queue_size_at_dequeue`、
`learner_idle_s`、`published_version`、`adapter_export_s`)が増える。
`mg-rl async-report <run_dir> [--json]` はこれらのイベントから、rollout と train の
wall-clock 上の重なり、staleness の分布、publish latency、スループット、queue の占有を要約する。

## RL 目的関数・batch planner・複数 replica・version 付き評価

分解の詳細は [algorithms.md](algorithms.md)。全キーは省略可能で、既定値はこれまでの GRPO 経路
(`trainer.args.loss_type: dapo`)と同じ意味になる。

```yaml
trainer:
  algorithm:
    advantage_estimator: group_zscore   # group_zscore | group_center | rloo
    policy_objective: grpo_clip         # grpo_clip | cispo | reinforce
    ratio_reference: proximal           # proximal(更新直前の learner)| behavior(sampling logprob)
    behavior_correction: truncate       # none | truncate | mask (重み min(exp(learner - sampling), behavior_cap))
    loss_normalization: action_token    # sequence | action_token | fixed_length(step 全体の分母、ragged でも同一)
    skip_old_logprob_forward: auto      # 数学的に同値なら old-logprob の no-grad forward を省く
  batch:
    max_model_tokens: 32768             # token 予算の micro-batch。RolloutGroup は分割しない
    max_rows: 8
policy:
  args:
    base_urls: [http://127.0.0.1:8101, http://127.0.0.1:8102]   # 複数 vLLM replica(または base_url)
async:
  eval_every_steps: 6                   # published version を lease して評価。eval-v<N>/ に保存
```

legacy の `trainer.args`(`loss_type`、`scale_rewards`、`epsilon*`、`vllm_importance_sampling_cap`)は
同じ意味のまま同じ spec に写像され、明示した `trainer.algorithm` と矛盾すると validation error になる。
GPU では全ての目的関数が unsloth_zoo の fused loss を通り、変わるのは advantage の計算と loss の kwargs だけ。

## learner のメモリ設定(`trainer.memory`、`unsloth-native`)

すべてのスイッチは exact(loss と勾配は浮動小数の丸め誤差内で同じ)で、変わるのはピーク
メモリだけである。

| キー | 既定値 | 効果 |
|---|---|---|
| `blocked_global_attention` | true | Gemma 4 の global causal attention に exact な flex/blocked を使う。false で両方無効 |
| `global_attention` | flex | `flex`(失敗時は `blocked`)または `blocked`。提供実測: RTX 5090、bf16、S=16384 の forward+backward は 147.7 ms / +0.13 GB、blocked(1024) は 216.6 ms / +2.38 GB |
| `sliding_attention` | flex | `flex` (共有メモリ 128 KiB 未満では 32×32) または `unsloth`。提供実測 RTX 5090 B=1 bf16 D=256、S=4096/8192 の forward+backward は 0.9/1.7 ms、dense mask 付き memory-efficient SDPA は 15.1/51.4 ms。失敗時は Unsloth に戻る |
| `materialize_sliding_mask` | false | 未使用の dense sliding mask を省く(S=30,000 で 862 MiB、Gemma 4 E2B、RTX 5090、2026-09-19)。true で従来どおり生成 |
| `query_chunk_size` | 1024 | `blocked` の query ブロック幅|
| `min_global_q_len` | 1024 | これより短い系列は stock の SDPA のまま |
| `token_chunked_logprobs` | true | 最終 hidden state から token チャンク単位で logprob を計算する(`[S, vocab]` を一度に作らない)。backward ではチャンクごとに再計算 |
| `logprob_chunk_size` | 64 | 1 チャンクの token 数 |
| `score_rows_one_at_a_time` | true | old/ref logprob の採点を 1 行ずつ forward する。`token_chunked_logprobs` と同じ値にする |

eos と異なる専用 pad token が必要。

## optimizer(`unsloth-native`)

optimizer は `trainer.args` が TRL に求めたものになる。`optim` / `optim_args` / `learning_rate` /
`weight_decay` は `GRPOConfig` と `Trainer.create_optimizer()` にそのまま渡る。Unsloth 環境の
既定は `adamw_8bit`。素の SGD は `optim: sgd`、`optim_args: "momentum=0"`、`weight_decay: 0.0`
(LR の桁は 2〜3 桁違う)。各 run は `optimizer` イベント(名前、torch クラス、
ハイパーパラメータ、学習対象パラメータ数)を記録し、同じ内容を各 checkpoint の
`trainer_state.json` に保存する。各 step は
`learner_step_peak_memory_gb`、`learner_peak_memory_gb` を報告する。

## `trainer.args` の制御対象キー(`unsloth-native`)

`trainer/unsloth_native/compat/trl_0_24.py::CONTROLLED_ARGS`。`reject` されたキーは
`validate`/`setup` 時に例外になり、`warn` されたキーはそのまま使われるが警告が出る。

| キー | 方針 | 理由 |
|---|---|---|
| `num_generations` | reject | `rollout.num_generations` から設定される |
| `num_iterations` | reject | 1 でなければならない: `train_step` ごとに 1 optimizer step |
| `gradient_accumulation_steps` | reject | 1 でなければならない。累積は複数グループを送ることで行う |
| `steps_per_generation` | reject | 同じ理由で 1 でなければならない |
| `generation_batch_size` | reject | 内部で導出される |
| `use_vllm`、`vllm_mode` | reject | 生成は常に rollout 側の policy が行う |
| `use_liger_loss` | reject | `_compute_loss` を迂回するため action mask が無視される |
| `remove_unused_columns` | reject | シムは dataloader を完全に迂回する |
| `output_dir` | reject | `logging.output_dir` から設定される |
| `report_to` | reject | メトリクス報告は run logger が所有する |
| `seed` | reject | `run.seed` から設定される |
| `temperature` | match | `rollout.sampling.temperature` と一致していなければならない |
| `max_completion_length` | warn | 既定は `model.max_seq_length`。completion を切り詰めはせず `dr_grpo` の損失を正規化するだけ |
| `mask_truncated_completions` | warn | renderer 基準の `finish_reason == "length"` で再実装している |

未知のキーは「もしかして」の提案付きで reject される。それ以外はそのまま `trl.GRPOConfig`
へ渡され、インストールされている TRL/Unsloth が宣言するフィールドに対して検証される。
Gemma 4 E 系 × transformers 5.5.0 では `per_device_train_batch_size` は 1 でなければならない。

## CLI

`mg-rl {validate,train,eval,resume,inspect,async-report}`。`--help` と `validate` は
torch/verifiers プラグインが無い環境でも動く。

```bash
uv run mg-rl validate <config.yaml>
uv run mg-rl train <config.yaml> --set rollout.max_steps=2
uv run mg-rl eval <config.yaml>                    # 単体 eval。steps.jsonl には書かない
uv run mg-rl resume <run_dir> --set rollout.max_steps=4

uv run mg-rl inspect <run_dir>                     # summary.json + 直近 5 step
uv run mg-rl inspect <run_dir> --steps --json
uv run mg-rl inspect <run_dir> --rollouts --last 3
uv run mg-rl inspect <run_dir> --events --last 3

uv run mg-rl async-report <run_dir> [--json]       # 重なり / staleness / publish の要約
```

`resume` は最新チェックポイントのバッチ位置から再開する。`steps.jsonl` の最終行からではない。
ログに step はあるがチェックポイントが無いランは resume を拒否する。
