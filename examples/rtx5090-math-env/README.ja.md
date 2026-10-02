# RTX 5090 1 枚での完全非同期 LoRA GRPO(math-env)

[English](README.md)

`google/gemma-4-E2B-it` を、1 枚の RTX 5090(32 GB)上で vLLM sampler と Unsloth learner を
同居させたまま完全非同期に学習する例。環境は Prime Intellect の verifiers v1 `math-env` 0.1.1
(Hendrycks-Math 7474 件)で、`math-verify` で boxed answer を照合して採点する。コンテナは不要。

```text
GPU 0 (RTX 5090, 32.6 GB)
  ├─ vllm serve        --gpu-memory-utilization 0.5  → 実測 18.5〜19.1 GB(bf16 重み + KV)
  └─ mg-rl train  load_in_4bit: true            → 実測 8.0〜9.6 GB(4bit 重み + LoRA + 活性)

  producer x2 ──[RolloutGroup]──▶ bounded queue ──▶ learner
       │                                              │
       └─ rollout(subprocess runtime)                 └─ adapter export → load_lora_adapter
```

同期の barrier は無い。producer は published version の adapter を lease してロールアウトし、
learner は queue から groups_per_step 個を取って 1 step 回し、publisher が新 version を
非同期に vLLM へ載せる。

## 環境について

Environments Hub の `primeintellect/math-env` は verifiers v0(`runtime: VERIFIERS_V0`、
`load_environment` だけを export)で、このツールの v1 専用経路に挿せない。`taskset.id` に
指定すると解決時に落ちる。

```text
AttributeError: taskset module 'math_env' defines no `__all__`; a taskset must export its
                Taskset subclass via `__all__`.
```

Prime Intellect の公式リポジトリ
[research-environments](https://github.com/PrimeIntellect-ai/research-environments)(Apache 2.0)
には同じ環境の v1 書き直しがあり、prime-rl が実際に学習に使っているのはこちらである。

| | Hub 版(v0) | research-environments 版(v1) |
|---|---|---|
| バージョン | `math-env` 0.1.6 | `math-env` 0.1.1 |
| 採点 | HybridMathRubric、prime-sandboxes | `math-verify`、依存は verifiers のみ |
| データ | — | Hendrycks-Math(別物の書き直し) |

v1 パッケージは `verifiers>=0.3.1` を要求するが、このリポジトリは `verifiers==0.2.0` を
pin している。`--no-deps` で入れる。使う v1 API(`Task` / `@vf.reward` の runtime 注入、
`TasksetConfig`、`TaskConfig`)は 0.2.0 に存在する。

## 手順

```bash
# 1. learner venv、math-env taskset、sampler venv
examples/rtx5090-math-env/setup.sh

# 2. ベース重みを examples/rtx5090-math-env/models/ へ(約 10.3 GB)
examples/rtx5090-math-env/download-model.sh

# 3. 学習(vLLM を先に上げてから learner を同じ GPU に載せる)
examples/rtx5090-math-env/run-async-1gpu.sh
```

生成物はすべてこのディレクトリの下に作るので、どの clone からでもそのまま動く。

| パス | 内容 | 作るもの |
|---|---|---|
| `vllm-env/` | sampler の venv(vllm 0.28.0) | `setup.sh` |
| `envs/` | `math-env` taskset | `setup.sh` |
| `models/gemma-4-E2B-it/` | ベース重み | `download-model.sh` |
| `cache/vllm/` | vLLM の torch.compile キャッシュ | `serve-vllm.sh` |
| `runs/` | run ディレクトリ(`logging.output_dir`) | `run-async-1gpu.sh` |
| `vllm-serve.log` | sampler のログ | `run-async-1gpu.sh` |

スクリプトはこのディレクトリへ `cd` するので、YAML の相対パス `model.name` と
`logging.output_dir` はここを基準に解決される。learner の venv はリポジトリの `.venv` を使う。

`setup.sh` は最後に設定を `mg-rl validate` にかける。taskset は venv ではなく
`examples/rtx5090-math-env/envs/` へ入れて `PYTHONPATH` で読む。`uv run` は既定で
`uv.lock` に合わせて `.venv` を同期し直すため、venv へ直接入れると次の `uv run` で消える。

途中経過とスループットは実行後に確認する。

```bash
uv run mg-rl inspect examples/rtx5090-math-env/runs/math-env-async-1gpu
uv run mg-rl async-report examples/rtx5090-math-env/runs/math-env-async-1gpu
```

## メモリ配分

32.6 GB を 2 プロセスで分ける。sampler 0.5 での実測は vLLM 18.5〜19.1 GB(KV cache
5.5〜6.1 GiB、8192 token/req で同時 55〜61 本)、learner 8.0〜9.6 GB、ピーク 28.7 GB。

| 変数 | 既定 | 効果 |
|---|---|---|
| `GPU_UTIL` | `0.5` | vLLM が確保する割合。上げると KV cache が増えて同時ロールアウトが捌ける |
| `MAX_MODEL_LEN` | `8192` | YAML の `model.max_seq_length` 以上にする |
| `model.load_in_4bit` | `true` | `false` にすると learner 側も bf16 で 9.5 GB 要る |
| `rollout.max_concurrent` | `16` | 同時ロールアウト数 |

learner を bf16 にしたい場合、bf16 の重みが 2 部で 19 GB になり、活性と KV cache を
足すと 32 GB に収まらない。sampler と learner を別カードに分けるか、`GPU_UTIL` を
0.35 前後まで下げて短い `max_seq_length` で回す。

## 起動時に踏んだもの

`~/.cache/vllm/torch_compile_cache/torch_aot_compile` が root 所有だと、engine core が
`PermissionError: [Errno 13]` で落ちる。過去に root で vLLM を動かすと残る。そのため
`serve-vllm.sh` は `VLLM_CACHE_ROOT` をこのディレクトリの `cache/vllm/` に向ける。
別の場所に置くときは上書きする。

## 実測(12 step)

`--set rollout.max_steps=12` で完走させた 1 本。RTX 5090 1 枚、`status: completed`、
checkpoint は step 10 と 12。

| 項目 | math-env |
|---|---|
| run 時間 / ロールアウト数 | 473 秒 / 440 本 |
| **model 呼び出しと learner 区間の重なり** | 135.3 / 137.7 秒 = **98.3%** |
| episode 区間との重なり | 98.4% |
| staleness 分布 | `{0: 14, 1: 10, 2: 1}` |
| staleness gate による drop | 1 group |
| publish latency | 平均 74 ms / 最大 80 ms |
| rollout throughput | 1299 tok/s |
| learner throughput(action token) | 475 tok/s |
| learner idle | 292 秒 = 61.8% |
| 1 ロールアウト平均 | 12.0 秒 |
| 1 エピソード(1 task x G=8)平均 | 15.3 秒 |
| zero-std drop | 29 group(producer 側) |
| 失敗系 | rollout 失敗 0、lease unavailable 0、fidelity violation 0、train skip 0 |
| 報酬 | 440 本中 129 本正解(29.3%) |
| IS cap 到達率 / `behavior_ess_relative` | 0.95% / 平均 0.69(最小 0.31) |

非同期であることはこの重なり(wall-clock の区間交差)で示している。GPU カーネルレベルの
同時実行は主張しない。1 枚を共有しているので SM は時分割である。

staleness 2 の group が 1 つ現れ、consumer 側の gate が捨てた。admission の予約は
publish の遅れを完全には吸収しないので、正しさを担保しているのは gate の方であるという
設計どおりの挙動が実際に発火している。

sampler が bf16、learner が 4bit である影響も出ている。ただし staleness 0 と 1 の間に
系統差は無い(0.629 対 0.798)ので、この差は staleness ではなく量子化ギャップに由来する。

rollout 律速(learner idle 61.8%)で、learner が待たされてはいない。

12 step では報酬の推移は step ごとの分散に埋もれる(0.125〜0.812、1 step 16 本)。
学習が効いているかどうかはこの本数では判定できない。

## 未検証

- `--max-loras 4` のまま新 version の load 中に旧 version の生成が続くか(publish の
  ロールバック・再試行経路も含めて発火していない)
- 12 step を超えた長時間運転、resume 経路
- `GPU_UTIL` を上げた場合、`load_in_4bit: false` にした場合のメモリ余裕
- 同期経路に対する速度優位(同じ設定の同期ベースラインを回していない)
