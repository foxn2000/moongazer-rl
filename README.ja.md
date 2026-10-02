# Moongazer-RL

[English](README.md)

Moongazer-RL は、[Prime Intellect verifiers](https://github.com/PrimeIntellect-ai/verifiers) v1 の
環境で LoRA policy を GRPO 系の強化学習で訓練するツールである。learner には Unsloth と TRL を、
sampler には外部の vLLM サーバーを使う。

環境には手を加えない。ロールアウト全体(task、harness、turn、judge、報酬)は verifiers が実行する。
Moongazer-RL はモデル呼び出しだけを差し替え、サンプルされた token と log 確率をそのまま記録し、
各 episode を学習バッチに変換する。

## 特徴

- **verifiers v1 の環境をそのまま使える。** `PYTHONPATH` 上の v1 taskset パッケージを
  `environment.taskset.id` で指定する。
- **token が一致した学習データ。** 再トークン化はしない。learner は sampler が返した token ID と
  sampling log 確率で学習する。この照合に通らない trajectory は除外し、件数を記録する。
- **同期と完全非同期を切り替えられる。** 同じ設定で、1 step ずつ進める同期経路と、rollout
  producer と learner を bounded queue でつないだ完全非同期経路のどちらでも動く。staleness には
  上限があり、各 trajectory は 1 つの policy version だけで生成され、off-policy の差は truncated
  importance sampling で補正する。
- **GPU 1 枚で動く。** vLLM sampler と 4bit の Unsloth learner を 32 GB のカード 1 枚に同居
  できる。learner は step ごとに LoRA adapter を vLLM へ載せ替える。
- **目的関数を組み替えられる。** advantage 推定量(`group_zscore`、`group_center`、`rloo`)、
  policy objective(`grpo_clip`、`cispo`、`reinforce`)、ratio の参照元、behavior 補正、loss の
  正規化を個別に設定できる。
- **run の中身を追える。** ロールアウト、モデル呼び出し、step をすべて JSONL に書き出す。
  `inspect` と `async-report` で run を要約でき、checkpoint から再開できる。Weights & Biases は
  任意で使える。

## 動作環境

- 学習には Linux と NVIDIA GPU が要る(`validate` だけなら CPU 環境で足りる)
- Python 3.12 と [uv](https://docs.astral.sh/uv/)
- vLLM 0.28.0 用の別の Python 環境(vLLM は Unsloth が受け付けない torch/transformers を固定するため)

## インストール

```bash
uv sync --extra unsloth    # learner: verifiers、TRL、transformers、peft、unsloth
```

sampler は独立した `vllm serve` プロセスとして動かす。

```bash
uv venv --python 3.12 vllm-env
VIRTUAL_ENV=vllm-env uv pip install "vllm==0.28.0"
```

## クイックスタート

[`examples/rtx5090-math-env/`](examples/rtx5090-math-env/) は、Prime Intellect の `math-env` で
`google/gemma-4-E2B-it` を学習する例である。sampler と learner を RTX 5090 1 枚に載せる。
ダウンロードしたものと出力はすべてこのディレクトリの中に置かれる。

```bash
examples/rtx5090-math-env/setup.sh            # learner の venv、math-env taskset、vLLM の venv
examples/rtx5090-math-env/download-model.sh   # ベース重み(約 10 GB)
examples/rtx5090-math-env/run-async-1gpu.sh   # vLLM を起動してから学習する
```

学習中または学習後に run を確認する。

```bash
uv run mg-rl inspect examples/rtx5090-math-env/runs/math-env-async-1gpu
uv run mg-rl async-report examples/rtx5090-math-env/runs/math-env-async-1gpu
```

自分の環境で学習するときは、YAML をコピーして `environment.taskset` と `model.name` を書き換え、
学習の前に `uv run mg-rl validate <config.yaml>` で確認する。

## ドキュメント

| 文書 | 内容 |
|---|---|
| [アーキテクチャ](docs/architecture.ja.md) | 構成要素、データの流れ、正規ロールアウトスキーマ |
| [設定と CLI](docs/configuration.ja.md) | 設定の全セクション、非同期経路、メモリ設定、コマンド |
| [アルゴリズム](docs/algorithms.md) | RL 目的関数の分解と Unsloth の loss への対応 |
| [Trainer backend](docs/backends.ja.md) | `unsloth-native` が TRL の `GRPOTrainer` を駆動する方法 |
| [Run ディレクトリ](docs/run-directory.ja.md) | run が書き出すファイル、ログの行キー、メトリクス、再開、W&B |
