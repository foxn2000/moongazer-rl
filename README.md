# Moongazer-RL

[日本語](README.ja.md)

Moongazer-RL trains LoRA policies with GRPO-style reinforcement learning on
[Prime Intellect verifiers](https://github.com/PrimeIntellect-ai/verifiers) v1 environments, using
Unsloth and TRL on the learner side and an external vLLM server as the sampler.

The environment is used as-is. verifiers runs the whole rollout (tasks, harness, turns, judges,
rewards); Moongazer-RL only replaces the model call, records exactly the tokens and log-probabilities
that were sampled, and turns each episode into a training batch.

## Features

- **Unmodified verifiers v1 environments.** Point `environment.taskset.id` at any v1 taskset
  package on `PYTHONPATH`.
- **Token-faithful training data.** Nothing is re-tokenized. The learner trains on the token IDs
  and sampling log-probabilities the sampler returned, and trajectories that fail this check are
  dropped and counted.
- **Synchronous or fully asynchronous.** The same config runs either step by step or with
  rollout producers and the learner decoupled by a bounded queue. Staleness is bounded, each
  trajectory comes from a single policy version, and truncated importance sampling corrects the
  off-policy gap.
- **Fits on one GPU.** The vLLM sampler and a 4-bit Unsloth learner can share a single 32 GB
  card; the learner hot-loads its LoRA adapter into vLLM after every step.
- **Configurable objective.** Advantage estimator (`group_zscore`, `group_center`, `rloo`),
  policy objective (`grpo_clip`, `cispo`, `reinforce`), ratio reference, behavior correction and
  loss normalization are separate settings.
- **Inspectable runs.** Every rollout, model call and step is written as JSONL; `inspect` and
  `async-report` summarize a run, checkpoints support resume, and Weights & Biases is optional.

## Requirements

- Linux and an NVIDIA GPU for training (the CPU install is enough for `validate`)
- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- A separate Python environment for vLLM 0.28.0, which pins a torch/transformers version that
  Unsloth does not accept

## Installation

```bash
uv sync --extra unsloth    # learner: verifiers, TRL, transformers, peft, unsloth
```

The sampler runs as its own `vllm serve` process:

```bash
uv venv --python 3.12 vllm-env
VIRTUAL_ENV=vllm-env uv pip install "vllm==0.28.0"
```

## Quick start

[`examples/rtx5090-math-env/`](examples/rtx5090-math-env/) trains `google/gemma-4-E2B-it` on
Prime Intellect's `math-env` with the sampler and the learner on one RTX 5090. Everything it
downloads and writes stays inside that directory.

```bash
examples/rtx5090-math-env/setup.sh            # learner venv, math-env taskset, vLLM venv
examples/rtx5090-math-env/download-model.sh   # base weights (about 10 GB)
examples/rtx5090-math-env/run-async-1gpu.sh   # start vLLM, then train
```

Follow the run while it trains or after it finishes:

```bash
uv run mg-rl inspect examples/rtx5090-math-env/runs/math-env-async-1gpu
uv run mg-rl async-report examples/rtx5090-math-env/runs/math-env-async-1gpu
```

For your own environment, copy the YAML, change `environment.taskset` and `model.name`, and check
it with `uv run mg-rl validate <config.yaml>` before training.

## Documentation

| document | content |
|---|---|
| [Architecture](docs/architecture.md) | components, data flow and the canonical rollout schema |
| [Configuration and CLI](docs/configuration.md) | every config section, the asynchronous path, memory settings, commands |
| [Algorithms](docs/algorithms.md) | how the RL objective is decomposed and mapped onto Unsloth's loss (Japanese) |
| [Trainer backends](docs/backends.md) | how `unsloth-native` drives TRL's `GRPOTrainer` |
| [Run directory](docs/run-directory.md) | files a run writes, log row keys, metrics, resume, W&B |
