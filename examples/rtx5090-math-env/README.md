# Fully asynchronous LoRA GRPO on one RTX 5090 (math-env)

[日本語](README.ja.md)

An example that trains `google/gemma-4-E2B-it` fully asynchronously on one RTX 5090
(32 GB), with the vLLM sampler and the Unsloth learner co-resident on the same GPU. The
environment is Prime Intellect's verifiers v1 `math-env` 0.1.1 (Hendrycks-Math, 7474 items),
scored by boxed-answer matching via `math-verify`. No container is needed.

```text
GPU 0 (RTX 5090, 32.6 GB)
  ├─ vllm serve        --gpu-memory-utilization 0.5  → measured 18.5-19.1 GB (bf16 weights + KV)
  └─ mg-rl train  load_in_4bit: true            → measured 8.0-9.6 GB (4bit weights + LoRA + activations)

  producer x2 ──[RolloutGroup]──▶ bounded queue ──▶ learner
       │                                              │
       └─ rollout (subprocess runtime)                └─ adapter export → load_lora_adapter
```

There is no synchronous barrier. Producers lease the published version's adapter to roll out,
the learner pulls `groups_per_step` groups from the queue and runs one step, and the publisher
loads the new version into vLLM asynchronously.

## About the environment

The Environments Hub's `primeintellect/math-env` is verifiers v0 (`runtime: VERIFIERS_V0`,
exports only `load_environment`) and cannot be plugged into this tool's v1-only path.
Specifying it in `taskset.id` fails at resolution time.

```text
AttributeError: taskset module 'math_env' defines no `__all__`; a taskset must export its
                Taskset subclass via `__all__`.
```

Prime Intellect's official repository
[research-environments](https://github.com/PrimeIntellect-ai/research-environments)
(Apache 2.0) has a v1 rewrite of the same environment, and this is what prime-rl actually uses
for training.

| | Hub version (v0) | research-environments version (v1) |
|---|---|---|
| version | `math-env` 0.1.6 | `math-env` 0.1.1 |
| scoring | HybridMathRubric, prime-sandboxes | `math-verify`, only verifiers as a dependency |
| data | — | Hendrycks-Math (a different rewrite) |

The v1 package requires `verifiers>=0.3.1`, but this repository pins `verifiers==0.2.0`.
Install it with `--no-deps`. The v1 API it uses (`Task` / `@vf.reward` runtime injection,
`TasksetConfig`, `TaskConfig`) exists in 0.2.0.

## Steps

```bash
# 1. learner venv, math-env taskset, sampler venv
examples/rtx5090-math-env/setup.sh

# 2. base weights to examples/rtx5090-math-env/models/ (about 10.3 GB)
examples/rtx5090-math-env/download-model.sh

# 3. training (bring vLLM up first, then load the learner onto the same GPU)
examples/rtx5090-math-env/run-async-1gpu.sh
```

Everything is created under this directory, so the example runs as-is from any clone:

| path | content | created by |
|---|---|---|
| `vllm-env/` | sampler venv (vllm 0.28.0) | `setup.sh` |
| `envs/` | `math-env` taskset | `setup.sh` |
| `models/gemma-4-E2B-it/` | base weights | `download-model.sh` |
| `cache/vllm/` | vLLM torch.compile cache | `serve-vllm.sh` |
| `runs/` | run directory (`logging.output_dir`) | `run-async-1gpu.sh` |
| `vllm-serve.log` | sampler log | `run-async-1gpu.sh` |

The scripts `cd` here, so the YAML's relative `model.name` and `logging.output_dir` resolve
against this directory. The learner venv is the repository's `.venv`.

`setup.sh` finishes by running the config through `mg-rl validate`. The taskset is placed
in `examples/rtx5090-math-env/envs/`, not the venv, and read via `PYTHONPATH`. `uv run`
re-syncs `.venv` to match `uv.lock` by default, so anything installed directly into the venv is
removed on the next `uv run`.

Check progress and throughput after the run.

```bash
uv run mg-rl inspect examples/rtx5090-math-env/runs/math-env-async-1gpu
uv run mg-rl async-report examples/rtx5090-math-env/runs/math-env-async-1gpu
```

## Memory allocation

The 32.6 GB is split between 2 processes. With the sampler at 0.5, measured usage is vLLM
18.5-19.1 GB (KV cache 5.5-6.1 GiB, 55-61 concurrent requests at 8192 tokens/req), learner
8.0-9.6 GB, peak 28.7 GB.

| variable | default | effect |
|---|---|---|
| `GPU_UTIL` | `0.5` | share vLLM reserves. Raising it grows the KV cache and increases concurrent rollout capacity |
| `MAX_MODEL_LEN` | `8192` | keep at or above the YAML's `model.max_seq_length` |
| `model.load_in_4bit` | `true` | setting `false` costs the learner side 9.5 GB in bf16 as well |
| `rollout.max_concurrent` | `16` | concurrent rollout count |

To run the learner in bf16, the bf16 weights become 19 GB across the two copies, and adding
activations and the KV cache no longer fits in 32 GB. Either split the sampler and learner
across separate cards, or lower `GPU_UTIL` to around 0.35 and run with a short
`max_seq_length`.

## What we hit at startup

If `~/.cache/vllm/torch_compile_cache/torch_aot_compile` is owned by root, the engine core
crashes with `PermissionError: [Errno 13]`. This is left behind by a past run of vLLM as root.
`serve-vllm.sh` therefore sets `VLLM_CACHE_ROOT` to `cache/vllm/` under this directory;
override it to put the cache elsewhere.

## Measured (12 steps)

One run, completed with `--set rollout.max_steps=12`. One RTX 5090; `status: completed`, with
checkpoints at step 10 and 12.

| item | math-env |
|---|---|
| run time / rollout count | 473 s / 440 rollouts |
| **overlap of model-call and learner intervals** | 135.3 / 137.7 s = **98.3%** |
| overlap with episode intervals | 98.4% |
| staleness distribution | `{0: 14, 1: 10, 2: 1}` |
| drops by the staleness gate | 1 group |
| publish latency | mean 74 ms / max 80 ms |
| rollout throughput | 1299 tok/s |
| learner throughput (action tokens) | 475 tok/s |
| learner idle | 292 s = 61.8% |
| mean per rollout | 12.0 s |
| mean per episode (1 task x G=8) | 15.3 s |
| zero-std drops | 29 groups (producer side) |
| failure modes | rollout failures 0, lease unavailable 0, fidelity violations 0, train skips 0 |
| reward | 129 of 440 correct (29.3%) |
| IS cap hit rate / `behavior_ess_relative` | 0.95% / mean 0.69 (min 0.31) |

Asynchrony is demonstrated by this overlap (wall-clock interval intersection). We do not claim
GPU-kernel-level concurrency — the SMs are time-sliced since one card is shared.

One group at staleness 2 appeared and was dropped by the consumer-side gate. Admission
reservations don't fully absorb publish delay, so it is the gate that guarantees correctness —
this is the designed behavior actually firing.

The effect of the sampler being bf16 and the learner being 4bit shows up. However, there is no
systematic difference between staleness 0 and 1 (0.629 vs. 0.798), so this gap comes from the
quantization gap, not staleness.

The run is rollout-bound (learner idle 61.8%) — the learner is not the one waiting.

Over 12 steps, reward trends are buried in per-step variance (0.125-0.812, 16 rollouts per
step). Whether training is actually working cannot be judged from this many steps.

## Not verified

- Whether generation on the old version continues while a new version loads with
  `--max-loras 4` still in place (the publish rollback/retry path also never fired)
- Long runs beyond 12 steps, the resume path
- Memory headroom with `GPU_UTIL` raised, or with `load_in_4bit: false`
- Speed advantage over the synchronous path (no synchronous baseline with the same config has
  been run)
