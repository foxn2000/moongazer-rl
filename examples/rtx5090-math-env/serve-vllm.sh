#!/usr/bin/env bash
# vLLM sampler for examples/rtx5090-math-env/math-env-async-1gpu.yaml, sharing GPU 0 with the
# learner. Runs in its OWN venv (vllm pins a torch/transformers that unsloth does not accept);
# examples/rtx5090-math-env/setup.sh creates it.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Everything lives under this directory. MODEL is relative to it and must equal the YAML's
# model.name, because vLLM serves the base model under the path string it was given.
cd "$HERE"
VLLM_VENV="${VLLM_VENV:-$HERE/vllm-env}"
MODEL="${MODEL:-models/gemma-4-E2B-it}"
PORT="${PORT:-8000}"
# 0.5 of a 32.6 GB card = ~16.3 GB: bf16 E2B weights 9.5 GB + KV cache. The rest of the card
# is the 4-bit learner's. Raise it only after watching the learner's peak with nvidia-smi.
GPU_UTIL="${GPU_UTIL:-0.5}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"   # >= model.max_seq_length in the YAML
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"     # >= model.lora.r in the YAML
# vLLM's torch.compile cache, kept here rather than under ~/.cache/vllm (a root-owned tree left
# there by an earlier run makes the engine die with PermissionError).
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HERE/cache/vllm}"

# VLLM_ALLOW_RUNTIME_LORA_UPDATING: /v1/load_lora_adapter is how each new weight version lands.
# --logprobs-mode processed_logprobs: report the probability the token was actually sampled
#   with, which is what the learner's truncated importance sampling expects.
# --max-loras / --max-cpu-loras: the async path keeps 2-4 adapter versions alive at once.
# --limit-mm-per-prompt: text-only rollouts, skip the vision/audio tower memory.
exec env VLLM_ALLOW_RUNTIME_LORA_UPDATING=True "$VLLM_VENV/bin/vllm" serve "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enable-lora --max-lora-rank "$MAX_LORA_RANK" --max-loras 4 --max-cpu-loras 8 \
  --logprobs-mode processed_logprobs \
  --limit-mm-per-prompt '{"image":0,"audio":0}' \
  "$@"
