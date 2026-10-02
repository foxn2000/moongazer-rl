#!/usr/bin/env bash
# Fully asynchronous run with the sampler and the learner on the SAME RTX 5090.
#   examples/rtx5090-math-env/run-async-1gpu.sh [-- extra mg-rl args]
#
# Order matters: vLLM starts first and reserves --gpu-memory-utilization of the card, then
# the learner allocates out of what is left. Starting the learner first leaves vLLM short.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-$HERE/math-env-async-1gpu.yaml}"
PORT="${PORT:-8000}"
GPU="${GPU:-0}"
VLLM_LOG="${VLLM_LOG:-$HERE/vllm-serve.log}"

# Relative paths in the YAML (model, output_dir) resolve against this directory.
cd "$HERE"

# No setsid: it forks when the caller is already a process-group leader, so $! would be a
# wrapper that exits immediately. serve-vllm.sh execs vllm, so plain `&` makes $VLLM_PID the
# server itself, which is what the trap has to signal.
CUDA_VISIBLE_DEVICES="$GPU" PORT="$PORT" GPU_UTIL="${GPU_UTIL:-0.5}" \
  nohup "$HERE/serve-vllm.sh" > "$VLLM_LOG" 2>&1 < /dev/null &
VLLM_PID=$!
echo "vllm serve pid $VLLM_PID (GPU $GPU, log $VLLM_LOG)"
trap 'kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  if curl -fs "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then break; fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "vLLM died; see $VLLM_LOG"; exit 1; fi
  sleep 5
done
curl -fs "http://127.0.0.1:$PORT/v1/models" > /dev/null || { echo "vLLM did not come up"; exit 1; }

# expandable_segments: the learner shares the card with vLLM, so its allocator must give
# fragmented blocks back instead of holding them.
CUDA_VISIBLE_DEVICES="$GPU" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH="$HERE/envs" \
  uv run --no-sync mg-rl train "$CONFIG" "$@"
