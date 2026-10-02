#!/usr/bin/env bash
# Fetch the base weights for examples/rtx5090-math-env/math-env-async-1gpu.yaml.
# google/gemma-4-E2B-it is ~10.3 GB (one safetensors shard) and is not gated.
#   examples/rtx5090-math-env/download-model.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-google/gemma-4-E2B-it}"
DEST="${DEST:-$HERE/models/gemma-4-E2B-it}"

mkdir -p "$DEST"
# uvx: the CLI runs from its own throwaway environment, so nothing lands in the repo venv.
uvx --from huggingface_hub hf download "$MODEL" --local-dir "$DEST"
echo "model: $DEST"
