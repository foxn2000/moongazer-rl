#!/usr/bin/env bash
# One-time setup for examples/rtx5090-math-env/: the learner venv, the math-env taskset and the
# sampler venv.
#   examples/rtx5090-math-env/setup.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VLLM_VENV="${VLLM_VENV:-$HERE/vllm-env}"
# math-env (v1, 0.1.1) lives in PrimeIntellect's research-environments repo, not on the
# Environments Hub (the Hub's math-env is a verifiers v0 package). Pinned to a commit so a run
# is reproducible.
RE_COMMIT="${RE_COMMIT:-c8c00c8cb969a7ac9456cb9d1b837dc1e45108b0}"
RE_REPO="git+https://github.com/PrimeIntellect-ai/research-environments.git@${RE_COMMIT}"
MATH_SPEC="math-env @ ${RE_REPO}#subdirectory=environments/math/math_env"

cd "$REPO"

# 1. Learner venv (.venv): this package plus the Linux-only unsloth extra.
uv sync --extra unsloth

# 2. The taskset, onto PYTHONPATH rather than into .venv -- `uv run` re-syncs .venv from
#    uv.lock and would prune anything installed outside it. --no-deps because math-env declares
#    verifiers>=0.3.1 while this repo pins verifiers==0.2.0; every v1 API it uses exists in
#    0.2.0, and verifiers is its only dependency.
uv pip install --python "$REPO/.venv/bin/python" --target "$HERE/envs" --no-deps "$MATH_SPEC"

# 3. Sampler venv: vllm pins a torch/transformers unsloth rejects, so it gets its own
#    interpreter and its own process.
uv venv --python 3.12 "$VLLM_VENV"
VIRTUAL_ENV="$VLLM_VENV" uv pip install "vllm==0.28.0"

cd "$HERE"
PYTHONPATH="$HERE/envs" uv run --no-sync mg-rl validate math-env-async-1gpu.yaml
