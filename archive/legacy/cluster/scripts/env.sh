#!/usr/bin/env bash
# Common environment, sourced by every script. Adapt the module lines to your cluster (ask the cluster docs / admin).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- cluster modules (EDIT: names differ per cluster; run `module avail` to see them) ---------------------------------
# module load python/3.11
# module load cuda/12.4

# --- python environment ---------------------------------------------------------------------------------------------
if [ -f "$ROOT/.venv/bin/activate" ]; then
  source "$ROOT/.venv/bin/activate"
fi

# --- Hugging Face cache and token ------------------------------------------------------------------------------------
# Models are downloaded once (scripts/download_models.sh on a node WITH internet) into this cache and then used offline.
export HF_HOME="${HF_HOME:-$ROOT/hf_cache}"
# The token is needed ONLY for scripts/download_models.py (gated Gemma / Llama). Either `export HF_TOKEN=...` in the shell, or
# put the token alone on one line in the file ROOT/.hf_token (create it yourself on the server: chmod 600 .hf_token).
if [ -z "${HF_TOKEN:-}" ] && [ -f "$ROOT/.hf_token" ]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$ROOT/.hf_token")"
fi
if [ "${OFFLINE:-0}" = "1" ]; then
  export HF_HUB_OFFLINE=1        # compute nodes without internet: use the cache only
  export TRANSFORMERS_OFFLINE=1
fi

# --- determinism / quiet logs ----------------------------------------------------------------------------------------
export PYTHONHASHSEED=0
export PYTHONUTF8=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
