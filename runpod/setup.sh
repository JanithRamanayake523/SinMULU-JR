#!/usr/bin/env bash
# One-time setup on a RunPod pod (PyTorch template, RTX 3090 or A40). Run from anywhere:
#   export HF_TOKEN=hf_...            # your token, in the shell only; never put it in a file in this repo
#   bash runpod/setup.sh              # final system only: google/gemma-3-4b-it (~8.6 GB)
#   bash runpod/setup.sh wave1 wave2 wave3    # every model of the screening study (~34 GB: needs a bigger disk)
# Model cache: /workspace/hf_cache (on the volume, so it survives a pod Stop; override with SINMULU_HF_HOME).
# The RunPod template presets HF_HOME=/workspace/.cache/huggingface; it is overridden here on purpose so that every
# script uses the same cache.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${SINMULU_HF_HOME:-/workspace/hf_cache}"
: "${HF_TOKEN:?export HF_TOKEN=hf_... first}"

python - <<'PY'
import torch
assert torch.cuda.is_available(), "no GPU visible"
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())
PY
pip install --progress-bar off -r requirements.txt 2>&1 | grep -E "^(Successfully|ERROR)" || true
python -m sinmmlu.build_dev
python -m pytest -q
python runpod/download.py "${@:-final}"
echo "setup done. HF_HOME=$HF_HOME"
