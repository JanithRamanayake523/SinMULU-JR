#!/usr/bin/env bash
# One-time setup on a RunPod pod (PyTorch template, RTX 3090 or A40). Run from anywhere:
#   export HF_TOKEN=hf_...            # your token, in the shell only; never put it in a file in this repo
#   bash runpod/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
# Always the container disk (50 GB): the /workspace volume (30 GB) is too small for all models, and the RunPod
# template presets HF_HOME=/workspace/.cache/huggingface, so it is overridden here on purpose.
export HF_HOME=/root/hf_cache
: "${HF_TOKEN:?export HF_TOKEN=hf_... first}"

python - <<'PY'
import torch
assert torch.cuda.is_available(), "no GPU visible"
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())
PY
pip install -q -r requirements.txt
python -m sinmmlu.build_dev
python -m pytest -q
python runpod/download.py wave1 wave2 wave3
echo "setup done. HF_HOME=$HF_HOME"
