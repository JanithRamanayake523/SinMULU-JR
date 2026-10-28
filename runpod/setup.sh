#!/usr/bin/env bash
# One-time setup on a RunPod pod (PyTorch template, torch 2.8, >= 24 GB GPU with bf16). Run from anywhere:
#   bash runpod/setup.sh              # final system only: Qwen/Qwen3.5-4B (~9 GB, not gated)
#   export HF_TOKEN=hf_...            # only for gated models (Gemma); in the shell only, never in a file in this repo
#   bash runpod/setup.sh gemma        # the Gemma-3-4B-it systems (submissions 1-2, Gemma LoRA runs)
#   bash runpod/setup.sh wave1 wave2 wave3    # every model of the screening study (~34 GB: needs a bigger disk)
# Model cache: /workspace/hf_cache (on the volume, so it survives a pod Stop; override with SINMULU_HF_HOME).
# The RunPod template presets HF_HOME=/workspace/.cache/huggingface; it is overridden here on purpose so that every
# script uses the same cache.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${SINMULU_HF_HOME:-/workspace/hf_cache}"

python - <<'PY'
import torch
assert torch.cuda.is_available(), "no GPU visible"
x = torch.ones(1024, 1024, device="cuda") @ torch.ones(1024, 1024, device="cuda")  # a GPU can be visible yet refuse work
torch.cuda.synchronize()
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())
PY
pip install --progress-bar off -r requirements.txt 2>&1 | grep -E "^(Successfully|ERROR)" || true
pip install --progress-bar off causal-conv1d==1.7.0 --no-build-isolation 2>&1 | grep -E "^(Successfully|ERROR)" || true
python -m sinmmlu.build_dev
python -m pytest -q
python runpod/download.py "${@:-final}"
echo "setup done. HF_HOME=$HF_HOME"
