#!/usr/bin/env bash
# One-time setup: create the python environment. Run on the login node (needs internet for pip).
#   bash scripts/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# module load python/3.11 cuda/12.4      # <-- EDIT for your cluster (see env.sh)

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 1) torch: install the build that matches the cluster's CUDA driver. Examples (pick ONE, check `nvidia-smi` for the CUDA version):
#    pip install torch --index-url https://download.pytorch.org/whl/cu124
#    pip install torch --index-url https://download.pytorch.org/whl/cu121
# If the cluster provides a torch module/conda env instead, skip this and load it in env.sh.
if ! python -c "import torch" 2>/dev/null; then
  echo "torch is not installed yet: install it first (see the comment above), then re-run this script."
  exit 1
fi

# 2) everything else, pinned to the versions used on Colab
pip install -r requirements-cluster.txt

python - <<'PY'
import sys, torch, transformers
print("python", sys.version.split()[0], "| torch", torch.__version__, "| transformers", transformers.__version__)
print("cuda available:", torch.cuda.is_available(), "| gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
echo "environment ready"
