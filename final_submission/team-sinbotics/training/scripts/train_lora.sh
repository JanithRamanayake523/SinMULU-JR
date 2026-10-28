#!/usr/bin/env bash
# Qwen3.5-4B LoRA (rank 16) on Global-MMLU si + sinmulu_si combined, 10% validation (2026-10-28, RTX 3090 24 GB).
#   bash scripts/train_lora.sh   (from training/; writes models/qwen35_4b_lora_combined/adapter = our adapter repo)
# Downloads the pinned model first; 12,288-token micro-batches to fit 24 GB.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs
python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"))
PY
python -m sinmmlu.train_lora --model configs/models/qwen35_4b.yaml \
  --train data/combined_si/competition_format_val10/train.json \
  --val data/combined_si/competition_format_val10/val.json \
  --max-batch-tokens 12288 --out models/qwen35_4b_lora_combined 2>&1 | tee logs/train_qwen_combined.log
