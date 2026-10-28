#!/usr/bin/env bash
# Thinking mode (cot_en_role_think, 4,096 tokens) with the Qwen3.5-4B combined-data LoRA (2026-10-28, A100 40 GB).
#   bash runpod/think_1028.sh [subset|test]      (default: subset = dev/test_subset300.jsonl)
# Needs lora_adapters/qwen3.5-4b_lora_combined-si_epoch1/adapter; merges it first if models/ has no merged copy.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${SINMULU_HF_HOME:-/workspace/hf_cache}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs
python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"))
PY
[ -f models/qwen35_4b_lora_combined/model.safetensors ] || python -m sinmmlu.merge_lora --model configs/models/qwen35_4b.yaml \
  --adapter lora_adapters/qwen3.5-4b_lora_combined-si_epoch1/adapter --out models/qwen35_4b_lora_combined
D=dev/test_subset300.jsonl; [ "${1:-subset}" = test ] && D=test
python -m sinmmlu.evaluate --model configs/models/qwen35_4b_lora_combined.yaml --data "$D" --prompt cot_en_role_think \
  --chat on --max-new-tokens 4096 2>&1 | tee "logs/think_qwen_combined_${1:-subset}.log"
