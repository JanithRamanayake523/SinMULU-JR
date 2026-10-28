#!/usr/bin/env bash
# Answer files for the level-balanced test subset dev/test_subset300.jsonl (2026-10-27): adapters B1 (sinmulu_si r16
# epoch 1) and A (Global-MMLU r16 epoch 1), prompts cot_si and cot_en, 1,024 tokens. No answers are known here: each run
# writes outputs/<exp_id>/predictions.jsonl/.zip (300 lines) for scoring outside this code.
#   bash runpod/experiments/test300_1027.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME="${SINMULU_HF_HOME:-/workspace/hf_cache}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs
exec > >(tee -a "logs/test300_$(date +%Y%m%d_%H%M).log") 2>&1
D=dev/test_subset300.jsonl
fail() { echo "FAILED: $1" | tee -a logs/failed.txt; }
for line in gemma3_4b_it_lora_sinmulu:lora_adapters/gemma3_4b_it_lora_sinmulu/adapter \
            gemma3_4b_it_lora_gmmlu:lora_adapters/gemma3_4b_it_lora_gmmlu/adapter; do
    M=${line%%:*}; A=${line#*:}
    echo "=== $M $(date +%H:%M)"
    [ -f models/$M/model.safetensors ] || python -m sinmmlu.merge_lora --model configs/models/gemma3_4b_it.yaml \
        --adapter $A --out models/$M || { fail "$M merge"; continue; }
    python -m sinmmlu.evaluate --model configs/models/$M.yaml --data $D --prompt cot_si cot_en --max-new-tokens 1024 \
        || fail "$M"
    rm -rf models/$M
done
ls outputs/*__test_subset300/predictions.zip
echo "=== test300 finished $(date +%H:%M)"
