#!/usr/bin/env bash
# Third batch on dev/test_subset300.jsonl (2026-10-27), after runpod/test300b_1027.sh: submission 2's prompt
# cot_en_role (1,024 tokens) for adapters A, B1 and the base model.
#   bash runpod/experiments/test300c_1027.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME="${SINMULU_HF_HOME:-/workspace/hf_cache}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
while pgrep -f "runpod/test300(b)?_1027.sh" >/dev/null; do sleep 20; done
mkdir -p logs
exec > >(tee -a "logs/test300c_$(date +%Y%m%d_%H%M).log") 2>&1
D=dev/test_subset300.jsonl
fail() { echo "FAILED: $1" | tee -a logs/failed.txt; }
for line in gemma3_4b_it_lora_gmmlu:lora_adapters/gemma3_4b_it_lora_gmmlu/adapter \
            gemma3_4b_it_lora_sinmulu:lora_adapters/gemma3_4b_it_lora_sinmulu/adapter gemma3_4b_it:-; do
    M=${line%%:*}; A=${line#*:}
    echo "=== $M $(date +%H:%M)"
    if [ "$A" != "-" ] && [ ! -f models/$M/model.safetensors ]; then
        python -m sinmmlu.merge_lora --model configs/models/gemma3_4b_it.yaml --adapter $A --out models/$M \
            || { fail "$M merge"; continue; }
    fi
    python -m sinmmlu.evaluate --model configs/models/$M.yaml --data $D --prompt cot_en_role --max-new-tokens 1024 \
        || fail "$M cot_en_role"
    [ "$A" != "-" ] && rm -rf models/$M
done
ls outputs/*__test_subset300/predictions.zip
echo "=== test300c finished $(date +%H:%M)"
