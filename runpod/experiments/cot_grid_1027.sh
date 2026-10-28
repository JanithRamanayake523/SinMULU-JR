#!/usr/bin/env bash
# Reasoning-prompt grid (2026-10-27): cot_si and cot_en, 1,024 tokens, on the 280-question subset, for adapters B1
# (sinmulu_si r16 epoch 1) and A (Global-MMLU r16 epoch 1), then the base model as reference, all on the same GPU. Each merged model is built from its adapter just before its runs and deleted
# after them (disk). Already done on this GPU and reused: cot_en_role for base, B1, B2.
#   bash runpod/experiments/cot_grid_1027.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME="${SINMULU_HF_HOME:-/workspace/hf_cache}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs
exec > >(tee -a "logs/cot_grid_$(date +%Y%m%d_%H%M).log") 2>&1
D=dev/inference_subset.jsonl
fail() { echo "FAILED: $1" | tee -a logs/failed.txt; }

run() {  # run <config name>
    python -m sinmmlu.evaluate --model configs/models/$1.yaml --data $D --prompt cot_si cot_en --max-new-tokens 1024 \
        || fail "$1"
}

# <config name>  <adapter folder>  (base: no adapter)
MODELS="gemma3_4b_it_lora_sinmulu:lora_adapters/gemma3_4b_it_lora_sinmulu/adapter
gemma3_4b_it_lora_gmmlu:lora_adapters/gemma3_4b_it_lora_gmmlu/adapter
gemma3_4b_it:-"

for line in $MODELS; do
    M=${line%%:*}; A=${line#*:}
    echo "=== $M $(date +%H:%M)"
    if [ "$A" != "-" ] && [ ! -f models/$M/model.safetensors ]; then
        python -m sinmmlu.merge_lora --model configs/models/gemma3_4b_it.yaml --adapter $A --out models/$M \
            || { fail "$M merge"; continue; }
    fi
    run $M
    [ "$A" != "-" ] && rm -rf models/$M
done
python -m sinmmlu.report --data $D --baseline gemma3_4b_it__none__cot_en_role__chaton__p1__c0__trnone__mt1024__inference_subset
echo "=== cot grid finished $(date +%H:%M)"
