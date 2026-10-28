#!/usr/bin/env bash
# Adapter B (sinmulu_si), epochs 1 and 2, vs the base model on the 280-question subset, all on the same GPU (2026-10-27).
#   bash runpod/experiments/loraB_1027.sh
# Main condition: submission 2 (cot_en_role, 1,024 tokens); also the plain digit prompt the adapters were trained with.
# Rule: >= +2 points over the base model's cot_en_role on the subset to go to the confirmation set.
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME="${SINMULU_HF_HOME:-/workspace/hf_cache}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs
exec > >(tee -a "logs/loraB_$(date +%Y%m%d_%H%M).log") 2>&1
D=dev/archive/inference_subset.jsonl
fail() { echo "FAILED: $1" | tee -a logs/failed.txt; }

for M in gemma3_4b_it gemma3_4b_it_lora_sinmulu gemma3_4b_it_lora_sinmulu_ep2; do
    python -m sinmmlu.evaluate --model configs/models/$M.yaml --data $D --prompt cot_en_role --max-new-tokens 1024 \
        || fail "$M cot"
done
for M in gemma3_4b_it gemma3_4b_it_lora_sinmulu gemma3_4b_it_lora_sinmulu_ep2; do
    python -m sinmmlu.evaluate --model configs/models/$M.yaml --data $D --prompt en_instr_nosubject || fail "$M direct"
done
python -m sinmmlu.report --data $D --baseline gemma3_4b_it__none__cot_en_role__chaton__p1__c0__trnone__mt1024__inference_subset
echo "=== loraB screening finished $(date +%H:%M)"
