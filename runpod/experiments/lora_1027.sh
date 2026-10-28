#!/usr/bin/env bash
# LoRA on Global-MMLU Sinhala (2026-10-27): screening on the 280-question subset, both models on the same GPU.
#   bash runpod/experiments/lora_1027.sh          (after python -m sinmmlu.train_lora ... --out models/gemma3_4b_it_lora_gmmlu)
# Conditions: submission 2 (cot_en_role, 1,024 tokens) and the plain digit prompt the adapter was trained with
# (en_instr_nosubject), each for the base model and the merged LoRA model.
# Rule: >= +2 points over the base model's cot_en_role on the subset to go to the confirmation set.
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME="${SINMULU_HF_HOME:-/workspace/hf_cache}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs
exec > >(tee -a "logs/lora_$(date +%Y%m%d_%H%M).log") 2>&1
D=dev/archive/inference_subset.jsonl
fail() { echo "FAILED: $1" | tee -a logs/failed.txt; }

for M in gemma3_4b_it_lora_gmmlu gemma3_4b_it; do
    python -m sinmmlu.evaluate --model configs/models/$M.yaml --data $D --prompt en_instr_nosubject || fail "$M direct"
    python -m sinmmlu.evaluate --model configs/models/$M.yaml --data $D --prompt cot_en_role --max-new-tokens 1024 \
        || fail "$M cot"
done
python -m sinmmlu.report --data $D --baseline gemma3_4b_it__none__cot_en_role__chaton__p1__c0__trnone__mt1024__inference_subset
echo "=== lora screening finished $(date +%H:%M)"
