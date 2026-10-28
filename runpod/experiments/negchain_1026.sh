#!/usr/bin/env bash
# Negation-consistency chain on the 280-question subset (2026-10-26):
#   reference = cot_en_role at 1,024 tokens (the submitted system); candidate = cot_en_role_negchain
#   (all correct options -> opposite question -> one final answer). Rule: >= +2 points on the subset to confirm.
#   bash runpod/experiments/negchain_1026.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME="${SINMULU_HF_HOME:-/workspace/hf_cache}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs
exec > >(tee -a "logs/negchain_$(date +%Y%m%d_%H%M).log") 2>&1
M=configs/models/gemma3_4b_it.yaml
D=dev/archive/inference_subset.jsonl
REF=gemma3_4b_it__none__cot_en_role__chaton__p1__c0__trnone__mt1024__inference_subset
python -m sinmmlu.evaluate --model $M --data $D --prompt cot_en_role --max-new-tokens 1024 \
    || echo "FAILED: reference" | tee -a logs/failed.txt
python -m sinmmlu.evaluate --model $M --data $D --prompt cot_en_role_negchain \
    || echo "FAILED: negchain" | tee -a logs/failed.txt
python -m sinmmlu.report --data $D --baseline $REF
python -m sinmmlu.qtypes --data $D
echo "=== negchain finished $(date +%H:%M)"
