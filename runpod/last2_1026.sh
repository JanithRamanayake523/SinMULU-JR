#!/usr/bin/env bash
# Last two screening ideas on the 280-question subset (2026-10-26), reference = cot_en_role at 1,024 tokens:
#  1. option-order averaging with reasoning: cot_en_role at 1,024 tokens under 2 cyclic option orders, averaged
#  2. ensemble of the three confirmed English-reasoning prompts, all at 1,024 tokens
# Rule: >= +2 points on the subset to go to the confirmation set.
#   bash runpod/last2_1026.sh      (needs dev/fewshot_cot_en.jsonl, the worked examples made on 24 Oct)
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${SINMULU_HF_HOME:-/workspace/hf_cache}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs
exec > >(tee -a "logs/last2_$(date +%Y%m%d_%H%M).log") 2>&1
M=configs/models/gemma3_4b_it.yaml
D=dev/inference_subset.jsonl
G=gemma3_4b_it__none__
T=__chaton__p1__c0__trnone__mt1024__inference_subset
fail() { echo "FAILED: $1" | tee -a logs/failed.txt; }

python -m sinmmlu.evaluate --model $M --data $D --prompt cot_en_role --max-new-tokens 1024 --perms 2 || fail "cot perms2"
python -m sinmmlu.evaluate --model $M --data $D --prompt cot_en cot_en_fewshot2 --max-new-tokens 1024 || fail "ensemble parts"
python -m sinmmlu.ensemble --name ens_cot3_mt1024 \
    --exps ${G}cot_en$T ${G}cot_en_role$T ${G}cot_en_fewshot2$T || fail "ensemble"
python -m sinmmlu.report --data $D --baseline ${G}cot_en_role$T
python -m sinmmlu.qtypes --data $D
echo "=== last2 finished $(date +%H:%M)"
