#!/usr/bin/env bash
# Submission 2 (2026-10-24 evening): confirm two candidates on the 1,483 held-out questions against the current
# winner (cot_en_role, 512 tokens, 45.6%), select by the rule (>= 1 point, Holm p < 0.05; if none passes: the
# 1,024-token version, which only fixes cut-off answers), then run the selected system on the 2,238 test questions.
#   bash runpod/confirm2.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${SINMULU_HF_HOME:-/workspace/hf_cache}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs
exec > >(tee -a "logs/confirm2_$(date +%Y%m%d_%H%M).log") 2>&1
M=configs/models/gemma3_4b_it.yaml
R=dev/inference_rest.jsonl
P=gemma3_4b_it__none__
T=__chaton__p1__c0__trnone
fail() { echo "FAILED: $1" | tee -a logs/failed.txt; exit 1; }

python -m sinmmlu.evaluate --data $R --model $M --prompt cot_en_role cot_en_role_negcheck --max-new-tokens 1024 \
    || fail "confirm2 evaluate"
python -m sinmmlu.report --data $R --baseline ${P}cot_en_role${T}__inference_rest
python -m sinmmlu.select_winner --data $R --baseline ${P}cot_en_role${T}__inference_rest \
    --candidates ${P}cot_en_role${T}__mt1024__inference_rest ${P}cot_en_role_negcheck${T}__mt1024__inference_rest \
    --fallback ${P}cot_en_role${T}__mt1024__inference_rest --out results/winner_submission2.json \
    || fail "confirm2 select_winner"
python -m sinmmlu.predict_winner --winner results/winner_submission2.json || fail "confirm2 predict_winner"
echo "=== submission 2 finished $(date +%H:%M)"
