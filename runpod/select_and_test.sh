#!/usr/bin/env bash
# After `run_subset.sh confirm`: apply the pre-registered rule on the 1,483 confirmation questions, then run the
# winner on the 2,238 public test questions -> outputs/<exp>/predictions.zip (Codabench upload is manual).
#   bash runpod/select_and_test.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${SINMULU_HF_HOME:-/workspace/hf_cache}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs
exec > >(tee -a "logs/test_$(date +%Y%m%d_%H%M).log") 2>&1
P=gemma3_4b_it__none__
T=__chaton__p1__c0__trnone__inference_rest
python -m sinmmlu.select_winner --data dev/inference_rest.jsonl --baseline ${P}paper_nosubject$T \
    --candidates ${P}cot_en_fewshot2$T ens_cot_si_en__inference_rest ${P}cot_en_role$T ${P}cot_en$T \
    || { echo "FAILED: select_winner" | tee -a logs/failed.txt; exit 1; }
python -m sinmmlu.predict_winner --winner results/winner_inference_rest.json \
    || { echo "FAILED: predict_winner" | tee -a logs/failed.txt; exit 1; }
echo "=== test predictions finished $(date +%H:%M)"
