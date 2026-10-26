#!/usr/bin/env bash
# 2026-10-26 rerun after the terminated pod lost its outputs:
#  1. reproduce submission 2 (cot_en_role, 1,024 tokens) on the 2,238 test questions; compare with the saved file
#     (submissions/2026-10-24_cot_en_role_mt1024/predictions.jsonl, copied to the pod as reference/)
#  2. the same system plus the baseline on the 1,483 confirmation questions -> per-subject and per-type tables
#   bash runpod/rerun_1026.sh          (after runpod/setup.sh)
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${SINMULU_HF_HOME:-/workspace/hf_cache}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs
exec > >(tee -a "logs/rerun_$(date +%Y%m%d_%H%M).log") 2>&1
M=configs/models/gemma3_4b_it.yaml
fail() { echo "FAILED: $1" | tee -a logs/failed.txt; }

python -m sinmmlu.evaluate --model $M --data test --prompt cot_en_role --max-new-tokens 1024 || fail "test rerun"
NEW=outputs/gemma3_4b_it__none__cot_en_role__chaton__p1__c0__trnone__mt1024__test/predictions.jsonl
REF=reference/submission2_predictions.jsonl
python - "$NEW" "$REF" <<'PY'
import json, sys
key = lambda r: (r["level"], r["subject"], r["category"], r["q_no"])
new = {key(r): r["answer"] for r in map(json.loads, open(sys.argv[1], encoding="utf-8"))}
ref = {key(r): r["answer"] for r in map(json.loads, open(sys.argv[2], encoding="utf-8"))}
assert new.keys() == ref.keys(), "different question sets"
diff = [k for k in ref if new[k] != ref[k]]
print(f"REPRODUCTION: {len(ref) - len(diff)}/{len(ref)} answers identical to submission 2, {len(diff)} differ")
for k in diff[:20]:
    print("  differs:", k, "saved", ref[k], "rerun", new[k])
PY

R=dev/inference_rest.jsonl
python -m sinmmlu.evaluate --model $M --data $R --prompt paper_nosubject || fail "rest baseline"
python -m sinmmlu.evaluate --model $M --data $R --prompt cot_en_role --max-new-tokens 1024 || fail "rest cot_en_role 1024"
python -m sinmmlu.report --data $R --baseline gemma3_4b_it__none__paper_nosubject__chaton__p1__c0__trnone__inference_rest
python -m sinmmlu.report --data $R --by-subject results/subject_accuracy_inference_rest.csv
python -m sinmmlu.qtypes --data $R
echo "=== rerun finished $(date +%H:%M)"
