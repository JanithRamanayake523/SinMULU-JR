#!/usr/bin/env bash
# Second batch on dev/test_subset300.jsonl (2026-10-27), after runpod/test300_1027.sh:
#   base model: cot_si, cot_en (1,024 tokens)
#   A, B1, base: direct prompts paper_nosubject, paper_subject_si, en_instr_nosubject, role_subject_si
#   bash runpod/experiments/test300b_1027.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME="${SINMULU_HF_HOME:-/workspace/hf_cache}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
while pgrep -f "runpod/test300_1027.sh" >/dev/null; do sleep 20; done
mkdir -p logs
exec > >(tee -a "logs/test300b_$(date +%Y%m%d_%H%M).log") 2>&1
D=dev/test_subset300.jsonl
DIRECT="paper_nosubject paper_subject_si en_instr_nosubject role_subject_si"
fail() { echo "FAILED: $1" | tee -a logs/failed.txt; }
for line in gemma3_4b_it_lora_gmmlu:lora_adapters/gemma3_4b_it_lora_gmmlu/adapter \
            gemma3_4b_it_lora_sinmulu:lora_adapters/gemma3_4b_it_lora_sinmulu/adapter gemma3_4b_it:-; do
    M=${line%%:*}; A=${line#*:}
    echo "=== $M $(date +%H:%M)"
    if [ "$A" != "-" ] && [ ! -f models/$M/model.safetensors ]; then
        python -m sinmmlu.merge_lora --model configs/models/gemma3_4b_it.yaml --adapter $A --out models/$M \
            || { fail "$M merge"; continue; }
    fi
    python -m sinmmlu.evaluate --model configs/models/$M.yaml --data $D --prompt $DIRECT || fail "$M direct"
    if [ "$A" = "-" ]; then
        python -m sinmmlu.evaluate --model configs/models/$M.yaml --data $D --prompt cot_si cot_en \
            --max-new-tokens 1024 || fail "$M cot"
    fi
    [ "$A" != "-" ] && rm -rf models/$M
done
ls outputs/*__test_subset300/predictions.zip
echo "=== test300b finished $(date +%H:%M)"
