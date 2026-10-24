#!/usr/bin/env bash
# Screening study on dev/inference_subset.jsonl (280 questions, 20 per subject). Every step skips finished
# experiments, so a wave can simply be restarted after a crash or a pod restart.
#   bash runpod/run_subset.sh smoke      # 8 questions through every code path (~5 min), no results rows
#   bash runpod/run_subset.sh wave1      # prompts, scoring, precision, model references
#   bash runpod/run_subset.sh wave2      # few-shot, translation (NLLB, self-translation)
#   bash runpod/run_subset.sh wave3      # reasoning (Sinhala / English), MADLAD translation
#   bash runpod/run_subset.sh report     # comparison table + per-subject accuracy
# Log: logs/<wave>_<time>.log; failed steps are listed in logs/failed.txt and do not stop the wave.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/root/hf_cache PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
WAVE="${1:?usage: run_subset.sh smoke|wave1|wave2|wave3|combos|confirm|report}"
mkdir -p logs
exec > >(tee -a "logs/${WAVE}_$(date +%Y%m%d_%H%M).log") 2>&1

M=configs/models/gemma3_4b_it.yaml
D=dev/inference_subset.jsonl
BASELINE=gemma3_4b_it__none__paper_nosubject__chaton__p1__c0__trnone__inference_subset

run() {
    echo; echo ">>> $*"
    "$@" || echo "FAILED ($(date +%H:%M)): $*" | tee -a logs/failed.txt
}
ev() { run python -m sinmmlu.evaluate --data "$D" "$@"; }

case "$WAVE" in
smoke)
    L="--limit 8"
    ev --model $M $L --prompt paper_nosubject paper_subject_si letters_nosubject fewshot3_nosubject --perms 1 4 --calibrate false true
    ev --model $M $L --prompt cloze_nosubject
    ev --model $M $L --prompt cot_en
    run python -m sinmmlu.translate --backend nllb --data "$D" $L
    run python -m sinmmlu.translate --backend self --model $M --data "$D" $L
    ev --model $M $L --prompt en_nosubject bilingual_nosubject si_q_en_opts --translator nllb self
    ev --model $M $L --prompt paper_nosubject --quant 4bit
    ;;
wave1)
    # A0 and B1-B6
    ev --model $M --prompt paper_nosubject paper_subject_si paper_subject_en strict_nosubject role_subject_si \
        en_instr_nosubject letters_nosubject
    ev --model $M --prompt paper_nosubject --chat off                              # A0b
    ev --model $M --prompt paper_nosubject --perms 1 4 --calibrate false true      # C1-C3
    ev --model $M --prompt cloze_nosubject                                         # C4
    ev --model $M --prompt paper_nosubject --quant 4bit                            # C8
    ev --model configs/models/gemma3_4b_pt.yaml --prompt paper_nosubject           # C9
    ev --model configs/models/gemma3_1b_it.yaml --prompt paper_nosubject
    ;;
wave2)
    ev --model $M --prompt fewshot1_nosubject fewshot3_nosubject fewshot5_nosubject   # B7
    run python -m sinmmlu.translate --backend nllb --data "$D"
    run python -m sinmmlu.translate --backend self --model $M --data "$D"
    ev --model $M --prompt en_nosubject bilingual_nosubject si_q_en_opts --translator nllb self   # D1, D2, D4, D5
    ;;
wave3)
    ev --model $M --prompt cot_si cot_en                                             # C6, C7
    run python -m sinmmlu.translate --backend madlad --data "$D"
    ev --model $M --prompt en_nosubject bilingual_nosubject si_q_en_opts --translator madlad      # D3
    ;;
combos)
    # step 1 after wave 3: combine the winners (English reasoning, subject/role, few-shot, base model)
    run python -m sinmmlu.rationales --model $M                                     # worked examples for cot_en_fewshot2
    ev --model $M --prompt cot_en_subject cot_en_role cot_en_fewshot2
    ev --model $M --prompt fewshot3_subject_si
    ev --model configs/models/gemma3_4b_pt.yaml --prompt paper_subject_si role_subject_si fewshot3_nosubject fewshot3_subject_si
    S=__inference_subset
    P=gemma3_4b_it__none__
    T=__chaton__p1__c0__trnone$S
    run python -m sinmmlu.ensemble --name ens_scored4 --exps ${P}role_subject_si$T ${P}fewshot3_nosubject$T \
        ${P}paper_subject_si$T ${P}en_instr_nosubject$T
    run python -m sinmmlu.ensemble --name ens_cot_si_en --exps ${P}cot_en$T ${P}cot_si$T
    run python -m sinmmlu.ensemble --name ens_cot_en_scored2 --exps ${P}cot_en$T ${P}role_subject_si$T ${P}fewshot3_nosubject$T
    ;;
confirm)
    # step 2: the pre-registered candidates (docs/log.md) + baseline on the 1,483 held-out questions (~2.5 h)
    R=dev/inference_rest.jsonl
    run python -m sinmmlu.evaluate --data $R --model $M --prompt paper_nosubject                   # A0
    run python -m sinmmlu.evaluate --data $R --model $M --prompt cot_en_fewshot2 cot_en cot_en_role cot_si
    P=gemma3_4b_it__none__
    T=__chaton__p1__c0__trnone__inference_rest
    run python -m sinmmlu.ensemble --name ens_cot_si_en --exps ${P}cot_en$T ${P}cot_si$T
    run python -m sinmmlu.report --data $R --baseline ${P}paper_nosubject$T
    ;;
report)
    run python -m sinmmlu.report --data "$D" --baseline "$BASELINE"
    run python -m sinmmlu.report --data "$D" --by-subject results/subject_accuracy_inference_subset.csv
    ;;
*)
    echo "unknown wave $WAVE"; exit 2 ;;
esac
echo; echo "=== $WAVE finished $(date +%H:%M); failures so far:"; cat logs/failed.txt 2>/dev/null || echo none
