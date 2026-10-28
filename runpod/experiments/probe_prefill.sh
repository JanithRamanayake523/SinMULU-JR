#!/usr/bin/env bash
# Label mass and accuracy on the first N subset questions with the answer pre-fill built into scoring, for the main
# prompt families (dev data only). Usage: bash runpod/experiments/probe_prefill.sh [N]
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME="${SINMULU_HF_HOME:-/workspace/hf_cache}"
N="${1:-40}"
M=configs/models/gemma3_4b_it.yaml
D=dev/archive/inference_subset.jsonl
dbg() { python -m sinmmlu.debug --model "$M" --data "$D" --n "$N" --quiet "$@" 2>&1 | grep -E "^SUMMARY"; }
dbg --prompt paper_nosubject --no-prefill
dbg --prompt paper_nosubject
dbg --prompt paper_nosubject --chat off
dbg --prompt fewshot3_nosubject
dbg --prompt en_instr_nosubject
dbg --prompt letters_nosubject
python -m sinmmlu.debug --model "$M" --data "$D" --n 2 --prompt letters_nosubject --generate 12 2>&1 \
    | grep -E "^===|label token|last tokens|top next|greedy"
