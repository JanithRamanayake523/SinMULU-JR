#!/usr/bin/env bash
# The debug notebook as a script: tokenisation, scoring table, next-token detail, summary. Output goes to a text file.
#   MODEL_KEY=qwen4b PROMPT=nosubject_strict CHAT=on PERMS=1 CALIBRATE=false N=20 bash scripts/run_debug.sh
# Variables: MODEL_KEY (qwen4b|gemma4b|llama3b, or the 1B models qwen|gemma|llama), PROMPT, CHAT (on|off), PERMS (1|4), CALIBRATE (true|false), N, QUESTION_INDEX,
#            PERM_INDEX (0-3), GENERATE (tokens, 0 = off), SECTIONS (tokenization,scoring,detail,summary)
source "$(dirname "$0")/env.sh"
MODEL_KEY="${MODEL_KEY:-qwen4b}"
PROMPT="${PROMPT:-nosubject_strict}"
CHAT="${CHAT:-auto}"
PERMS="${PERMS:-1}"
CALIBRATE="${CALIBRATE:-false}"
N="${N:-20}"
mkdir -p outputs/debug
LOG="outputs/debug/debug_${MODEL_KEY}_${PROMPT}_chat${CHAT}_p${PERMS}_c${CALIBRATE}_n${N}.txt"
python -X utf8 src/debug_cli.py --model-key "$MODEL_KEY" --prompt "$PROMPT" --chat "$CHAT" --perms "$PERMS" \
  --calibrate "$CALIBRATE" --n "$N" --question-index "${QUESTION_INDEX:-0}" --perm-index "${PERM_INDEX:-0}" \
  --generate "${GENERATE:-100}" --sections "${SECTIONS:-tokenization,scoring,detail,summary}" 2>&1 | tee "$LOG"
echo "log saved to $LOG"
