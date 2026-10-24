#!/usr/bin/env bash
# The debug-notebook results (tokenisation, scoring table, per-question next-token detail) for the FIRST 5 DEV QUESTIONS of
# every model. Dev data only. (No summary section.)
#   bash scripts/run_first5.sh
#   PROMPT=nosubject MODELS="gemma4b" bash scripts/run_first5.sh
# Fixed settings, for every model: its DEFAULT way (chat template on for these instruct models), ONE option order (perms = 1),
# NO calibration. The greedy continuation (token generation) is produced in this setting.
# Variables: MODELS (default "gemma4b llama3b qwen4b"; the 1B models: qwen gemma llama), PROMPT (default nosubject_strict), N (default 5),
#            GENERATE (tokens of greedy continuation per question, default 100)
# Output: outputs/debug/first5/<model>_<prompt>_chat<on>_p1_c0_n5.txt (+ .csv scoring table),
#         and outputs/debug/first5/tokenizers_compare.txt (one question through the three tokenizers).
source "$(dirname "$0")/env.sh"
MODELS="${MODELS:-gemma4b llama3b qwen4b}"
CHAT_LIST="auto"   # the model's default way: chat on if chat capable (all three models are)
PROMPT="${PROMPT:-nosubject_strict}"
PERMS=1
CALIBRATE=false
N="${N:-5}"
OUT="outputs/debug/first5"
mkdir -p "$OUT"

for m in $MODELS; do
  for c in $CHAT_LIST; do
    LOG="$OUT/${m}_${PROMPT}_chat${c}_p${PERMS}_c${CALIBRATE}_n${N}.txt"
    echo "=== $m chat=$c prompt=$PROMPT perms=$PERMS calibrate=$CALIBRATE n=$N -> $LOG"
    python -X utf8 src/debug_cli.py --model-key "$m" --prompt "$PROMPT" --chat "$c" --perms "$PERMS" \
      --calibrate "$CALIBRATE" --n "$N" --generate "${GENERATE:-100}" --sections tokenization,scoring,detail --no-compare \
      > "$LOG" 2>&1 || echo "FAILED: $m chat=$c (see $LOG)"
  done
done
python -X utf8 src/debug_cli.py --compare-only > "$OUT/tokenizers_compare.txt" 2>&1 || echo "tokenizer comparison failed (see $OUT/tokenizers_compare.txt)"
echo "done: $OUT"
ls -l "$OUT"
