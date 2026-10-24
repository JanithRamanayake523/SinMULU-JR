#!/usr/bin/env bash
# Predictions for the public test set (Codabench format). Only after the dev results have decided which setting to use.
# No diagnostics are printed for the test set. Output: outputs/<EXP_ID>_test/predictions.zip
#   CONFIG=configs/gemma3_4b.yaml PROMPT=nosubject_strict CHAT=on EXP_ID=gemma3_4b_chaton bash scripts/run_test_predictions.sh
# Optional: PERMS=4, CALIBRATE=true
source "$(dirname "$0")/env.sh"
: "${CONFIG:?set CONFIG, e.g. configs/gemma3_4b.yaml}"
: "${EXP_ID:?set EXP_ID, e.g. gemma3_4b_chaton}"
PROMPT="${PROMPT:-nosubject_strict}"
CHAT="${CHAT:-auto}"
PERMS="${PERMS:-1}"
CALIBRATE="${CALIBRATE:-false}"
python -X utf8 src/eval_logprob.py --config "$CONFIG" --data test --prompt "$PROMPT" --chat "$CHAT" \
  --perms "$PERMS" --calibrate "$CALIBRATE" --exp-id "$EXP_ID"
