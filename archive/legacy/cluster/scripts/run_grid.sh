#!/usr/bin/env bash
# Token-length histogram + experiment grid + subject-level accuracy, for the chosen models and prompt.
# Finished experiments are skipped (read from results/results.csv), so this can be resubmitted after a time-limit kill.
#
#   bash scripts/run_grid.sh                                              # gemma-3-4b-it, Llama-3.2-3B-Instruct, Qwen3-4B-Instruct-2507, prompt nosubject_strict
#   CONFIGS="configs/gemma3_4b.yaml" PROMPT=nosubject bash scripts/run_grid.sh
#   CHAT_MODES=both bash scripts/run_grid.sh                              # also chat off for instruct models (not needed by default)
# Variables: CONFIGS (space-separated config files), PROMPT (nosubject | nosubject_strict | subject_si | subject_en),
#            CHAT_MODES (default | both), OFFLINE=1 (use the HF cache only)
source "$(dirname "$0")/env.sh"
CONFIGS="${CONFIGS:-configs/gemma3_4b.yaml configs/llama32_3b.yaml configs/qwen3_4b_instruct_2507.yaml}"
PROMPT="${PROMPT:-nosubject_strict}"
CHAT_MODES="${CHAT_MODES:-default}"

nvidia-smi || true
python -X utf8 tests/test_eval_core.py
python -X utf8 src/build_dev.py
python -X utf8 src/token_hist.py --configs $CONFIGS --prompt "$PROMPT"
python -X utf8 src/run_grid.py --dry-run --configs $CONFIGS --prompt "$PROMPT" --chat-modes "$CHAT_MODES"
python -X utf8 src/run_grid.py --configs $CONFIGS --prompt "$PROMPT" --chat-modes "$CHAT_MODES"
python -X utf8 src/subject_report.py
