#!/usr/bin/env bash
# Offline sanity checks + rebuild of the dev set. No GPU needed.
#   bash scripts/run_checks.sh
source "$(dirname "$0")/env.sh"
python -X utf8 tests/test_eval_core.py
python -X utf8 src/validate_data.py
python -X utf8 src/build_dev.py
