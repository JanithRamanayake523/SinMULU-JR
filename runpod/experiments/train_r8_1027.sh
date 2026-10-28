#!/usr/bin/env bash
# Adapter C (2026-10-27): sinmulu_si (10% validation split), rank 8, alpha 16, dropout 0.1, fresh, 1 epoch. Waits for a running train_lora first.
#   bash runpod/experiments/train_r8_1027.sh
cd "$(dirname "$0")/../.."
export HF_HOME="${SINMULU_HF_HOME:-/workspace/hf_cache}" PYTHONUNBUFFERED=1
[ -f data/sinmulu_si/competition_format_val10/train.json ] || python -m sinmmlu.sinmulu --val-fraction 0.10
while pgrep -f "sinmmlu.train_lora" >/dev/null; do sleep 30; done
echo "=== start rank-8 training $(date +%H:%M)"
python -m sinmmlu.train_lora --model configs/models/gemma3_4b_it.yaml \
    --train data/sinmulu_si/competition_format_val10/train.json --val data/sinmulu_si/competition_format_val10/val.json \
    --rank 8 --alpha 16 --dropout 0.1 --out models/gemma3_4b_it_lora_sinmulu_r8 2>&1 | tee logs/train_sinmulu_r8.log
