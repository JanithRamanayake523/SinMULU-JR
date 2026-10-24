#!/usr/bin/env bash
# Bundle files for transfer (no model weights, no secrets).
#   bash runpod/pack.sh code      # on your computer: code + data + git history -> ../sinmulu_code.tar.gz (upload to the pod)
#   bash runpod/pack.sh results   # on the pod: results, outputs, logs, translation metadata -> /workspace/sinmulu_results_<time>.tar.gz
# The archives contain benchmark questions or derived files: keep them private (task licence).
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:?usage: pack.sh code|results}" in
code)
    tar --exclude=./archive/legacy/cluster_package --exclude='*.zip' --exclude='__pycache__' --exclude=./outputs \
        --exclude=./cache --exclude=./logs -czf ../sinmulu_code.tar.gz .
    echo "wrote $(cd .. && pwd)/sinmulu_code.tar.gz" ;;
results)
    out="/workspace/sinmulu_results_$(date +%Y%m%d_%H%M).tar.gz"
    tar -czf "$out" results logs outputs $(ls cache/translations/*.meta.json 2>/dev/null)
    echo "wrote $out" ;;
esac
