#!/usr/bin/env bash
# Network check for a new pod, BEFORE installing or downloading anything (writes nothing outside /tmp).
#   bash runpod/speedtest.sh
# Good pod: Python packages >= 5 MB/s and Hugging Face >= 50 MB/s (Gemma-3-4B-it, 8.6 GB, then downloads in ~3 min).
set -u
T=$(mktemp -d)
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
python -c "import torch; torch.ones(1, device='cuda'); torch.cuda.synchronize(); print('GPU compute: ok')" 2>/dev/null     || echo "GPU compute: FAILED (the GPU is visible but refuses work: replace the pod)"
echo "Disk: $(df -h /workspace | tail -1 | awk '{print $4 " free on /workspace"}'), $(df -h / | tail -1 | awk '{print $4 " free on /"}')"
curl -sS -o /dev/null -w "PyPI index:       %{http_code} in %{time_total}s\n" https://pypi.org/simple/numpy/
start=$(date +%s.%N)
if timeout 90 pip download -q --no-deps --progress-bar off -d "$T" numpy==2.1.3 >/dev/null 2>&1; then
    end=$(date +%s.%N)
    bytes=$(du -sb "$T" | cut -f1)
    echo "Python package:   $(awk -v b="$bytes" -v s="$start" -v e="$end" 'BEGIN{printf "%.1f MB/s (%.0f MB in %.1fs)", b/(e-s)/1e6, b/1e6, e-s}')"
else
    echo "Python package:   FAILED or slower than 16 MB in 90 s"
fi
curl -sS -L -r 0-199999999 -o /dev/null --max-time 90 -w "%{speed_download} %{size_download} %{time_total}\n" \
    https://huggingface.co/google-bert/bert-base-uncased/resolve/main/model.safetensors \
    | awk '{printf "Hugging Face:     %.1f MB/s (%.0f MB in %.1fs)\n", $1/1e6, $2/1e6, $3}'
rm -rf "$T"
