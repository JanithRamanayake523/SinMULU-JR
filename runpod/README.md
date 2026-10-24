# Running on RunPod (RTX 3090, 24 GB; an A40 works too)

Pod: PyTorch template (CUDA 12.x, torch ≥ 2.6), RTX 3090 (Ampere: bf16 works), code and results on the `/workspace` volume; the model cache (~34 GB) on the container disk at `/root/hf_cache` (re-downloaded after a pod stop).
Before starting: accept the Gemma licence on huggingface.co (gemma-3-4b-it, gemma-3-4b-pt, gemma-3-1b-it) with the
account whose token you use.

## 1. Code to the pod
On your computer: `bash runpod/pack.sh code` → `../sinmulu_code.tar.gz` (code, data, git history; keep it private).
Upload it (RunPod web terminal upload, `runpodctl send`, or scp) and unpack:
```bash
mkdir -p /workspace/SinMULU-JR && tar -xzf sinmulu_code.tar.gz -C /workspace/SinMULU-JR && cd /workspace/SinMULU-JR
```

## 2. Setup (once per pod)
```bash
export HF_TOKEN=hf_...          # shell only, never in a file
bash runpod/setup.sh            # GPU check, pip install, dev files, tests, model downloads (~10-15 min)
```

## 3. Runs (inside tmux, so a closed browser tab does not kill them)
```bash
tmux new -s run
bash runpod/run_subset.sh smoke     # ~5 min: every code path on 8 questions; must finish without FAILED lines
bash runpod/run_subset.sh wave1     # ~20 min
bash runpod/run_subset.sh wave2     # ~30 min
bash runpod/run_subset.sh wave3     # ~45 min
bash runpod/run_subset.sh report
```
Steps skip finished experiments, so rerunning a wave after a crash continues where it stopped.
Failed steps are listed in `logs/failed.txt`.

## 4. Results back
`bash runpod/pack.sh results` → `/workspace/sinmulu_results_<time>.tar.gz` (results.csv, outputs, logs). Download it
and unpack in the project folder on your computer.
