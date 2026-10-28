# Running on RunPod

Pod: PyTorch template (CUDA 12.x, torch 2.8), a GPU with bf16 and >= 24 GB (RTX 3090 / 4090, L40, A100). First run a GPU test and `bash runpod/speedtest.sh`. Several pods had a visible GPU that refused every CUDA call ("CUDA unknown error"); `setup.sh` checks for this. A good pod downloads Python packages at >= 5 MB/s and from Hugging Face at >= 50 MB/s. Copy results off the pod before ending it, and prefer Stop to Terminate: pods without a volume lose everything.

## Final system (Qwen3.5-4B + combined-data LoRA, thinking mode)
```bash
bash runpod/setup.sh                        # GPU check, packages (incl. flash-linear-attention, causal-conv1d), tests, Qwen3.5-4B
bash runpod/train_qwen_combined_1028.sh     # LoRA on data/combined_si/competition_format_val10 (8.8 h on an RTX 3090)
bash runpod/think_1028.sh test              # merge (if needed) + cot_en_role_think on the 2,238 test questions (~5 h, A100)
```
`think_1028.sh` merges `lora_adapters/qwen3.5-4b_lora_combined-si_epoch1/adapter` if `models/` holds no merged copy. Generation is saved batch by batch, so an interrupted run resumes. `think_rest_1028.sh` reuses already generated texts. Watch a run with `runpod/watch.cmd` (Windows) or `runpod/watch.sh`; closing the viewer does not stop the run.

## Earlier experiments (Gemma-3-4B-it)
The scripts of the screening, confirmation, LoRA and grid rounds are in `runpod/experiments/` (dated). Gemma is gated: accept its licence on huggingface.co and `export HF_TOKEN=...` before `bash runpod/setup.sh gemma`.

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
bash runpod/experiments/run_subset.sh smoke     # ~5 min: every code path on 8 questions; must finish without FAILED lines
bash runpod/experiments/run_subset.sh wave1     # ~20 min
bash runpod/experiments/run_subset.sh wave2     # ~30 min
bash runpod/experiments/run_subset.sh wave3     # ~45 min
bash runpod/experiments/run_subset.sh report
```
Steps skip finished experiments, so rerunning a wave after a crash continues where it stopped.
Failed steps are listed in `logs/failed.txt`.

## 4. Results back
`bash runpod/pack.sh results` → `/workspace/sinmulu_results_<time>.tar.gz` (results.csv, outputs, logs). Download it
and unpack in the project folder on your computer.
