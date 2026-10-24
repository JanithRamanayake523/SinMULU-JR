# Running the Sinhala MMLU experiments on a GPU cluster (scripts only, no notebooks)

Everything in this folder is what the cluster needs. All paths are relative, so the folder can sit anywhere
(for example `~/sinmulu`). Eval set = public train (Easy only, never trained on). The public test set is used only to write
predictions. **Keep this folder private: the task licence forbids redistributing the test files.**

```
src/            all code (grid, scoring, token histogram, debug scripts, dedup, ...)
configs/        one YAML per model (model id, chat capability, dtype, prompt)
tests/          offline logic tests
data/           public train + public test (+ dev/ built from them)
results/        results.csv with the experiments finished so far (the grid skips them) + fertility
outputs/        per-experiment outputs finished so far
scripts/        setup_env.sh, download_models.py, run_checks.sh, run_grid.sh, run_test_predictions.sh, run_debug.sh, slurm_job.sh, env.sh
requirements-cluster.txt
```

## 1. Copy to the cluster
```bash
scp -r cluster_package  <user>@<cluster>:~/sinmulu        # or rsync -av cluster_package/ <user>@<cluster>:~/sinmulu/
ssh <user>@<cluster>
cd ~/sinmulu
```

## 2. One-time setup (login node, internet needed)
1. Edit the `module load ...` lines in `scripts/env.sh` and `scripts/setup_env.sh` for your cluster (`module avail python cuda`).
2. Create the environment and install torch + the pinned libraries:
   ```bash
   bash scripts/setup_env.sh        # follow its message if torch must be installed first (CUDA build matching `nvidia-smi`)
   ```
3. Download the models into a cache (Gemma and Llama are gated: accept their licences on the Hugging Face model pages first):
   ```bash
   export HF_TOKEN=hf_xxx           # your own token (or put it alone on one line in a file named .hf_token on the server, then chmod 600 .hf_token; scripts/env.sh reads it)
   source .venv/bin/activate
   python scripts/download_models.py                    # the three models: gemma-3-4b-it, Llama-3.2-3B-Instruct, Qwen3-4B-Instruct-2507 (~8 + 6.5 + 8 GB)
   python scripts/download_models.py --only gemma3_4b   # just one model (config name without .yaml)
   python scripts/download_models.py --all              # also the 0.6-1B models
   python scripts/download_models.py --extra google/gemma-3-1b-pt   # optionally more
   ```
   The cache goes to `./hf_cache` (override with `export HF_HOME=/path/with/space`). Compute nodes often have no internet:
   the job script sets `OFFLINE=1`, which makes them use only this cache.

## 3. Run
Quick check (no GPU): `bash scripts/run_checks.sh` (offline tests, data validation, dev set).

**With SLURM** (edit the `#SBATCH` lines in `scripts/slurm_job.sh` first: partition, memory, time):
```bash
mkdir -p logs
sbatch --export=ALL,TASK=checks scripts/slurm_job.sh
sbatch --export=ALL,TASK=grid,PROMPT=nosubject_strict scripts/slurm_job.sh
sbatch --export=ALL,TASK=grid,CONFIGS="configs/gemma3_1b.yaml configs/llama32_1b.yaml",PROMPT=nosubject scripts/slurm_job.sh
sbatch --export=ALL,TASK=debug,MODEL_KEY=qwen,CHAT=on,N=20 scripts/slurm_job.sh
```
**Without a scheduler** (interactive GPU node): run the scripts directly, for example
`PROMPT=nosubject_strict bash scripts/run_grid.sh`.

What each task does:
| Task | Script | What it produces |
|---|---|---|
| checks | `run_checks.sh` | tests, data validation, `dev/*.jsonl` |
| grid | `run_grid.sh` | token histogram, fertility, the experiment grid (`results/results.csv`, `outputs/<exp>/`), subject-level accuracy (`results/subject_accuracy/`) |
| debug | `run_debug.sh` | the debug notebook as text: tokenisation, scoring table, per-question next-token detail (`outputs/debug/*.txt`, `*.csv`) |
| first5 | `run_first5.sh` | the debug-notebook output for the **first 5 dev questions of every model** (3 models, each in its default way: chat template on): `outputs/debug/first5/*.txt` + `*.csv` |
| test_predictions | `run_test_predictions.sh` | `outputs/<exp>_test/predictions.zip` for Codabench (no diagnostics printed) |

Grid per model, in its **default way** (chat template on for instruct models; raw prompt for models without a chat template, e.g. `-pt`, `chat_capable: false` in the config):
{calibration off + 1 option order, calibration off + 4 option orders, calibration on + 1 option order} = 3 experiments per model.
The only things varied are the calibration and the permutations. `CHAT_MODES=both` would also run chat off (not needed). Finished experiments (same model, prompt, chat, perms, calibration) are skipped, so a job killed
by the time limit can simply be resubmitted.

First 5 questions of every model (the debug-notebook results):
```bash
bash scripts/run_first5.sh                         # nosubject_strict; fixed: default way (chat on), perms 1, no calibration (token generation only here)
PROMPT=nosubject MODELS="gemma" bash scripts/run_first5.sh   # other prompt / one model
sbatch --export=ALL,TASK=first5 scripts/slurm_job.sh
```

Options of the debug script (environment variables): `MODEL_KEY` (qwen|gemma|llama), `PROMPT`, `CHAT` (on|off), `PERMS` (1|4),
`CALIBRATE` (true|false), `N`, `QUESTION_INDEX`, `PERM_INDEX` (0-3), `GENERATE`, `SECTIONS`.

## 4. Bring the results back
Copy these back to your computer (they are small) and merge them into the project:
```bash
rsync -av <user>@<cluster>:~/sinmulu/results/  ./results_from_cluster/
rsync -av <user>@<cluster>:~/sinmulu/outputs/  ./outputs_from_cluster/
```
Tell the assistant where you put them; `results.csv` rows and `outputs/` folders are merged by experiment name, and the subject-level
files are rebuilt with `python src/subject_report.py`.

## Models (default set)
| Config | Model | Params | dtype | Notes |
|---|---|---|---|---|
| `gemma3_4b.yaml` | `google/gemma-3-4b-it` | ~4B (multimodal checkpoint, text only here) | bf16 | gated; if `AutoModelForCausalLM` cannot load it the code falls back to `Gemma3ForConditionalGeneration` |
| `llama32_3b.yaml` | `meta-llama/Llama-3.2-3B-Instruct` | 3.2B | bf16 | gated |
| `qwen3_4b_instruct_2507.yaml` | `Qwen/Qwen3-4B-Instruct-2507` | 4.0B | bf16 | no gating notice; non-thinking only, no thinking switch in its chat template |

All three are instruction-tuned with chat templates (checked on their model cards), so each runs in its default way (chat template on); only calibration and permutations are varied.
The earlier 0.6-1B configs are still there (`configs/qwen3_0.6b.yaml`, `gemma3_1b.yaml`, `llama32_1b.yaml`; use `CONFIGS=...` / `MODELS="qwen gemma llama"`).
**bf16 needs an Ampere-or-newer GPU** (A100, A6000, RTX 30xx/40xx, H100). On V100/T4 set `dtype: fp16` in the config (Gemma 3 may then overflow: use `fp32`, ~17 GB).
Memory: about 9 GB for Gemma-4B, 7 GB for Llama-3B and 9 GB for Qwen-4B in bf16, plus activations. Any GPU with 16 GB or more is enough.

## Notes
- **GPU differences:** the grid was first run on a Colab T4 (no bf16, so Gemma runs in fp32). A different GPU can give tiny
  numerical differences. Every run stores torch, transformers, CUDA and the GPU name in `outputs/<exp>/config.json`.
- **Memory:** the 0.6-1B models need about 2-5 GB of GPU memory (Gemma fp32 is the largest). Any GPU with 8 GB or more is enough.
- **Never put your Hugging Face token into a file** in this folder. Use `export HF_TOKEN=...` in the shell.
