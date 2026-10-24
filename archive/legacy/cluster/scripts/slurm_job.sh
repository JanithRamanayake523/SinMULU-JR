#!/usr/bin/env bash
#SBATCH --job-name=sinmulu
#SBATCH --partition=gpu            # EDIT: your GPU partition / queue name
#SBATCH --gres=gpu:1               # one GPU is enough for the 0.6-1B models
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00            # EDIT: the grid is resumable, so a shorter limit is fine
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Submit ONE of the tasks (create the logs folder first: mkdir -p logs):
#   sbatch --export=ALL,TASK=checks scripts/slurm_job.sh
#   sbatch --export=ALL,TASK=grid,PROMPT=nosubject_strict scripts/slurm_job.sh
#   sbatch --export=ALL,TASK=grid,CONFIGS="configs/gemma3_4b.yaml",PROMPT=nosubject scripts/slurm_job.sh
#   sbatch --export=ALL,TASK=debug,MODEL_KEY=qwen4b,CHAT=on,N=20 scripts/slurm_job.sh
#   sbatch --export=ALL,TASK=first5 scripts/slurm_job.sh
#   sbatch --export=ALL,TASK=test_predictions,CONFIG=configs/gemma3_4b.yaml,EXP_ID=gemma3_4b_chaton scripts/slurm_job.sh
# If the cluster does not use SLURM, the scripts in this folder run as they are (bash scripts/run_grid.sh); tell me the
# scheduler (PBS/LSF/...) and I will write the matching header.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export OFFLINE=1                   # compute nodes usually have no internet: models come from the cache (download_models.py)
case "${TASK:?set TASK=checks|grid|debug|first5|test_predictions}" in
  checks)           bash scripts/run_checks.sh ;;
  grid)             bash scripts/run_grid.sh ;;
  debug)            bash scripts/run_debug.sh ;;
  first5)          bash scripts/run_first5.sh ;;
  test_predictions) bash scripts/run_test_predictions.sh ;;
  *) echo "unknown TASK"; exit 1 ;;
esac
