#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH -p nvidia
#SBATCH --mem=64G
#SBATCH -t 3-23:59:59
#SBATCH -o /scratch/mk8737/farah/Adaptation/tmp_logs/job_%j.out
#SBATCH -e /scratch/mk8737/farah/Adaptation/tmp_logs/job_%j.err

# make sure '/scratch/mk8737/farah/Adaptation/tmp_logs/' already exists before running this script; slurm won't create it automatically

set -euo pipefail

START_TS=$(date +%s)
echo "START: $(date -Is)"

log_job_timing() {
  local exit_code=$?
  local end_ts
  local elapsed_sec
  local elapsed_h
  local elapsed_m
  local elapsed_s
  end_ts=$(date +%s)
  elapsed_sec=$((end_ts - START_TS))
  elapsed_h=$((elapsed_sec / 3600))
  elapsed_m=$(((elapsed_sec % 3600) / 60))
  elapsed_s=$((elapsed_sec % 60))
  echo "END: $(date -Is)"
  printf 'ELAPSED_HMS=%02d:%02d:%02d\n' "$elapsed_h" "$elapsed_m" "$elapsed_s"
  echo "EXIT_CODE=${exit_code}"
}
trap log_job_timing EXIT

if [[ $# -lt 3 ]]; then
  echo "Usage: sbatch jobs/hpc_submit_eval_cli.sh <TASK_NUM> <MODEL_TYPE> <DATASET>"
  echo "Example: sbatch jobs/hpc_submit_eval_cli.sh 1 mistral_7b test"
  exit 1
fi

TASK_NUM="$1"
MODEL_TYPE="$2"
DATASET="$3"

PROJECT_ROOT="/scratch/mk8737/farah/Adaptation"
CONFIG_PATH="${PROJECT_ROOT}/configs/task${TASK_NUM}/${MODEL_TYPE}_${DATASET}.yaml"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Error: Config not found: $CONFIG_PATH"
  exit 1
fi

module purge
module load cuda/11.8.0

# Make sure Conda is initialised for non-interactive shells
set +u
eval "$(/share/apps/NYUAD5/miniconda/3-4.11.0/bin/conda shell.bash hook)"
conda activate adaptation
set -u

# Copy outputs into designated directory
LOGS_DIR=${PROJECT_ROOT}/logs/task${TASK_NUM}/${MODEL_TYPE}_${DATASET}_${SLURM_JOB_ID}
mkdir -p "$LOGS_DIR"

# Main Command
cd "$PROJECT_ROOT"
if [[ -f ".env" ]]; then
  set -a
  source .env
  set +a
fi
export HF_HUB_ENABLE_HF_TRANSFER=1
python scripts/run_evaluation.py "configs/task${TASK_NUM}/${MODEL_TYPE}_${DATASET}.yaml"

# Task 2 additional step: run judge LLM on the generated predictions
if [[ "$TASK_NUM" == "2" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Error: OPENAI_API_KEY is not set. Add it to .env or export it before sbatch."
    exit 1
  fi
  python scripts/run_judge.py "configs/task${TASK_NUM}/${MODEL_TYPE}_${DATASET}.yaml"
fi

# Cleanup: Move logs to the designated directory
mv "${PROJECT_ROOT}/tmp_logs/job_${SLURM_JOB_ID}.out" "$LOGS_DIR/job_${SLURM_JOB_ID}.out"
mv "${PROJECT_ROOT}/tmp_logs/job_${SLURM_JOB_ID}.err" "$LOGS_DIR/job_${SLURM_JOB_ID}.err"
