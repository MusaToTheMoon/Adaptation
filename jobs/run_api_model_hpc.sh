#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 23:00:00
#SBATCH -o /scratch/mk8737/farah/Adaptation/tmp_logs/job_%j.out
#SBATCH -e /scratch/mk8737/farah/Adaptation/tmp_logs/job_%j.err

set -euo pipefail

PROJECT_ROOT="/scratch/mk8737/farah/Adaptation"
TASK_NUM="${1:-1}"
MODEL_TYPE="${2:-gpt52}"
DATASET="${3:-test}"
CONFIG_PATH="${PROJECT_ROOT}/configs/task${TASK_NUM}/${MODEL_TYPE}_${DATASET}.yaml"

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

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Error: Config not found: $CONFIG_PATH"
  echo "Usage: sbatch jobs/run_api_model_hpc.sh [task_num] [model_type] [dataset]"
  echo "Example: sbatch jobs/run_api_model_hpc.sh 1 gpt52 test"
  exit 1
fi

cd "$PROJECT_ROOT"

# Make sure Conda is initialised for non-interactive shells
set +u
eval "$(/share/apps/NYUAD5/miniconda/3-4.11.0/bin/conda shell.bash hook)"
conda activate adaptation
set -u

if [[ -f ".env" ]]; then
  set -a
  source .env
  set +a
fi

# if [[ -z "${OPENAI_API_KEY:-}" ]]; then
#   echo "Error: OPENAI_API_KEY is not set."
#   echo "Set it in shell or in ${PROJECT_ROOT}/.env"
#   exit 1
# fi

# Copy outputs into designated directory
LOGS_DIR=${PROJECT_ROOT}/logs/task${TASK_NUM}/${MODEL_TYPE}_${DATASET}_${SLURM_JOB_ID}
mkdir -p "$LOGS_DIR"

python scripts/run_evaluation.py "configs/task${TASK_NUM}/${MODEL_TYPE}_${DATASET}.yaml"

# Cleanup: Move logs to the designated directory
mv "${PROJECT_ROOT}/tmp_logs/job_${SLURM_JOB_ID}.out" "$LOGS_DIR/job_${SLURM_JOB_ID}.out"
mv "${PROJECT_ROOT}/tmp_logs/job_${SLURM_JOB_ID}.err" "$LOGS_DIR/job_${SLURM_JOB_ID}.err"
