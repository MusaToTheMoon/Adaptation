#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p nvidia
#SBATCH --mem=64G
#SBATCH -t 2-0:59:59
#SBATCH -o /scratch/mk8737/farah/Adaptation/tmp_logs/job_%j.out
#SBATCH -e /scratch/mk8737/farah/Adaptation/tmp_logs/job_%j.err

# make sure '/scratch/mk8737/farah/Adaptation/tmp_logs/' already exists before running this script; slurm won't create it automatically

set -euo pipefail

# #SBATCH --constrain=80g
# #SBATCH --gres=gpu:a100:1
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

if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch jobs/hpc_submit_recompute_bertscore_cli.sh <TASK_NUM> <DATASET> [LANG] [WRITE_MODE] [COPY_SUFFIX]"
  echo "  TASK_NUM: 2 or 3"
  echo "  WRITE_MODE: copy (default) | in-place"
  echo "  COPY_SUFFIX: suffix for copied CSV names (default: _recomputed)"
  echo "Example (default copy): sbatch jobs/hpc_submit_recompute_bertscore_cli.sh 2 medarabenchv2 ar"
  echo "Example (task3): sbatch jobs/hpc_submit_recompute_bertscore_cli.sh 3 medarabenchv2 ar"
  echo "Example (in-place): sbatch jobs/hpc_submit_recompute_bertscore_cli.sh 2 medarabenchv2 ar in-place"
  exit 1
fi

TASK_NUM="$1"
DATASET="$2"
LANG="${3:-ar}"
WRITE_MODE="${4:-copy}"
COPY_SUFFIX="${5:-_recomputed}"

PROJECT_ROOT="/scratch/mk8737/farah/Adaptation"
PREDICTIONS_DIR="${PROJECT_ROOT}/results/predictions/task${TASK_NUM}/${DATASET}"

if [[ "$TASK_NUM" != "2" && "$TASK_NUM" != "3" ]]; then
  echo "Error: TASK_NUM must be 2 or 3."
  exit 1
fi

if [[ ! -d "$PREDICTIONS_DIR" ]]; then
  echo "Error: Predictions directory not found: $PREDICTIONS_DIR"
  exit 1
fi

if ! compgen -G "${PREDICTIONS_DIR}/*.csv" > /dev/null; then
  echo "Error: No CSV files found under: $PREDICTIONS_DIR"
  exit 1
fi

if [[ "$WRITE_MODE" != "copy" && "$WRITE_MODE" != "in-place" ]]; then
  echo "Error: WRITE_MODE must be one of: copy, in-place"
  exit 1
fi

RECOMPUTE_EXTRA_ARGS=()
if [[ "$WRITE_MODE" == "in-place" ]]; then
  RECOMPUTE_EXTRA_ARGS+=("--in-place")
else
  RECOMPUTE_EXTRA_ARGS+=("--copy-suffix" "$COPY_SUFFIX")
fi

module purge
module load cuda/11.8.0

# Make sure Conda is initialised for non-interactive shells
set +u
eval "$(/share/apps/NYUAD5/miniconda/3-4.11.0/bin/conda shell.bash hook)"
conda activate adaptation
set -u

# Copy outputs into designated directory
LOGS_DIR=${PROJECT_ROOT}/logs/task${TASK_NUM}/recompute_bertscore_${DATASET}_${SLURM_JOB_ID}
mkdir -p "$LOGS_DIR"

# Main Command
cd "$PROJECT_ROOT"
if [[ -f ".env" ]]; then
  set -a
  source .env
  set +a
fi
export HF_HUB_ENABLE_HF_TRANSFER=1
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true

python scripts/recompute_bertscore_arabert.py \
  --inputs "$PREDICTIONS_DIR" \
  --recursive \
  --lang "$LANG" \
  --device auto \
  "${RECOMPUTE_EXTRA_ARGS[@]}"

# Keep tmp_logs files and copy them into the designated directory
cp "${PROJECT_ROOT}/tmp_logs/job_${SLURM_JOB_ID}.out" "$LOGS_DIR/job_${SLURM_JOB_ID}.out"
cp "${PROJECT_ROOT}/tmp_logs/job_${SLURM_JOB_ID}.err" "$LOGS_DIR/job_${SLURM_JOB_ID}.err"
