#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --constrain=80g
#SBATCH --mem=128G
#SBATCH -p nvidia
#SBATCH -t 1-0:59:59
#SBATCH -o /scratch/mk8737/farah/Adaptation/tmp_logs/job_%j.out
#SBATCH -e /scratch/mk8737/farah/Adaptation/tmp_logs/job_%j.err

# make sure '/scratch/mk8737/farah/Adaptation/tmp_logs/' already exists before running this script; slurm won't create it automatically

set -euo pipefail

# SBATCH --constrain=80g

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

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch jobs/hpc_submit_align_search_cli.sh <WINDOW> [--resume]"
  echo "WINDOW: l1_34 | l1_40 | both"
  exit 1
fi

WINDOW="$1"
shift

RESUME_FLAG=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)
      RESUME_FLAG=(--resume)
      shift
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

PROJECT_ROOT="/scratch/mk8737/farah/Adaptation"

TRAIN_FILE="${PROJECT_ROOT}/datasets/train/splits/train_task1_translated.json"
VAL_FILE="${PROJECT_ROOT}/datasets/train/splits/val_task1_translated.json"
if [[ ! -f "$TRAIN_FILE" ]]; then
  TRAIN_FILE="${PROJECT_ROOT}/datasets/train/train_task1_translated.json"
fi
if [[ ! -f "$VAL_FILE" ]]; then
  VAL_FILE="${PROJECT_ROOT}/datasets/train/val_task1_translated.json"
fi

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "Error: Train file not found: $TRAIN_FILE"
  exit 1
fi
if [[ ! -f "$VAL_FILE" ]]; then
  echo "Error: Val file not found: $VAL_FILE"
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
LOGS_DIR=${PROJECT_ROOT}/logs/train/lr_search_${WINDOW}_${SLURM_JOB_ID}
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

run_search() {
  local layer_end="$1"
  local search_dir="$2"

  python models/run_align_search.py \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --search_dir "$search_dir" \
    --layer_start 0 --layer_end "$layer_end" --probe_layers 34 \
    --auto_beta --n_trials 5 --nproc 1 --num_epochs 10 --seed 42 \
    "${RESUME_FLAG[@]}"
}

case "$WINDOW" in
  l1_34)
    run_search 33 "${PROJECT_ROOT}/outputs/lr_search_l1_34_probe34_kl"
    ;;
  l1_40)
    run_search 39 "${PROJECT_ROOT}/outputs/lr_search_l1_40_probe34_kl"
    ;;
  both)
    run_search 33 "${PROJECT_ROOT}/outputs/lr_search_l1_34_probe34_kl"
    run_search 39 "${PROJECT_ROOT}/outputs/lr_search_l1_40_probe34_kl"
    ;;
  *)
    echo "Error: Unknown WINDOW '$WINDOW' (use l1_34, l1_40, or both)"
    exit 1
    ;;
 esac

# Cleanup: Move logs to the designated directory
mv "${PROJECT_ROOT}/tmp_logs/job_${SLURM_JOB_ID}.out" "$LOGS_DIR/job_${SLURM_JOB_ID}.out"
mv "${PROJECT_ROOT}/tmp_logs/job_${SLURM_JOB_ID}.err" "$LOGS_DIR/job_${SLURM_JOB_ID}.err"
