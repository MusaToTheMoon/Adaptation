#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:2
#SBATCH --constrain=80g
#SBATCH --mem=128G
#SBATCH -t 3-00:00:00
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

PROJECT_ROOT="/scratch/mk8737/farah/Adaptation"
HF_CACHE="/scratch/mk8737/huggingface"
MEDGEMMA_REPO="${HF_CACHE}/models--google--medgemma-27b-text-it"
MEDGEMMA_REF_FILE="${MEDGEMMA_REPO}/refs/main"

TRAIN_FILE="${PROJECT_ROOT}/datasets/train/splits/train_task1_translated.json"
VAL_FILE="${PROJECT_ROOT}/datasets/train/splits/val_task1_translated.json"
SEARCH_DIR="${PROJECT_ROOT}/outputs/medgemma_lr_search_w4_l50_62_probe50_kl"
TRAIN_SCRIPT="${PROJECT_ROOT}/models/train_lora_align_medgemma.py"

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "Error: Train file not found: $TRAIN_FILE"
  exit 1
fi

if [[ ! -f "$VAL_FILE" ]]; then
  echo "Error: Validation file not found: $VAL_FILE"
  exit 1
fi

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "Error: MedGemma training script not found: $TRAIN_SCRIPT"
  exit 1
fi

if [[ ! -f "$MEDGEMMA_REF_FILE" ]]; then
  echo "Error: MedGemma ref file not found: $MEDGEMMA_REF_FILE"
  exit 1
fi

MEDGEMMA_SNAPSHOT="$(cat "$MEDGEMMA_REF_FILE")"
MEDGEMMA="${MEDGEMMA_REPO}/snapshots/${MEDGEMMA_SNAPSHOT}"

if [[ ! -d "$MEDGEMMA" ]]; then
  echo "Error: MedGemma model snapshot not found: $MEDGEMMA"
  exit 1
fi

module purge
module load cuda/11.8.0

set +u
eval "$(/share/apps/NYUAD5/miniconda/3-4.11.0/bin/conda shell.bash hook)"
conda activate adaptation
set -u

LOGS_DIR="${PROJECT_ROOT}/logs/train/medgemma_lr_search_w4_l50_62_${SLURM_JOB_ID}"
mkdir -p "$LOGS_DIR"

cd "$PROJECT_ROOT"
if [[ -f ".env" ]]; then
  set -a
  source .env
  set +a
fi
export HF_HOME="$HF_CACHE"
export HF_HUB_OFFLINE=1
export HF_HUB_ENABLE_HF_TRANSFER=1
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "HF_HOME=${HF_HOME}"
echo "MEDGEMMA=${MEDGEMMA}"
echo "TRAIN_FILE=${TRAIN_FILE}"
echo "VAL_FILE=${VAL_FILE}"
echo "SEARCH_DIR=${SEARCH_DIR}"
nvidia-smi -L || true

run_step() {
  local step_name="$1"
  shift
  echo
  echo "===== ${step_name} ====="
  echo "CMD: $*"
  "$@"
}

run_step "MedGemma W4 L50-L62 LR search" \
  python models/run_align_search.py \
    --train_script "$TRAIN_SCRIPT" \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --search_dir "$SEARCH_DIR" \
    --model_name "$MEDGEMMA" \
    --layer_start 49 \
    --layer_end 61 \
    --probe_layers 50 \
    --auto_beta \
    --n_trials 5 \
    --nproc 2 \
    --num_epochs 10 \
    --seed 42

mv "${PROJECT_ROOT}/tmp_logs/job_${SLURM_JOB_ID}.out" "$LOGS_DIR/job_${SLURM_JOB_ID}.out"
mv "${PROJECT_ROOT}/tmp_logs/job_${SLURM_JOB_ID}.err" "$LOGS_DIR/job_${SLURM_JOB_ID}.err"
