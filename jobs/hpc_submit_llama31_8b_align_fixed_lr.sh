#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -p nvidia
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH -t 1-00:00:00
#SBATCH --exclude=cn270  # faulty GPUs (illegal memory access, jobs 16603420/22/23) — remove once admins confirm fixed
#SBATCH -o /scratch/mk8737/farah/Adaptation/tmp_logs/job_%j.out
#SBATCH -e /scratch/mk8737/farah/Adaptation/tmp_logs/job_%j.err

# Default: train the remaining non-sweep windows w1-w4 with the best w5 LR.
# Usage:
#   sbatch jobs/hpc_submit_llama31_8b_align_fixed_lr.sh
#   sbatch jobs/hpc_submit_llama31_8b_align_fixed_lr.sh w2
#   sbatch jobs/hpc_submit_llama31_8b_align_fixed_lr.sh --lr 1.057792e-04 w1 w3
#   sbatch jobs/hpc_submit_llama31_8b_align_fixed_lr.sh all

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

BEST_LR="1.057792e-04"
RESUME_TRAIN_FLAG=()
WINDOWS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lr)
      if [[ $# -lt 2 ]]; then
        echo "Error: --lr requires a value"
        exit 2
      fi
      BEST_LR="$2"
      shift 2
      ;;
    --resume)
      RESUME_TRAIN_FLAG=(--resume_from_checkpoint auto)
      shift
      ;;
    w1|w2|w3|w4|w5|all|remaining)
      WINDOWS+=("$1")
      shift
      ;;
    *)
      echo "Unknown arg: $1"
      echo "Usage: sbatch jobs/hpc_submit_llama31_8b_align_fixed_lr.sh [--lr LR] [--resume] [w1|w2|w3|w4|w5|all|remaining ...]"
      exit 2
      ;;
  esac
done

if [[ ${#WINDOWS[@]} -eq 0 ]]; then
  WINDOWS=(w1 w2 w3 w4)
fi

EXPANDED_WINDOWS=()
for window in "${WINDOWS[@]}"; do
  case "$window" in
    remaining)
      EXPANDED_WINDOWS+=(w1 w2 w3 w4)
      ;;
    all)
      EXPANDED_WINDOWS+=(w1 w2 w3 w4 w5)
      ;;
    *)
      EXPANDED_WINDOWS+=("$window")
      ;;
  esac
done

PROJECT_ROOT="/scratch/mk8737/farah/Adaptation"
HF_CACHE="/scratch/mk8737/huggingface"
LLAMA_REPO="${HF_CACHE}/models--meta-llama--Llama-3.1-8B-Instruct"
LLAMA_REF_FILE="${LLAMA_REPO}/refs/main"

TRAIN_FILE="${PROJECT_ROOT}/datasets/train/splits/train_task1_translated.json"
VAL_FILE="${PROJECT_ROOT}/datasets/train/splits/val_task1_translated.json"
TRAIN_SCRIPT="${PROJECT_ROOT}/models/train_lora_align_llama31_8b.py"
W5_BEST_ADAPTER="${PROJECT_ROOT}/outputs/llama31_8b_lr_search_w5_l1_32_probe28_kl/trial_00_lr1.06e-04"

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "Error: Train file not found: $TRAIN_FILE"
  exit 1
fi

if [[ ! -f "$VAL_FILE" ]]; then
  echo "Error: Validation file not found: $VAL_FILE"
  exit 1
fi

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "Error: Llama training script not found: $TRAIN_SCRIPT"
  exit 1
fi

if [[ ! -f "$LLAMA_REF_FILE" ]]; then
  echo "Error: Llama ref file not found: $LLAMA_REF_FILE"
  exit 1
fi

LLAMA_SNAPSHOT="$(cat "$LLAMA_REF_FILE")"
LLAMA="${LLAMA_REPO}/snapshots/${LLAMA_SNAPSHOT}"

if [[ ! -d "$LLAMA" ]]; then
  echo "Error: Llama model snapshot not found: $LLAMA"
  exit 1
fi

module purge
module load cuda/11.8.0

set +u
eval "$(/share/apps/NYUAD5/miniconda/3-4.11.0/bin/conda shell.bash hook)"
conda activate adaptation
set -u

LOGS_DIR="${PROJECT_ROOT}/logs/train/llama31_8b_fixed_lr_${SLURM_JOB_ID:-local}"
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
echo "LLAMA=${LLAMA}"
echo "TRAIN_FILE=${TRAIN_FILE}"
echo "VAL_FILE=${VAL_FILE}"
echo "BEST_LR=${BEST_LR}"
echo "WINDOWS=${EXPANDED_WINDOWS[*]}"
echo "W5_BEST_ADAPTER=${W5_BEST_ADAPTER}"
nvidia-smi -L || true

run_step() {
  local step_name="$1"
  shift
  echo
  echo "===== ${step_name} ====="
  echo "CMD: $*"
  "$@"
}

run_fixed_window() {
  local window="$1"
  local paper_window="$2"
  local layer_start="$3"
  local layer_end="$4"
  local output_dir="${PROJECT_ROOT}/outputs/llama31_8b_lr_reuse_${window}_${paper_window}_probe28_kl/lr${BEST_LR}"
  local master_port=$((20000 + RANDOM % 20000))

  run_step "Llama-3.1-8B ${window} ${paper_window} fixed LR ${BEST_LR}" \
    torchrun \
      --nproc_per_node=2 \
      --master_port="$master_port" \
      "$TRAIN_SCRIPT" \
      --train_file "$TRAIN_FILE" \
      --val_file "$VAL_FILE" \
      --model_name "$LLAMA" \
      --output_dir "$output_dir" \
      --layer_start "$layer_start" \
      --layer_end "$layer_end" \
      --probe_layers 28 \
      --alpha 1.0 \
      --auto_beta \
      --num_train_epochs 10 \
      --learning_rate "$BEST_LR" \
      --early_stopping_patience 1 \
      --save_total_limit 1 \
      --max_length 1024 \
      "${RESUME_TRAIN_FLAG[@]}"
}

for window in "${EXPANDED_WINDOWS[@]}"; do
  case "$window" in
    w1)
      run_fixed_window "w1" "l1_18" 0 17
      ;;
    w2)
      run_fixed_window "w2" "l18_32" 17 31
      ;;
    w3)
      run_fixed_window "w3" "l1_28" 0 27
      ;;
    w4)
      run_fixed_window "w4" "l28_32" 27 31
      ;;
    w5)
      echo "Note: w5 best sweep adapter already exists at: ${W5_BEST_ADAPTER}"
      echo "Rerunning w5 only because it was explicitly requested."
      run_fixed_window "w5" "l1_32" 0 31
      ;;
    *)
      echo "Error: unsupported window after expansion: $window"
      exit 2
      ;;
  esac
done

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  mv "${PROJECT_ROOT}/tmp_logs/job_${SLURM_JOB_ID}.out" "$LOGS_DIR/job_${SLURM_JOB_ID}.out"
  mv "${PROJECT_ROOT}/tmp_logs/job_${SLURM_JOB_ID}.err" "$LOGS_DIR/job_${SLURM_JOB_ID}.err"
fi
