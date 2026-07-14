#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -p nvidia
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH -t 3-00:00:00
#SBATCH -o /scratch/mk8737/farah/Adaptation/tmp_logs/job_%j.out
#SBATCH -e /scratch/mk8737/farah/Adaptation/tmp_logs/job_%j.err

# make sure '/scratch/mk8737/farah/Adaptation/tmp_logs/' already exists before running this script; slurm won't create it automatically

set -euo pipefail

START_TS=$(date +%s)
echo "START: $(date -Is)"
#SBATCH --constrain=80g

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

RESUME_FLAG=()
RESUME_TRAIN_FLAG=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)
      RESUME_FLAG=(--resume)
      RESUME_TRAIN_FLAG=(--resume_from_checkpoint auto)
      shift
      ;;
    *)
      echo "Unknown arg: $1"
      echo "Usage: sbatch jobs/hpc_submit_llama31_8b_align_search_w5.sh [--resume]"
      exit 2
      ;;
  esac
done

PROJECT_ROOT="/scratch/mk8737/farah/Adaptation"
HF_CACHE="/scratch/mk8737/huggingface"
LLAMA_REPO="${HF_CACHE}/models--meta-llama--Llama-3.1-8B-Instruct"
LLAMA_REF_FILE="${LLAMA_REPO}/refs/main"

TRAIN_FILE="${PROJECT_ROOT}/datasets/train/splits/train_task1_translated.json"
VAL_FILE="${PROJECT_ROOT}/datasets/train/splits/val_task1_translated.json"
SEARCH_DIR="${PROJECT_ROOT}/outputs/llama31_8b_lr_search_w5_l1_32_probe28_kl"
TRAIN_SCRIPT="${PROJECT_ROOT}/models/train_lora_align_llama31_8b.py"
BEST_LR_SCRIPT="${PROJECT_ROOT}/models/select_best_align_lr.py"

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

if [[ ! -f "$BEST_LR_SCRIPT" ]]; then
  echo "Error: Best-LR selector not found: $BEST_LR_SCRIPT"
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

if [[ ! -e "${LLAMA}/tokenizer.json" && ! -e "${LLAMA}/tokenizer.model" ]]; then
  echo "Error: Llama tokenizer files not found in: $LLAMA"
  exit 1
fi

module purge
module load cuda/11.8.0

set +u
eval "$(/share/apps/NYUAD5/miniconda/3-4.11.0/bin/conda shell.bash hook)"
conda activate adaptation
set -u

LOGS_DIR="${PROJECT_ROOT}/logs/train/llama31_8b_align_w5_reuse_${SLURM_JOB_ID:-local}"
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

run_step "Llama-3.1-8B W5 L1-L32 LR search" \
  python models/run_align_search.py \
    --train_script "$TRAIN_SCRIPT" \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --search_dir "$SEARCH_DIR" \
    --model_name "$LLAMA" \
    --layer_start 0 \
    --layer_end 31 \
    --probe_layers 28 \
    --auto_beta \
    --n_trials 5 \
    --nproc 2 \
    --num_epochs 10 \
    --seed 42 \
    "${RESUME_FLAG[@]}"

BEST_LR="$(python "$BEST_LR_SCRIPT" "${SEARCH_DIR}/search_manifest.json" --field lr)"
BEST_TRIAL="$(python "$BEST_LR_SCRIPT" "${SEARCH_DIR}/search_manifest.json" --field trial)"
echo "BEST_W5_TRIAL=${BEST_TRIAL}"
echo "BEST_W5_LR=${BEST_LR}"

run_fixed_window() {
  local window="$1"
  local paper_window="$2"
  local layer_start="$3"
  local layer_end="$4"
  local output_dir="${PROJECT_ROOT}/outputs/llama31_8b_lr_reuse_${window}_${paper_window}_probe28_kl/lr${BEST_LR}"
  local master_port=$((20000 + RANDOM % 20000))

  run_step "Llama-3.1-8B ${window} ${paper_window} fixed LR ${BEST_LR}" \
    torchrun \
      --nproc_per_node=1 \
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

run_fixed_window "w1" "l1_18" 0 17
# run_fixed_window "w2" "l18_32" 17 31
# run_fixed_window "w3" "l1_28" 0 27
# run_fixed_window "w4" "l28_32" 27 31

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  mv "${PROJECT_ROOT}/tmp_logs/job_${SLURM_JOB_ID}.out" "$LOGS_DIR/job_${SLURM_JOB_ID}.out"
  mv "${PROJECT_ROOT}/tmp_logs/job_${SLURM_JOB_ID}.err" "$LOGS_DIR/job_${SLURM_JOB_ID}.err"
fi
