#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -p nvidia
#SBATCH --gres=gpu:2
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
CSV_PATH="${PROJECT_ROOT}/diagnosis/medgemma_sampled_quadrants.csv"
VAL_FILE="${PROJECT_ROOT}/datasets/train/splits/val_task1_translated.json"

LENS_DIR="${PROJECT_ROOT}/diagnosis/tuned_lens_medgemma"
LENS_OUT_DIR="${PROJECT_ROOT}/diagnosis/tuned_lens_medgemma_out"
PATCH_OUT_DIR="${PROJECT_ROOT}/diagnosis/activation_patching_medgemma_out"
PATCH_REVERSE_OUT_DIR="${PROJECT_ROOT}/diagnosis/activation_patching_medgemma_reverse_out"
KL_OUT_DIR="${PROJECT_ROOT}/diagnosis/kl_profile_medgemma_out"
PATCH_PANEL_OUT_DIR="${PROJECT_ROOT}/diagnosis/activation_patching_panel_out"
PATCH_REVERSE_PANEL_OUT_DIR="${PROJECT_ROOT}/diagnosis/activation_patching_panel_reverse_out"

if [[ ! -f "$CSV_PATH" ]]; then
  echo "Error: CSV not found: $CSV_PATH"
  exit 1
fi

if [[ ! -f "$VAL_FILE" ]]; then
  echo "Error: Validation file not found: $VAL_FILE"
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

LOGS_DIR="${PROJECT_ROOT}/logs/diagnosis/medgemma_${SLURM_JOB_ID}"
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
echo "CSV_PATH=${CSV_PATH}"
echo "VAL_FILE=${VAL_FILE}"
nvidia-smi -L || true

run_step() {
  local step_name="$1"
  shift
  echo
  echo "===== ${step_name} ====="
  echo "CMD: $*"
  "$@"
}

# Already run:
# run_step "Diagnose MedGemma" \
#   python diagnosis/diagnose.py \
#     --csv "$CSV_PATH" \
#     --model_path "$MEDGEMMA" \
#     --n_examples 3

# Already run:
# run_step "Train tuned lens" \
#   python diagnosis/tuned_lens_medgemma.py \
#     --mode train \
#     --train_csv "$CSV_PATH" \
#     --model_path "$MEDGEMMA" \
#     --lens_dir "$LENS_DIR" \
#     --train_n 400 \
#     --epochs 10 \
#     --train_batch 4

# Already run:
# run_step "Eval tuned lens" \
#   python diagnosis/tuned_lens_medgemma.py \
#     --mode eval \
#     --csv "$CSV_PATH" \
#     --model_path "$MEDGEMMA" \
#     --lens_dir "$LENS_DIR" \
#     --out_dir "$LENS_OUT_DIR" \
#     --batch_size 1

# Already run:
# run_step "Activation patching EN -> AR" \
#   python diagnosis/activation_patching_medgemma.py \
#     --csv "$CSV_PATH" \
#     --model_path "$MEDGEMMA" \
#     --out_dir "$PATCH_OUT_DIR" \
#     --batch_size 1

# Already run:
# run_step "Activation patching AR -> EN" \
#   python diagnosis/activation_patching_medgemma_reverse.py \
#     --csv "$CSV_PATH" \
#     --model_path "$MEDGEMMA" \
#     --out_dir "$PATCH_REVERSE_OUT_DIR" \
#     --batch_size 1

run_step "KL profiling" \
  python diagnosis/probe_kl_profile_medgemma1.py \
    --data_file "$VAL_FILE" \
    --output_dir "$KL_OUT_DIR" \
    --model_name "$MEDGEMMA" \
    --n_examples 300 \
    --l_patch 40

# run_step "Activation patching line panel" \
#   python diagnosis/plot_patching_line_panel.py \
#     --medgemma_dir "$PATCH_OUT_DIR" \
#     --out_dir "$PATCH_PANEL_OUT_DIR"

# run_step "Reverse activation patching line panel" \
#   python diagnosis/plot_patching_panel_reverse_lines.py \
#     --medgemma_dir "$PATCH_REVERSE_OUT_DIR" \
#     --out_dir "$PATCH_REVERSE_PANEL_OUT_DIR"

mv "${PROJECT_ROOT}/tmp_logs/job_${SLURM_JOB_ID}.out" "$LOGS_DIR/job_${SLURM_JOB_ID}.out"
mv "${PROJECT_ROOT}/tmp_logs/job_${SLURM_JOB_ID}.err" "$LOGS_DIR/job_${SLURM_JOB_ID}.err"
