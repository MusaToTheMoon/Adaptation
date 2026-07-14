#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH --exclude=cn270
#SBATCH -o /scratch/mk8737/farah/Adaptation/tmp_logs/job_%j.out
#SBATCH -e /scratch/mk8737/farah/Adaptation/tmp_logs/job_%j.err

# make sure '/scratch/mk8737/farah/Adaptation/tmp_logs/' already exists before running this script; slurm won't create it automatically
#
# Runs LoRA inference + BERTScore only (GPU node, offline — no internet for the
# OpenAI judge). To add judge metrics afterward, run scripts/run_judge_standalone.py
# against the resulting predictions CSV on a node with internet access, same as
# jobs/hpc_submit_medgemma_lora_inference_task3.sh.

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

if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch jobs/hpc_submit_llama31_8b_lora_inference_task3.sh <w1|w2|w3|w4|w5> <dataset>"
  exit 1
fi

WINDOW="$1"
DATASET="$2"

PROJECT_ROOT="/scratch/mk8737/farah/Adaptation"
HF_CACHE="/scratch/mk8737/huggingface"
LLAMA_REPO="${HF_CACHE}/models--meta-llama--Llama-3.1-8B-Instruct"
LLAMA_REF_FILE="${LLAMA_REPO}/refs/main"
INFER_SCRIPT="${PROJECT_ROOT}/models/inference_lora_task3.py"
TEST_FILE="${PROJECT_ROOT}/datasets/task3/${DATASET}.json"

case "$DATASET" in
  emirati|emirati_row42)
    INSTRUCTION_FILE="${PROJECT_ROOT}/prompts/task3-emirati.txt"
    ;;
  jordanian)
    INSTRUCTION_FILE="${PROJECT_ROOT}/prompts/task3-jordanian.txt"
    ;;
  egyptian)
    INSTRUCTION_FILE="${PROJECT_ROOT}/prompts/task3-egyptian.txt"
    ;;
  moroccan)
    INSTRUCTION_FILE="${PROJECT_ROOT}/prompts/task3-moroccan.txt"
    ;;
  msa)
    INSTRUCTION_FILE="${PROJECT_ROOT}/prompts/task3-MSA.txt"
    ;;
  test)
    INSTRUCTION_FILE="${PROJECT_ROOT}/prompts/task3-emirati.txt"
    ;;
  *)
    INSTRUCTION_FILE=""
    ;;
esac

case "$WINDOW" in
  w1)
    ADAPTER_PATH="${PROJECT_ROOT}/outputs/llama31_8b_lr_reuse_w1_l1_18_probe28_kl/lr1.057792e-04"
    ;;
  w2)
    ADAPTER_PATH="${PROJECT_ROOT}/outputs/llama31_8b_lr_reuse_w2_l18_32_probe28_kl/lr1.057792e-04"
    ;;
  w3)
    ADAPTER_PATH="${PROJECT_ROOT}/outputs/llama31_8b_lr_reuse_w3_l1_28_probe28_kl/lr1.057792e-04"
    ;;
  w4)
    ADAPTER_PATH="${PROJECT_ROOT}/outputs/llama31_8b_lr_reuse_w4_l28_32_probe28_kl/lr1.057792e-04"
    ;;
  w5)
    ADAPTER_PATH="${PROJECT_ROOT}/outputs/llama31_8b_lr_search_w5_l1_32_probe28_kl/trial_00_lr1.06e-04"
    ;;
  *)
    echo "Error: unsupported window '$WINDOW'. Expected one of: w1 w2 w3 w4 w5"
    exit 1
    ;;
esac

OUTPUT_FILE="${PROJECT_ROOT}/results/predictions/task3/${DATASET}/llama31_8b_lora_${WINDOW}_best_lr.csv"
METRICS_FILE="${PROJECT_ROOT}/results/metrics/task3/${DATASET}/llama31_8b_lora_${WINDOW}_best_lr.json"

if [[ ! -f "$INFER_SCRIPT" ]]; then
  echo "Error: Inference script not found: $INFER_SCRIPT"
  exit 1
fi

if [[ ! -f "$TEST_FILE" ]]; then
  echo "Error: Test file not found: $TEST_FILE"
  exit 1
fi

if [[ -z "$INSTRUCTION_FILE" || ! -f "$INSTRUCTION_FILE" ]]; then
  echo "Error: Instruction file not found for dataset '$DATASET': $INSTRUCTION_FILE"
  exit 1
fi

if [[ ! -d "$ADAPTER_PATH" ]]; then
  echo "Error: Adapter path not found: $ADAPTER_PATH"
  exit 1
fi

if [[ ! -f "${ADAPTER_PATH}/adapter_config.json" ]]; then
  echo "Error: adapter_config.json not found in: $ADAPTER_PATH"
  exit 1
fi

if [[ ! -f "${ADAPTER_PATH}/adapter_model.safetensors" ]]; then
  echo "Error: adapter_model.safetensors not found in: $ADAPTER_PATH"
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

mkdir -p "$(dirname "$OUTPUT_FILE")"
mkdir -p "$(dirname "$METRICS_FILE")"

module purge
module load cuda/11.8.0

set +u
eval "$(/share/apps/NYUAD5/miniconda/3-4.11.0/bin/conda shell.bash hook)"
conda activate adaptation
set -u

LOGS_DIR="${PROJECT_ROOT}/logs/inference/llama31_8b_lora_task3_${DATASET}_${WINDOW}_${SLURM_JOB_ID}"
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
echo "WINDOW=${WINDOW}"
echo "DATASET=${DATASET}"
echo "TEST_FILE=${TEST_FILE}"
echo "INSTRUCTION_FILE=${INSTRUCTION_FILE}"
echo "ADAPTER_PATH=${ADAPTER_PATH}"
echo "OUTPUT_FILE=${OUTPUT_FILE}"
echo "METRICS_FILE=${METRICS_FILE}"
nvidia-smi -L || true

run_step() {
  local step_name="$1"
  shift
  echo
  echo "===== ${step_name} ====="
  echo "CMD: $*"
  "$@"
}

run_step "Llama-3.1-8B LoRA Task3 inference (${DATASET}, ${WINDOW} best LR)" \
  python "$INFER_SCRIPT" \
    --test_file "$TEST_FILE" \
    --instruction_file "$INSTRUCTION_FILE" \
    --model_name llama \
    --base_model "$LLAMA" \
    --adapter_path "$ADAPTER_PATH" \
    --output_file "$OUTPUT_FILE" \
    --metrics_file "$METRICS_FILE" \
    --bert_device cuda:0 \
    --offline

mv "${PROJECT_ROOT}/tmp_logs/job_${SLURM_JOB_ID}.out" "$LOGS_DIR/job_${SLURM_JOB_ID}.out"
mv "${PROJECT_ROOT}/tmp_logs/job_${SLURM_JOB_ID}.err" "$LOGS_DIR/job_${SLURM_JOB_ID}.err"
