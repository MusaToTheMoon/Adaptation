#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 23:00:00
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

if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch jobs/hpc_submit_medgemma_lora_inference_task3.sh <w1|w2|w3|w4|w5> <dataset>"
  exit 1
fi

WINDOW="$1"
DATASET="$2"

PROJECT_ROOT="/scratch/mk8737/farah/Adaptation"
HF_CACHE="/scratch/mk8737/huggingface"
MEDGEMMA_REPO="${HF_CACHE}/models--google--medgemma-27b-text-it"
MEDGEMMA_REF_FILE="${MEDGEMMA_REPO}/refs/main"
INFER_SCRIPT="${PROJECT_ROOT}/models/inference_lora_task3.py"
TEST_FILE="${PROJECT_ROOT}/datasets/task3/${DATASET}.json"
JUDGE_PROMPT_FILE="${PROJECT_ROOT}/prompts/judge-task3.txt"

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
    ADAPTER_PATH="${PROJECT_ROOT}/outputs/medgemma_lr_search_w1_l1_40_probe50_kl/trial_04_lr1.51e-04"
    ;;
  w2)
    ADAPTER_PATH="${PROJECT_ROOT}/outputs/medgemma_lr_search_w2_l40_62_probe50_kl/trial_03_lr2.28e-05"
    ;;
  w3)
    ADAPTER_PATH="${PROJECT_ROOT}/outputs/medgemma_lr_search_w3_l1_50_probe50_kl/trial_02_lr2.76e-05"
    ;;
  w4)
    ADAPTER_PATH="${PROJECT_ROOT}/outputs/medgemma_lr_search_w4_l50_62_probe50_kl/trial_02_lr2.76e-05"
    ;;
  w5)
    ADAPTER_PATH="${PROJECT_ROOT}/outputs/medgemma_lr_search_w5_l1_62_probe50_kl/trial_02_lr2.76e-05"
    ;;
  *)
    echo "Error: unsupported window '$WINDOW'. Expected one of: w1 w2 w3 w4 w5"
    exit 1
    ;;
esac

MEDGEMMA_SNAPSHOT=""
MEDGEMMA=""
if [[ -f "$MEDGEMMA_REF_FILE" ]]; then
  MEDGEMMA_SNAPSHOT="$(cat "$MEDGEMMA_REF_FILE")"
  MEDGEMMA="${MEDGEMMA_REPO}/snapshots/${MEDGEMMA_SNAPSHOT}"
fi

OUTPUT_FILE="${PROJECT_ROOT}/results/predictions/task3/${DATASET}/medgemma_lora_${WINDOW}_best_lr.csv"
METRICS_FILE="${PROJECT_ROOT}/results/metrics/task3/${DATASET}/medgemma_lora_${WINDOW}_best_lr.json"

if [[ ! -f "$TEST_FILE" ]]; then
  echo "Error: Test file not found: $TEST_FILE"
  exit 1
fi

if [[ ! -f "$JUDGE_PROMPT_FILE" ]]; then
  echo "Error: Judge prompt file not found: $JUDGE_PROMPT_FILE"
  exit 1
fi

if [[ ! -f "$OUTPUT_FILE" ]]; then
  echo "Error: Predictions CSV not found: $OUTPUT_FILE"
  exit 1
fi

mkdir -p "$(dirname "$METRICS_FILE")"

set +u
eval "$(/share/apps/NYUAD5/miniconda/3-4.11.0/bin/conda shell.bash hook)"
conda activate adaptation
set -u

LOGS_DIR="${PROJECT_ROOT}/logs/judge/task3_medgemma_lora_${DATASET}_${WINDOW}_${SLURM_JOB_ID}"
mkdir -p "$LOGS_DIR"

cd "$PROJECT_ROOT"
if [[ -f ".env" ]]; then
  set -a
  source .env
  set +a
fi

echo "WINDOW=${WINDOW}"
echo "DATASET=${DATASET}"
echo "TEST_FILE=${TEST_FILE}"
echo "OUTPUT_FILE=${OUTPUT_FILE}"
echo "METRICS_FILE=${METRICS_FILE}"
echo "OPENAI_API_KEY_SET=$([[ -n "${OPENAI_API_KEY:-}" ]] && echo yes || echo no)"

run_step() {
  local step_name="$1"
  shift
  echo
  echo "===== ${step_name} ====="
  echo "CMD: $*"
  "$@"
}


# run_step "MedGemma LoRA Task3 inference (${DATASET}, ${WINDOW} best LR)" \
#   env -u OPENAI_API_KEY python "$INFER_SCRIPT" \
#     --test_file "$TEST_FILE" \
#     --instruction_file "$INSTRUCTION_FILE" \
#     --model_name medgemma \
#     --base_model "$MEDGEMMA" \
#     --adapter_path "$ADAPTER_PATH" \
#     --output_file "$OUTPUT_FILE" \
#     --metrics_file "$METRICS_FILE" \
#     --bert_device cuda:0 \
#     --offline

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Error: OPENAI_API_KEY is not set. Add it to .env or export it before sbatch for Task 3 judge."
  exit 1
fi

run_step "Task3 standalone judge (${DATASET}, ${WINDOW})" \
  python scripts/run_judge_standalone.py \
    --predictions_csv "$OUTPUT_FILE" \
    --metrics_json "$METRICS_FILE" \
    --task_type dialogue_completion \
    --dataset_json "$TEST_FILE" \
    --judge_prompt_file "$JUDGE_PROMPT_FILE"

mv "${PROJECT_ROOT}/tmp_logs/job_${SLURM_JOB_ID}.out" "$LOGS_DIR/job_${SLURM_JOB_ID}.out"
mv "${PROJECT_ROOT}/tmp_logs/job_${SLURM_JOB_ID}.err" "$LOGS_DIR/job_${SLURM_JOB_ID}.err"
