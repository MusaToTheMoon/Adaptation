#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 0-23:59:59
#SBATCH -o /scratch/mk8737/farah/Adaptation/tmp_logs/job_%A_%a.out
#SBATCH -e /scratch/mk8737/farah/Adaptation/tmp_logs/job_%A_%a.err

set -euo pipefail

PROJECT_ROOT="/scratch/mk8737/farah/Adaptation"

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
  echo "Usage: sbatch --array=0-<N-1>%3 jobs/hpc_submit_eval_array.sh <pairs_file>"
  echo "Example: sbatch --array=0-5%3 jobs/hpc_submit_eval_array.sh jobs/pairs_task1.txt"
  exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Error: This script must be submitted as a Slurm job array (--array=...)."
  exit 1
fi

PAIRS_FILE="$1"
if [[ ! -f "$PAIRS_FILE" ]]; then
  echo "Error: Pairs file not found: $PAIRS_FILE"
  exit 1
fi

mapfile -t PAIRS < <(grep -vE '^\s*(#|$)' "$PAIRS_FILE")
if [[ ${#PAIRS[@]} -eq 0 ]]; then
  echo "Error: No valid model/dataset entries found in $PAIRS_FILE"
  exit 1
fi

INDEX=${SLURM_ARRAY_TASK_ID}
if (( INDEX < 0 || INDEX >= ${#PAIRS[@]} )); then
  echo "Error: SLURM_ARRAY_TASK_ID=$INDEX out of range (0..$(( ${#PAIRS[@]} - 1 )))"
  exit 1
fi

PAIR="${PAIRS[$INDEX]}"
TASK_NUM=""
MODEL_TYPE=""
DATASET=""

if [[ "$PAIR" == *:*:* ]]; then
  TASK_NUM="${PAIR%%:*}"
  REST="${PAIR#*:}"
  MODEL_TYPE="${REST%%:*}"
  DATASET="${REST##*:}"
else
  read -r TASK_NUM MODEL_TYPE DATASET <<< "$PAIR"
fi

if [[ -z "$TASK_NUM" || -z "$MODEL_TYPE" || -z "$DATASET" ]]; then
  echo "Error: Invalid pair format at index $INDEX: '$PAIR'"
  echo "Use either 'task_num:model:dataset' or 'task_num model dataset'"
  exit 1
fi

CONFIG_PATH="${PROJECT_ROOT}/configs/task${TASK_NUM}/${MODEL_TYPE}_${DATASET}.yaml"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Error: Config not found: $CONFIG_PATH"
  exit 1
fi

module purge
module load cuda/11.8.0

set +u
eval "$(/share/apps/NYUAD5/miniconda/3-4.11.0/bin/conda shell.bash hook)"
conda activate adaptation
set -u

LOGS_DIR="${PROJECT_ROOT}/logs/task${TASK_NUM}/${MODEL_TYPE}_${DATASET}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$LOGS_DIR"

echo "ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
echo "ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "TASK_NUM=${TASK_NUM}"
echo "MODEL_TYPE=${MODEL_TYPE}"
echo "DATASET=${DATASET}"

echo "Running config: configs/task${TASK_NUM}/${MODEL_TYPE}_${DATASET}.yaml"
cd "$PROJECT_ROOT"
if [[ -f ".env" ]]; then
  set -a
  source .env
  set +a
fi
export HF_HUB_ENABLE_HF_TRANSFER=1
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true

python scripts/run_evaluation.py "configs/task${TASK_NUM}/${MODEL_TYPE}_${DATASET}.yaml"

# Task 2/3 additional step: run judge LLM on the generated predictions
if [[ "$TASK_NUM" == "2" || "$TASK_NUM" == "3" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Error: OPENAI_API_KEY is not set. Add it to .env or export it before sbatch."
    exit 1
  fi
  PREDICTIONS_CSV="${PROJECT_ROOT}/results/predictions/task3/${DATASET}/${MODEL_TYPE}.csv"
  METRICS_JSON="${PROJECT_ROOT}/results/metrics/task3/${DATASET}/${MODEL_TYPE}.json"
  TASK_TYPE="dialogue_completion"
  JUDGE_PROMPT_FILE="${PROJECT_ROOT}/prompts/judge-task3.txt"

  python scripts/run_judge_standalone.py \
    --predictions_csv "$PREDICTIONS_CSV" \
    --metrics_json "$METRICS_JSON" \
    --task_type "$TASK_TYPE" \
    --dataset_json datasets/task3/${DATASET}.json \
    --judge_prompt_file "$JUDGE_PROMPT_FILE"
fi

mv "${PROJECT_ROOT}/tmp_logs/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "$LOGS_DIR/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
mv "${PROJECT_ROOT}/tmp_logs/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "$LOGS_DIR/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
