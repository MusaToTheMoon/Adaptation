#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/scratch/mk8737/farah/Adaptation"
TASK_NUM="${1:-1}"
MODEL_TYPE="${2:-gpt52}"
DATASET="${3:-test}"
CONFIG_PATH="${PROJECT_ROOT}/configs/task${TASK_NUM}/${MODEL_TYPE}_${DATASET}.yaml"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Error: Config not found: $CONFIG_PATH"
  echo "Usage: jobs/run_api_model_local.sh [task_num] [model_type] [dataset]"
  echo "Example: jobs/run_api_model_local.sh 1 gpt52 test"
  exit 1
fi

cd "$PROJECT_ROOT"
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

python scripts/run_evaluation_old.py "configs/task${TASK_NUM}/${MODEL_TYPE}_${DATASET}.yaml"
