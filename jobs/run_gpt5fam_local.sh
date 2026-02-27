#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/scratch/mk8737/farah/Adaptation"
MODEL_TYPE="${1:-gpt52}"
DATASET="${2:-test}"
CONFIG_PATH="${PROJECT_ROOT}/configs/task1/${MODEL_TYPE}_${DATASET}.yaml"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Error: Config not found: $CONFIG_PATH"
  echo "Usage: jobs/run_gpt5fam_local.sh [model_type] [dataset]"
  echo "Example: jobs/run_gpt5fam_local.sh gpt52 test"
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

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Error: OPENAI_API_KEY is not set."
  echo "Set it in shell or in ${PROJECT_ROOT}/.env"
  exit 1
fi

python scripts/run_evaluation.py "configs/task1/${MODEL_TYPE}_${DATASET}.yaml"
