#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 3-00:00:00
#SBATCH -o /scratch/mk8737/farah/Adaptation/tmp_logs/job_%j.out
#SBATCH -e /scratch/mk8737/farah/Adaptation/tmp_logs/job_%j.err

set -euo pipefail

START_TS=$(date +%s)
trap 'END_TS=$(date +%s); ELAPSED=$((END_TS-START_TS)); printf "\nTotal runtime: %02d:%02d:%02d\n" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))' EXIT

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <gemma3_27b_it|mistral_7b|llama31_8b_inst|falcon|fanar>"
  exit 2
fi

MODEL_KEY="$1"

PROJECT_ROOT="/scratch/mk8737/farah/Adaptation"
HF_CACHE="/scratch/mk8737/huggingface"
VAL_FILE="${PROJECT_ROOT}/datasets/train/splits/val_task1_translated.json"

case "$MODEL_KEY" in
  gemma3_27b_it)
    MODEL_REPO="${HF_CACHE}/models--google--gemma-3-27b-it"
    CSV_PATH="${PROJECT_ROOT}/diagnosis/gemma3_27b_it_sampled_quadrants.csv"
    MODEL_LABEL="Gemma-3-27B-IT"
    ;;
  mistral_7b)
    MODEL_REPO="${HF_CACHE}/models--mistralai--Mistral-7B-Instruct-v0.3"
    CSV_PATH="${PROJECT_ROOT}/diagnosis/mistral_7b_sampled_quadrants.csv"
    MODEL_LABEL="Mistral-7B-Instruct-v0.3"
    ;;
  llama31_8b_inst)
    MODEL_REPO="${HF_CACHE}/models--meta-llama--Llama-3.1-8B-Instruct"
    CSV_PATH="${PROJECT_ROOT}/diagnosis/llama31_8b_inst_sampled_quadrants.csv"
    MODEL_LABEL="Llama-3.1-8B-Instruct"
    ;;
  falcon)
    MODEL_REPO="${HF_CACHE}/models--tiiuae--Falcon-H1-7B-Instruct"
    CSV_PATH="${PROJECT_ROOT}/diagnosis/falcon_sampled_quadrants.csv"
    MODEL_LABEL="Falcon-H1-7B-Instruct"
    ;;
  fanar)
    MODEL_REPO="${HF_CACHE}/models--QCRI--Fanar-1-9B"
    CSV_PATH="${PROJECT_ROOT}/diagnosis/fanar_sampled_quadrants.csv"
    MODEL_LABEL="Fanar-1-9B"
    ;;
  *)
    echo "Error: unsupported model key: $MODEL_KEY"
    exit 2
    ;;
esac

MODEL_REF_FILE="${MODEL_REPO}/refs/main"
LENS_DIR="${PROJECT_ROOT}/diagnosis/tuned_lens_${MODEL_KEY}"
LENS_OUT_DIR="${PROJECT_ROOT}/diagnosis/tuned_lens_${MODEL_KEY}_out"
PATCH_OUT_DIR="${PROJECT_ROOT}/diagnosis/activation_patching_${MODEL_KEY}_out"
PATCH_REVERSE_OUT_DIR="${PROJECT_ROOT}/diagnosis/activation_patching_${MODEL_KEY}_reverse_out"
KL_OUT_DIR="${PROJECT_ROOT}/diagnosis/kl_profile_${MODEL_KEY}_out"
LOGS_DIR="${PROJECT_ROOT}/logs/diagnosis/${MODEL_KEY}_${SLURM_JOB_ID:-local}"

if [[ ! -f "$CSV_PATH" ]]; then echo "Error: CSV not found: $CSV_PATH"; exit 1; fi
if [[ ! -f "$VAL_FILE" ]]; then echo "Error: Validation file not found: $VAL_FILE"; exit 1; fi
if [[ ! -f "$MODEL_REF_FILE" ]]; then echo "Error: model ref file not found: $MODEL_REF_FILE"; exit 1; fi

MODEL_SNAPSHOT="$(cat "$MODEL_REF_FILE")"
MODEL_PATH="${MODEL_REPO}/snapshots/${MODEL_SNAPSHOT}"
if [[ ! -d "$MODEL_PATH" ]]; then echo "Error: model snapshot not found: $MODEL_PATH"; exit 1; fi
if [[ ! -e "${MODEL_PATH}/tokenizer.json" && ! -e "${MODEL_PATH}/tokenizer.model" && ! -e "${MODEL_PATH}/tekken.json" ]]; then
  echo "Error: tokenizer files not found in: $MODEL_PATH"
  echo "Cache or copy tokenizer.json/tokenizer.model/tekken.json for ${MODEL_LABEL}, then rerun."
  exit 1
fi

module purge
module load cuda/11.8.0

set +u
eval "$(/share/apps/NYUAD5/miniconda/3-4.11.0/bin/conda shell.bash hook)"
conda activate adaptation
set -u

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

echo "MODEL_KEY=${MODEL_KEY}"
echo "MODEL_LABEL=${MODEL_LABEL}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "CSV_PATH=${CSV_PATH}"
echo "VAL_FILE=${VAL_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true

run_step() {
  local step_name="$1"
  shift
  echo
  echo "===== ${step_name} ====="
  echo "CMD: $*"
  "$@"
}

# run_step "Diagnose ${MODEL_LABEL}" \
#   python diagnosis/diagnose.py \
#     --csv "$CSV_PATH" \
#     --model_path "$MODEL_PATH" \
#     --n_examples 3

# run_step "Train tuned lens" \
#   python diagnosis/tuned_lens_generic.py \
#     --mode train \
#     --train_csv "$CSV_PATH" \
#     --model_path "$MODEL_PATH" \
#     --model_label "$MODEL_LABEL" \
#     --lens_dir "$LENS_DIR" \
#     --train_n 400 \
#     --epochs 10 \
#     --train_batch 4

# run_step "Eval tuned lens" \
#   python diagnosis/tuned_lens_generic.py \
#     --mode eval \
#     --csv "$CSV_PATH" \
#     --model_path "$MODEL_PATH" \
#     --model_label "$MODEL_LABEL" \
#     --lens_dir "$LENS_DIR" \
#     --out_dir "$LENS_OUT_DIR" \
#     --batch_size 1

# run_step "Activation patching EN -> AR and AR -> EN" \
#   python diagnosis/activation_patching_generic.py \
#     --csv "$CSV_PATH" \
#     --model_path "$MODEL_PATH" \
#     --model_label "$MODEL_LABEL" \
#     --out_dir "$PATCH_OUT_DIR" \
#     --reverse_out_dir "$PATCH_REVERSE_OUT_DIR" \
#     --batch_size 1 \
#     --direction both

# L_PATCH="$(python diagnosis/derive_l_patch.py --patch_npz "${PATCH_OUT_DIR}/patching_results.npz" --threshold 80)"
# echo "Derived L_PATCH=${L_PATCH} from ${PATCH_OUT_DIR}/patching_results.npz"

run_step "KL profiling" \
  python diagnosis/probe_kl_profile_medgemma1.py \
    --data_file "$VAL_FILE" \
    --output_dir "$KL_OUT_DIR" \
    --model_name "$MODEL_PATH" \
    --model_label "$MODEL_LABEL" \
    --n_examples 300 \
    --l_patch "$L_PATCH"

# run_step "Activation patching line plot" \
#   python diagnosis/plot_patching_line.py \
#     --patch_dir "$PATCH_OUT_DIR" \
#     --out_dir "$PATCH_OUT_DIR" \
#     --model_label "$MODEL_LABEL"

# run_step "Reverse activation patching line plot" \
#   python diagnosis/plot_patching_reverse_line.py \
#     --patch_dir "$PATCH_REVERSE_OUT_DIR" \
#     --out_dir "$PATCH_REVERSE_OUT_DIR" \
#     --model_label "$MODEL_LABEL"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  mv "${PROJECT_ROOT}/tmp_logs/job_${SLURM_JOB_ID}.out" "$LOGS_DIR/job_${SLURM_JOB_ID}.out"
  mv "${PROJECT_ROOT}/tmp_logs/job_${SLURM_JOB_ID}.err" "$LOGS_DIR/job_${SLURM_JOB_ID}.err"
fi
