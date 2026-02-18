#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p nvidia
#SBATCH --mem=32G
#SBATCH -t 23:00:00
#SBATCH -o /scratch/mk8737/farah/Adaptation/tmp_logs/job_%j.out
#SBATCH -e /scratch/mk8737/farah/Adaptation/tmp_logs/job_%j.err

# make sure '/scratch/mk8737/farah/Adaptation/tmp_logs/' already exists before running this script; slurm won't create it automatically

module purge
module load cuda/11.8.0

# Make sure Conda is initialised for non-interactive shells
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate adaptation

# Set variables
MODEL_TYPE="mistral_small"
DATASET="mmlu-arabic"

# Copy outputs into designated directory
LOGS_DIR=/scratch/mk8737/farah/Adaptation/logs/${MODEL_TYPE}_${DATASET}_${SLURM_JOB_ID}
mkdir -p $LOGS_DIR

# Main Command
source .env
python scripts/run_evaluation.py configs/task1/${MODEL_TYPE}_${DATASET}.yaml

# Cleanup: Move logs to the designated directory
mv /scratch/mk8737/farah/Adaptation/tmp_logs/job_${SLURM_JOB_ID}.out $LOGS_DIR/job_${SLURM_JOB_ID}.out
mv /scratch/mk8737/farah/Adaptation/tmp_logs/job_${SLURM_JOB_ID}.err $LOGS_DIR/job_${SLURM_JOB_ID}.err