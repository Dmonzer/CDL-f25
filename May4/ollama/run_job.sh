#!/bin/bash
#SBATCH --job-name=scores_${MONTH}
#SBATCH --output=logs/scores_%x_%j.out
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --partition=gengpu
#SBATCH --account=pXXXXX
#SBATCH --gres=gpu:a100:1

mkdir -p logs

export OLLAMA_HOST=http://127.0.0.1:$((11434 + SLURM_JOB_ID % 1000))

ollama serve &
sleep 10

source /projects/pXXXXX/envs/ollama-env/bin/activate

python -u Get_scores_batches.py --month $MONTH