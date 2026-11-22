#!/bin/bash
#SBATCH --job-name=${SCRIPT}
#SBATCH --output=logs/${SCRIPT}_%j.out
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --partition=gengpu
#SBATCH --account=p32001
#SBATCH --gres=gpu:a100:1

mkdir -p logs

source /projects/p32001/envs/ollama-env/bin/activate
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/projects/p32001/envs/ollama-env/cuda

python -u $SCRIPT