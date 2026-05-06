#!/bin/bash
#SBATCH --job-name=flash_rl_cifar100_fairness
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/flash_rl_cifar100_fairness_%j.out
#SBATCH --error=logs/flash_rl_cifar100_fairness_%j.err

echo "Working directory: $(pwd)" >&2

conda activate flash_rl_env

mkdir -p logs

python -u flash_rl_cifar100_fairness.py
