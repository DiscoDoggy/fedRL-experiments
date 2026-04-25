#!/bin/bash
#SBATCH --job-name=test_flash_rl
#SBATCH --partition=gpuquick
#SBATCH --gres=gpu:1                 # Request 1 GPU (keep this minimal)
#SBATCH --cpus-per-task=2            # Reduce CPU to increase schedulability
#SBATCH --output=logs/flash_mnist_%j.out
#SBATCH --error=logs/flash_mnist_%j.err

echo "Working directory: $(pwd)" >&2

conda activate flash_rl_env

python -u tutorial_mnist.py
