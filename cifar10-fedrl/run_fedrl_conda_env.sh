#!/bin/bash
#SBATCH --job-name=fedrl
#SBATCH --partition=gpuquick
#SBATCH --gres=gpu:1                 # Request 1 GPU (keep this minimal)
#SBATCH --cpus-per-task=2            # Reduce CPU to increase schedulability
#SBATCH --output=logs/fedrl_%j.out
#SBATCH --error=logs/fedrl_%j.err

echo "Working directory: $(pwd)" >&2

conda activate flash_rl_env

python -u main_non_iid.py 
