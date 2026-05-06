#!/bin/bash
#SBATCH --job-name=fairness_fedrl
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/fairness_fedrl_%j.out
#SBATCH --error=logs/fairness_fedrl_%j.err

echo "Working directory: $(pwd)" >&2

conda activate flash_rl_env

python -u fairness_experiment.py --method fedrl
