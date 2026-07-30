#!/bin/bash
#SBATCH --job-name=fedrl_femnist_50r
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/fedrl_femnist_50r_%j.out
#SBATCH --error=logs/fedrl_femnist_50r_%j.err

cd "$(dirname "$0")/.." || exit 1
echo "Working directory: $(pwd)" >&2
mkdir -p logs

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /Users/924322786/.conda/envs/flash_rl_env

python -u main_non_iid.py --config configs/femnist_50r.yaml
