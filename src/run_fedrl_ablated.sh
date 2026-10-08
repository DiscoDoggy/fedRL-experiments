#!/bin/bash
#SBATCH --job-name=fedrl_ablated
#SBATCH --output=logs/fedrl_ablated_%j.out
#SBATCH --error=logs/fedrl_ablated_%j.err
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# cd "$(dirname "$0")/.."
# mkdir -p logs

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /Users/924322786/.conda/envs/flash_rl_env

python -u main_non_iid.py \
    --config          configs/cifar10.yaml \
    --reward_formula  acc_only