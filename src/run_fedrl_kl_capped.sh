#!/bin/bash
#SBATCH --job-name=fedrl_kl_capped
#SBATCH --output=logs/fedrl_kl_capped_%j.out
#SBATCH --error=logs/fedrl_kl_capped_%j.err
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /Users/924322786/.conda/envs/flash_rl_env

python -u main_non_iid.py \
    --config          configs/cifar10.yaml \
    --reward_formula  kl_capped \
    --alpha           0.5 \
    --beta            0.3 \
    --gamma           2.0
