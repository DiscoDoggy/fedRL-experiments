#!/bin/bash
#SBATCH --job-name=fedrl_exp_tau020
#SBATCH --output=logs/fedrl_exp_tau020_%j.out
#SBATCH --error=logs/fedrl_exp_tau020_%j.err
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /Users/924322786/.conda/envs/flash_rl_env

python -u main_non_iid.py \
    --config          configs/cifar10.yaml \
    --reward_formula  exp_fairness \
    --alpha           0.5 \
    --beta            1.0 \
    --gamma           2.0 \
    --exp_temp        0.20
