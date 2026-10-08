#!/bin/bash
#SBATCH --job-name=fedrl_losshyb_cifar10
#SBATCH --output=logs/fedrl_losshyb_cifar10_%j.out
#SBATCH --error=logs/fedrl_losshyb_cifar10_%j.err
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /Users/924322786/.conda/envs/flash_rl_env

python -u main_non_iid.py \
    --config          configs/cifar10.yaml \
    --reward_formula  loss_hybrid \
    --alpha           0.5 \
    --beta            1.0 \
    --gamma           2.0 \
    --loss_weight     1.0 \
    --use_target_network false \
    --clients_per_round 20 30 