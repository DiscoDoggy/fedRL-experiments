#!/bin/bash
#SBATCH --job-name=fedrl_fedavg_cifar10
#SBATCH --output=logs/fedrl_fedavg_cifar10_%j.out
#SBATCH --error=logs/fedrl_fedavg_cifar10_%j.err
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /Users/924322786/.conda/envs/flash_rl_env

python -u main_non_iid.py --config configs/cifar10.yaml --no_rl true
