#!/bin/bash
#SBATCH --job-name=fedavg_mnist
#SBATCH --output=logs/fedavg_mnist_%j.out
#SBATCH --error=logs/fedavg_mnist_%j.err
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate /Users/924322786/.conda/envs/flash_rl_env 2>/dev/null

/Users/924322786/.conda/envs/flash_rl_env/bin/python -u main_non_iid.py \
    --config          configs/mnist.yaml \
    --no_rl           true \
    --clients_per_round 5 10 20 30