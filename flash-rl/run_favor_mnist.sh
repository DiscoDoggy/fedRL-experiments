#!/bin/bash
#SBATCH --job-name=favor_mnist
#SBATCH --output=logs/favor_mnist_%j.out
#SBATCH --error=logs/favor_mnist_%j.err
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /Users/924322786/.conda/envs/flash_rl_env

python favor_main.py \
    --dataset         mnist \
    --model           simplemnistcnn \
    --num_rounds      200 \
    --num_clients     100 \
    --clients_per_round 5 10 20 30 \
    --dirichlet_alpha 0.5 \
    --partition       dirichlet \
    --M               2.0 \
    --omega           0.5 \
    --results_dir     favor_results_unified