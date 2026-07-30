#!/bin/bash
#SBATCH --job-name=flashrl_cifar10
#SBATCH --output=logs/flashrl_cifar10_%j.out
#SBATCH --error=logs/flashrl_cifar10_%j.err
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

python flash_rl_main.py \
    --dataset         cifar10 \
    --model           resnet \
    --num_rounds      200 \
    --num_clients     100 \
    --clients_per_round 5 10 20 30 \
    --dirichlet_alpha 0.5 \
    --partition       dirichlet \
    --results_dir     flash_rl_results_unified
