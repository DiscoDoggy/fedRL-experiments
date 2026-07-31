#!/bin/bash
#SBATCH --job-name=favor_cifar10
#SBATCH --output=logs/favor_cifar10_%j.out
#SBATCH --error=logs/favor_cifar10_%j.err
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

python favor_main.py \
    --dataset         cifar10 \
    --model           resnet \
    --num_rounds      200 \
    --num_clients     100 \
    --clients_per_round 20 30 \
    --dirichlet_alpha 0.5 \
    --partition       dirichlet \
    --M               2.0 \
    --omega           0.5 \
    --results_dir     favor_results_unified
