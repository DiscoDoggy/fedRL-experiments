#!/bin/bash
#SBATCH --job-name=favor_cifar10_test
#SBATCH --output=logs/favor_cifar10_test_%j.out
#SBATCH --error=logs/favor_cifar10_test_%j.err
#SBATCH --partition=gpuquick
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

python favor_main.py \
    --dataset         cifar10 \
    --model           resnet \
    --num_rounds      2 \
    --num_clients     10 \
    --clients_per_round 3 \
    --dirichlet_alpha 0.5 \
    --partition       dirichlet \
    --M               2.0 \
    --omega           0.5 \
    --results_dir     favor_results_test
