#!/bin/bash
#SBATCH --job-name=qfedavg_cifar10
#SBATCH --output=logs/qfedavg_cifar10_%j.out
#SBATCH --error=logs/qfedavg_cifar10_%j.err
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

python qfedavg_main.py \
    --dataset         cifar10 \
    --model           resnet \
    --q               0 \
    --num_rounds      200 \
    --num_clients     100 \
    --local_epochs    3 \
    --batch_size      32 \
    --lr              0.01 \
    --dirichlet_alpha 0.5 \
    --results_dir     qfedavg_results \
    --clients_per_round 5 10 20 30

