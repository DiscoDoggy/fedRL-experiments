#!/bin/bash
#SBATCH --job-name=qfedavg_cifar100_test
#SBATCH --output=logs/qfedavg_cifar100_test_%j.out
#SBATCH --error=logs/qfedavg_cifar100_test_%j.err
#SBATCH --partition=gpuquick
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# set -e
# cd "$(dirname "$0")"

python qfedavg_main.py \
    --dataset         cifar100 \
    --model           mobilenet \
    --num_rounds      5 \
    --num_clients     10 \
    --clients_per_round 3 \
    --q               7 \
    --results_dir     qfedavg_results_test
