#!/bin/bash
#SBATCH --job-name=flashrl_cifar100_test
#SBATCH --output=logs/flashrl_cifar100_test_%j.out
#SBATCH --error=logs/flashrl_cifar100_test_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1

set -e
cd "$(dirname "$0")"
mkdir -p logs

python flash_rl_main.py \
    --dataset         cifar100 \
    --model           mobilenet \
    --num_rounds      5 \
    --num_clients     10 \
    --clients_per_round 3 \
    --results_dir     flash_rl_results_test
