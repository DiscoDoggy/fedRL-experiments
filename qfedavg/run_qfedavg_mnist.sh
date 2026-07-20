#!/bin/bash
#SBATCH --job-name=qfedavg_mnist
#SBATCH --output=logs/qfedavg_mnist_%j.out
#SBATCH --error=logs/qfedavg_mnist_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

set -e
cd "$(dirname "$0")"
mkdir -p logs

python qfedavg_main.py \
    --dataset         mnist \
    --model           simplemnistcnn \
    --q               7 \
    --num_rounds      200 \
    --num_clients     100 \
    --local_epochs    3 \
    --batch_size      32 \
    --lr              0.01 \
    --dirichlet_alpha 0.5 \
    --results_dir     qfedavg_results \
    --clients_per_round 5 10 20 30
