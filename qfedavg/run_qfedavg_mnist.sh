#!/bin/bash
#SBATCH --job-name=qfedavg_mnist
#SBATCH --output=logs/qfedavg_mnist_%j.out
#SBATCH --error=logs/qfedavg_mnist_%j.err
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:1


python qfedavg_main.py \
    --dataset         mnist \
    --model           simplemnistcnn \
    --q               0.1 \
    --num_rounds      200 \
    --num_clients     100 \
    --local_epochs    3 \
    --batch_size      128 \
    --lr              0.01 \
    --optimizer       sgd \
    --eta_s           0.12 \
    --dirichlet_alpha 0.5 \
    --results_dir     qfedavg_results \
    --clients_per_round 5 10 20 30
