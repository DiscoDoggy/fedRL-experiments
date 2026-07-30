#!/bin/bash
#SBATCH --job-name=qfedavg_sent140
#SBATCH --output=logs/qfedavg_sent140_%j.out
#SBATCH --error=logs/qfedavg_sent140_%j.err
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

python qfedavg_main.py \
    --dataset         sent140 \
    --model           lstm \
    --q               1.0 \
    --num_rounds      200 \
    --num_clients     1101 \
    --clients_per_round 10 \
    --lr              0.03 \
    --batch_size      32 \
    --eta_s           0.12 \
    --results_dir     qfedavg_results
