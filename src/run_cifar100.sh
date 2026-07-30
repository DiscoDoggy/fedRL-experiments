#!/bin/bash
#SBATCH --job-name=fedrl_cifar_100
#SBATCH --partition=gpuquick
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/fedrl_cifar100_%j.out
#SBATCH --error=logs/fedrl_cifar100_%j.err

# Move to the project root (parent of src/) so all relative paths resolve.
cd "$(dirname "$0")/.." || exit 1
echo "Working directory: $(pwd)" >&2
mkdir -p logs

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /Users/924322786/.conda/envs/flash_rl_env

python -u main_non_iid.py --config configs/cifar100_test.yaml