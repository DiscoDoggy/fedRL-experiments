#!/bin/bash
# Run a quick smoke-test on the CPU cluster (no GPU needed, few rounds).
# Use this to verify the code works before submitting the full job.
#
#   sbatch run_test.sh
#   sbatch run_test.sh --dataset mnist      (default)
#   sbatch run_test.sh --dataset cifar10
#
#SBATCH --job-name=fedrl_test
#SBATCH --partition=cpucluster
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/fedrl_test_%j.out
#SBATCH --error=logs/fedrl_test_%j.err

# Move to the project root (parent of src/) so all relative paths resolve.
cd "$(dirname "$0")/.." || exit 1
echo "Working directory: $(pwd)" >&2
mkdir -p logs

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /Users/924322786/.conda/envs/flash_rl_env

DATASET=${1:-mnist}

if [ "$DATASET" = "cifar10" ]; then
    CONFIG=configs/cifar10_test.yaml
elif [ "$DATASET" = "femnist" ]; then
    CONFIG=configs/femnist_test.yaml
else
    CONFIG=configs/mnist_test.yaml
fi

echo "Using config: $CONFIG" >&2
python -u main_non_iid.py --config "$CONFIG" --num_rounds 2
