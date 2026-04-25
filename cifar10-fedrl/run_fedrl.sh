#!/bin/bash
#SBATCH --job-name=fedrl_resnet
#SBATCH --partition=gpuquick
#SBATCH --gres=gpu:1                 # Request 1 GPU (keep this minimal)
#SBATCH --cpus-per-task=2            # Reduce CPU to increase schedulability
#SBATCH --output=logs/fedrl_%j.out
#SBATCH --error=logs/fedrl_%j.err

# Activate your virtual environment
source /Users/924322786/programming/Untitled Folder/FedRL/cifar-10/100_clients_non-iid_cifar-10/Non-IID_FedRL_Main/FedRL-main/Cifar-10_Non_IID_FedRL-main_100_clients/fedrl_env/bin/activate

# Navigate to your project directory
cd /Users/924322786/programming/Untitled Folder/FedRL/cifar-10/100_clients_non-iid_cifar-10/Non-IID_FedRL_Main/FedRL-main/Cifar-10_Non_IID_FedRL-main_100_clients

# Run your Python script
python main_non_iid.py 

