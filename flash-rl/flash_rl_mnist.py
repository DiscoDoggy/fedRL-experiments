"""
FLASH-RL Tutorial Script for MNIST
===================================

This script demonstrates the usage of FLASH-RL for federated learning with bias-based non-IID data.
Uses FedRL's MNIST preprocessing and SimpleMNISTCNN model.

Stages:
1. Create a bias-based (80/20) division of MNIST
2. Client and server FL initialization
3. Launch training
4. Visualize results
"""

# ==============================================================================
# Package importation
# ==============================================================================

import serverFL.Server_FLASHRL as Server_FLASHRL

import torchvision
from torchvision import datasets, transforms
import data_manipulation.Data_distribution as Data_distribution

# Import bias partitioner instead of Dirichlet
import bias_partitioner

import models.MNIST.CNN as CNN
from models.MNIST.CNN import SimpleMNISTCNN

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import torch.nn as nn
import timeit
import pickle
import torch

import prints.result_plot as result_plot

# ==============================================================================
# Experiment Configuration
# ==============================================================================
# 
# Modify these parameters to run different experiments:
# 
# DATASET_NAME options:
#   - "mnist" for MNIST dataset
# 
# NUM_CLIENTS: Total number of clients (e.g., 50, 100, 200)
# 
# CLIENTS_PER_ROUND: Number of clients selected each round (e.g., 5, 10, 20)
#   - Will be converted to fraction C = CLIENTS_PER_ROUND / NUM_CLIENTS
# 
# COMM_ROUNDS: Number of communication rounds (e.g., 20, 50, 100)
# 
# LOCAL_EPOCHS: Local training epochs per client (e.g., 5, 10)
# 
# PRIMARY_BIAS: Data bias for non-IID (e.g., 0.8 = 80% dominant class)
# 
# Results will be saved in: flash_rl_results/{EXPERIMENT_ID}/
# ==============================================================================

# Experiment parameters - modify these for different runs
DATASET_NAME = "mnist"  # Options: "mnist"
NUM_CLIENTS = 100  # Total number of clients
CLIENTS_PER_ROUND_LIST = [5, 10, 20, 30]  # List of clients per round to run experiments for
COMM_ROUNDS = 200 # Number of communication rounds
LOCAL_EPOCHS = 3  # Number of local epochs per client
PRIMARY_BIAS = 0.8  # Bias for non-IID data (80% dominant class)

# Generate base experiment identifier for data partitioning
BASE_EXPERIMENT_ID = f"{DATASET_NAME}_{NUM_CLIENTS}clients_bias{int(PRIMARY_BIAS*100)}"
print(f"\n{'='*80}")
print(f"Running experiments with CLIENTS_PER_ROUND: {CLIENTS_PER_ROUND_LIST}")
print(f"{'='*80}\n")

# ==============================================================================
# GPU Configuration
# ==============================================================================

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*80}")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"{'='*80}\n")

# ==============================================================================
# Creation of a bias-based non-IID division of MNIST
# ==============================================================================

# We'll create a bias-based non-IID partition where each client has:
# - 80% of samples from a dominant class (the client's "preferred" class)
# - 20% of samples uniformly distributed across the other 9 classes
#
# This creates strong non-IID heterogeneity while ensuring each client has 
# some diversity in their local data.

# Define transformations for MNIST matching FedRL's preprocessing
# Training transform with data augmentation
trans_mnist_train = transforms.Compose([
    transforms.RandomRotation(10),  # Random rotation for augmentation
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST empirical mean and std
])

# Test transform without augmentation
trans_mnist_test = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST empirical mean and std
])

# MNIST training dataset
dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist_train)

# MNIST testing dataset
dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist_test)

# Display dataset information
print("Number of training samples: ", len(dataset_train))
print("Size of an image: ", list(dataset_train[0][0].shape))

# Create the bias-based data partition (80% dominant class, 20% other classes)

seed = 40
num_classes = 10

# Use bias-based partitioner with the configured parameters
client_dict_bias, client_preferences = bias_partitioner.bias_partition_mnist(
    dataset=dataset_train,
    num_clients=NUM_CLIENTS,
    primary_bias=PRIMARY_BIAS,  # Use configured bias
    num_classes=num_classes,
    seed=seed
)

# Generate partition report for visualization
csv_file = f"./partition-reports/{BASE_EXPERIMENT_ID}.csv"
import os
os.makedirs("./partition-reports", exist_ok=True)

partition_df = bias_partitioner.create_partition_report(
    dataset=dataset_train,
    client_dict=client_dict_bias,
    num_classes=num_classes,
    output_file=csv_file
)

print(f"Created bias-based partition with {NUM_CLIENTS} clients")
print(f"Each client has ~{int(PRIMARY_BIAS*100)}% from dominant class, ~{int((1-PRIMARY_BIAS)*100)}% from other classes")
print(f"Partition report saved to: {csv_file}")

col_names = [f"class{i}" for i in range(num_classes)]

# Print the bias-based data distribution

client_bias_df = pd.read_csv(csv_file, header=1)
client_bias_df = client_bias_df.set_index('client')
    
# Create a new figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the distribution for the first 10 clients
colors = ['#4371C4', '#B4A7D7', '#EEB954', '#009201','#F56367' , '#ee8d0b', '#17BECF','#CCCCCC', '#AA5843','#9a48e5']

client_bias_df[col_names][:10].plot.barh(stacked=True, ax=ax, color=colors)  

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Set x-label and y-label
ax.set_xlabel('Number of data sample', fontsize=12, fontweight='semibold')
ax.set_ylabel('Client', fontsize=12, fontweight='semibold')
ax.set_title('Bias-based partition: 80% dominant class per client', fontsize=12)

# Add a transparent axis for grid lines on the same figure
ax_grid = ax.inset_axes([0, 0, 1, 1], transform=ax.transAxes, zorder=-1)
ax_grid.grid(True, color='#CCCCCC', linestyle='--')

ax_grid.set_xticklabels([])
ax_grid.set_yticklabels([])

# Save the figure with the plot and grid lines
os.makedirs("images", exist_ok=True)
plt.savefig(f"images/{BASE_EXPERIMENT_ID}_partition.pdf", dpi=400, bbox_inches='tight')

plt.show()  # Show the plot

# Structure the bias-based non-IID division
db = Data_distribution.Data_distribution(dataset_train, NUM_CLIENTS)

# Extract label lists for each client (for compatibility with existing code)
list_label = []
for client_id in range(NUM_CLIENTS):
    indices = client_dict_bias[client_id]
    client_labels = [dataset_train.targets[idx].item() if isinstance(dataset_train.targets[idx], torch.Tensor) else dataset_train.targets[idx] for idx in indices]
    unique_labels = np.unique(client_labels)
    list_label.append(unique_labels)
    
# Use the bias-based partition directly
dict_users = db.division_noniid_fedlab(client_dict_bias.values(), list_label)

print(f"Structured partition for {len(dict_users)} clients")
print(f"Sample client 0 dominant class: {client_preferences[0]}")

# Save the bias-based non-IID data division
os.makedirs(f'data_division/{DATASET_NAME.upper()}', exist_ok=True)

data_division_path = f'data_division/{DATASET_NAME.upper()}/{BASE_EXPERIMENT_ID}_dict.pkl'
with open(data_division_path, 'wb') as handle:
    pickle.dump(dict_users, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print(f"Saved bias-based partition to: {data_division_path}")

# ==============================================================================
# Generate client info files if they don't exist for MNIST
# ==============================================================================

clients_info_dir = f"clients_info/{DATASET_NAME.upper()}"
os.makedirs(clients_info_dir, exist_ok=True)

# Check if files exist, if not generate them
names_list_path = f"{clients_info_dir}/names_list.pkl"
cores_list_path = f"{clients_info_dir}/cores_list.pkl"
frequency_list_path = f"{clients_info_dir}/frequency_list.pkl"
bandwidth_list_path = f"{clients_info_dir}/bandwidth_list.pkl"

if not os.path.exists(names_list_path):
    print("Generating client info files for MNIST...")
    import random
    random.seed(42)
    
    # Generate client names
    name_list = [f"client_{i}" for i in range(NUM_CLIENTS)]
    
    # Generate CPU cores (1-8 cores per client)
    cores_list = [random.choice([1, 2, 4, 6, 8]) for _ in range(NUM_CLIENTS)]
    
    # Generate CPU frequencies (GHz) - list of possible frequencies per client
    frequency_list = [[random.uniform(1.5, 3.5) for _ in range(3)] for _ in range(NUM_CLIENTS)]
    
    # Generate bandwidth (Mbps) - list of possible bandwidths per client
    bandwidth_list = [[random.uniform(10, 100) for _ in range(3)] for _ in range(NUM_CLIENTS)]
    
    # Save the generated files
    with open(names_list_path, 'wb') as f:
        pickle.dump(name_list, f)
    with open(cores_list_path, 'wb') as f:
        pickle.dump(cores_list, f)
    with open(frequency_list_path, 'wb') as f:
        pickle.dump(frequency_list, f)
    with open(bandwidth_list_path, 'wb') as f:
        pickle.dump(bandwidth_list, f)
    
    print(f"Generated client info files in {clients_info_dir}/")

# ==============================================================================
# Loop over different CLIENTS_PER_ROUND configurations
# ==============================================================================

all_experiment_results = {}  # Store results from all experiments for comparison

for CLIENTS_PER_ROUND in CLIENTS_PER_ROUND_LIST:
    # Generate experiment identifier for this configuration
    EXPERIMENT_ID = f"{DATASET_NAME}_{NUM_CLIENTS}clients_{CLIENTS_PER_ROUND}selected_bias{int(PRIMARY_BIAS*100)}"
    
    print(f"\n{'='*80}")
    print(f"Starting Experiment: {EXPERIMENT_ID}")
    print(f"CLIENTS_PER_ROUND = {CLIENTS_PER_ROUND}")
    print(f"{'='*80}\n")

    # ==============================================================================
    # Client and server FL initialization
    # ==============================================================================

    # Read clients' information from stored files

    # Load the dictionary containing client data from the bias-based partition
    with open(data_division_path, 'rb') as f:
        Dict_users = pickle.load(f)

    # Load lists containing clients' names, cores, frequencies, and bandwidths from respective pickle files
    with open(f"{clients_info_dir}/names_list.pkl", "rb") as file:
        name_list = pickle.load(file)

    with open(f"{clients_info_dir}/cores_list.pkl", "rb") as file:
        cores_list = pickle.load(file)

    with open(f"{clients_info_dir}/frequency_list.pkl", "rb") as file:
        frequency_list = pickle.load(file)

    with open(f"{clients_info_dir}/bandwidth_list.pkl", "rb") as file:
        bandwidth_list = pickle.load(file)

    # Calculate the number of samples for each client
    number_samples = []
    for key in Dict_users.keys():
        number_samples.append(len(Dict_users[key]))

    # Combine clients' information into a list
    clients_info = list(zip(name_list, number_samples, cores_list, frequency_list, bandwidth_list))

    print(f"Loaded bias-based partition with {len(Dict_users)} clients")
    print(f"Sample sizes range: {min(number_samples)} to {max(number_samples)}")

    # Initialize SimpleMNISTCNN model (from FedRL)
    model_MNIST = SimpleMNISTCNN(num_classes=10)

    # Move model to GPU if available
    model_MNIST = model_MNIST.to(device)
    print(f"SimpleMNISTCNN model initialized and moved to {device}")

    # Initialize the FL server with specific parameters
    Serveur_FLASHRL = Server_FLASHRL.Server_FLASHRL(
        num_clients=NUM_CLIENTS,  # Number of clients participating in Federated Learning
        global_model=model_MNIST,  # Global model
        dict_clients=Dict_users,  # Dictionary containing clients data
        loss_fct=torch.nn.CrossEntropyLoss(),  # Loss function for training
        B=128,  # Local batch size for training on each client
        dataset_test=dataset_test,  # Testing dataset for evaluation
        learning_rate=0.001,  # Learning rate for Adam optimizer
        momentum=0.9,  # Momentum value (unused with Adam optimizer)
        clients_info=clients_info,  # Information about clients (names, num_samples, cores, etc.)
        device=device  # Device for GPU/CPU computation
    )

    print(f"Server initialized with {len(Serveur_FLASHRL.list_clients)} clients on {device}")

    # ==============================================================================
    # FL Training
    # ==============================================================================

    # Perform Federated Learning training on the initialized server

    # Calculate C (fraction of clients selected per round)
    C = CLIENTS_PER_ROUND / NUM_CLIENTS

    # Create results directory for checkpoints (separate from CIFAR-10 results)
    results_dir = f'flash_rl_results_mnist/{EXPERIMENT_ID}'
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nStarting training with:")
    print(f"  - {COMM_ROUNDS} communication rounds")
    print(f"  - {CLIENTS_PER_ROUND} clients selected per round (C={C:.2f})")
    print(f"  - {LOCAL_EPOCHS} local epochs per client")
    print(f"  - Checkpoints will be saved to: {results_dir}/\n")

    # Initiate the Federated Learning training process on the server
    results = Serveur_FLASHRL.global_train(
        comms_round=COMM_ROUNDS,  # Number of communication rounds for training
        C=C,  # Fraction of clients selected to participate in each training round
        E=LOCAL_EPOCHS,  # Number of local epochs for each client's training
        mu=0,  # FedProx hyperparameter (by default = 0)
        lamb=0.6,  # Past contribution factor
        rep_init=1 / NUM_CLIENTS,  # Initial reputation value
        batch_size=128,  # Batch size used in each local training iteration
        verbose_test=1,  # Verbosity level for test results (e.g., showing test accuracy)
        verbos=1,  # Verbosity level for training process (e.g., printing progress)
        checkpoint_dir=results_dir  # Save checkpoints every 10 rounds
    )

    # ==============================================================================
    # Save Results
    # ==============================================================================

    # Save the results to disk (results_dir already created before training for checkpoints)

    print(f"\nSaving final results to: {results_dir}/")

    # Save the complete results dictionary
    with open(f'{results_dir}/training_results.pkl', 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save accuracy, loss, and other metrics as CSV for easy analysis
    import pandas as pd

    metrics_df = pd.DataFrame({
        'round': range(1, len(results['Accuracy']) + 1),
        'accuracy': [acc.cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in results['Accuracy']],
        'accuracy_delta': [delta.cpu().numpy() if isinstance(delta, torch.Tensor) else delta for delta in results['Accuracy_deltas']],
        'loss': results['Loss'],
        'time_per_round': results['Timeurounds'],
        'cumulative_time': results['Timesum'],
        'dql_loss': results['LossDQL']
    })
    metrics_df.to_csv(f'{results_dir}/training_metrics.csv', index=False)

    # Save selected clients per round
    selected_clients_df = pd.DataFrame(results['Selected_clients'])
    selected_clients_df.columns = [f'client_{i}' for i in range(len(results['Selected_clients'][0]))]
    selected_clients_df.insert(0, 'round', range(1, len(results['Selected_clients']) + 1))
    selected_clients_df.to_csv(f'{results_dir}/selected_clients_per_round.csv', index=False)

    # Save utility scores per round (contribution of each selected client)
    utility_data = []
    for round_idx, (selected_clients, utility_scores) in enumerate(zip(results['Selected_clients'], results['Utility_scores'])):
        for client_idx, utility in zip(selected_clients, utility_scores):
            utility_data.append({
                'round': round_idx + 1,
                'client_id': client_idx,
                'utility_score': utility,
                'accuracy_delta': results['Accuracy_deltas'][round_idx].cpu().numpy() if isinstance(results['Accuracy_deltas'][round_idx], torch.Tensor) else results['Accuracy_deltas'][round_idx]
            })
        
    utility_df = pd.DataFrame(utility_data)
    utility_df.to_csv(f'{results_dir}/client_utility_contributions.csv', index=False)

    # Save reputation history
    reputation_df = pd.DataFrame(results['Reputation'])
    reputation_df.to_csv(f'{results_dir}/reputation_history.csv', index=False)

    # Save rewards history
    rewards_df = pd.DataFrame({
        'round': range(1, len(results['Rewards']) + 1),
        'reward': results['Rewards']
    })
    rewards_df.to_csv(f'{results_dir}/rewards_history.csv', index=False)

    # Save the best model
    torch.save(results['Best_model_weights'], f'{results_dir}/best_model.pt')

    # Save the DQN model (main network and target network)
    torch.save(results['DQL_model_weights'], f'{results_dir}/dqn_model.pt')
    torch.save(results['DQL_target_weights'], f'{results_dir}/dqn_target.pt')

    # Create client selection frequency analysis
    client_selection_count = {}
    client_avg_utility = {}
    client_total_acc_impact = {}

    for round_idx, (selected_clients, utility_scores) in enumerate(zip(results['Selected_clients'], results['Utility_scores'])):
        acc_delta = results['Accuracy_deltas'][round_idx]
        acc_delta_value = acc_delta.cpu().numpy() if isinstance(acc_delta, torch.Tensor) else acc_delta
    
        for client_idx, utility in zip(selected_clients, utility_scores):
            if client_idx not in client_selection_count:
                client_selection_count[client_idx] = 0
                client_avg_utility[client_idx] = []
                client_total_acc_impact[client_idx] = 0.0
        
            client_selection_count[client_idx] += 1
            client_avg_utility[client_idx].append(utility)
            client_total_acc_impact[client_idx] += acc_delta_value

    # Create summary dataframe
    client_summary = []
    for client_id in range(NUM_CLIENTS):
        if client_id in client_selection_count:
            # Calculate average accuracy impact per selection
            avg_acc_impact = client_total_acc_impact[client_id] / client_selection_count[client_id]
            client_summary.append({
                'client_id': client_id,
                'selection_count': client_selection_count[client_id],
                'avg_utility': np.mean(client_avg_utility[client_id]),
                'avg_acc_impact': avg_acc_impact,
                'dominant_class': client_preferences[client_id]
            })
        else:
            client_summary.append({
                'client_id': client_id,
                'selection_count': 0,
                'avg_utility': 0.0,
                'avg_acc_impact': 0.0,
                'dominant_class': client_preferences[client_id]
            })

    client_summary_df = pd.DataFrame(client_summary)
    client_summary_df = client_summary_df.sort_values('selection_count', ascending=False)
    client_summary_df.to_csv(f'{results_dir}/client_selection_summary.csv', index=False)

    print(f"\nResults saved to {results_dir}/:")
    print(f"  - training_results.pkl (complete results)")
    print(f"  - training_metrics.csv (accuracy, accuracy_delta, loss, time per round)")
    print(f"  - selected_clients_per_round.csv (which clients were selected each round)")
    print(f"  - client_utility_contributions.csv (utility score & acc delta per client per round)")
    print(f"  - client_selection_summary.csv (how often each client was selected & their impact)")
    print(f"  - reputation_history.csv (reputation values per client per round)")
    print(f"  - rewards_history.csv (rewards per round)")
    print(f"  - best_model.pt (best global FL model weights)")
    print(f"  - dqn_model.pt (DQN main network weights)")
    print(f"  - dqn_target.pt (DQN target network weights)")
    print(f"  - checkpoint_latest.pt (latest checkpoint for resuming training)")

    # ==============================================================================
    # Results Visualization
    # ==============================================================================

    # Initialize an object of the plot class to visualize results
    print_result = result_plot.result_plot()

    # Obtain the max accuracy values and corresponding training rounds for plotting
    maxaccuracy_flash_rl, traininground_flash_rl = print_result.assendante_list(results['Accuracy'])

    # Plotting the max accuracy per training round

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Add gridlines to the plot
    ax.grid(True, color='#CCCCCC', linestyle='--')

    # Plot the accuracy values per the training rounds for FLASH-RL
    ax.plot(traininground_flash_rl, maxaccuracy_flash_rl, color="#4371C4", label="FLASH-RL", linestyle='dashdot')

    # Set the linewidth for the plot axes
    ax.spines['bottom'].set_linewidth(1)  # x-axis
    ax.spines['left'].set_linewidth(1)  # y-axis

    # Add a legend and labels for better visualization understanding
    ax.legend()
    ax.set_xlabel("Training round")
    ax.set_ylabel("Accuracy (%)")

    # Save the plot
    plt.savefig(f'{results_dir}/accuracy_plot.pdf', dpi=400, bbox_inches='tight')
    plt.savefig(f'{results_dir}/accuracy_plot.png', dpi=400, bbox_inches='tight')

    # Display the plot
    plt.show()

    # Note: As the number of training rounds increases, the results tend to improve,
    # enabling the RL agent to converge towards the optimal clients.

    # Obtain the cumulative latency values for plotting
    latence_cum_flashrl = print_result.cummulative_list(results['Timeurounds'])

    # Plotting the end-to-end latency per training round

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Add gridlines to the plot
    ax.grid(True, color='#CCCCCC', linestyle='--')

    # Plot the cumulative latency values against the training rounds for FLASH-RL
    ax.plot(range(0, len(latence_cum_flashrl)), latence_cum_flashrl, color="#4371C4", label="FLASH-RL", linestyle='dashdot')

    # Set the linewidth for the plot axes
    ax.spines['bottom'].set_linewidth(1)  # x-axis
    ax.spines['left'].set_linewidth(1)  # y-axis

    # Add a legend and labels for better visualization understanding
    ax.legend()
    ax.set_xlabel("Training round")
    ax.set_ylabel("Latence (s)")

    # Save the plot
    plt.savefig(f'{results_dir}/latency_plot.pdf', dpi=400, bbox_inches='tight')
    plt.savefig(f'{results_dir}/latency_plot.png', dpi=400, bbox_inches='tight')

    # Display the plot
    plt.show()

    # Create a figure for the plot
    plt.figure(figsize=(12, 6))

    # Add gridlines to the plot
    plt.grid(True, color='#CCCCCC', linestyle='--')

    # Create a histogram of reputation values (use last available round or round 10 if available)
    rep_idx = min(10, len(results['Reputation']) - 1)
    plt.hist(results['Reputation'][rep_idx], bins=20, color="#4371C4", edgecolor='black', linewidth=2)

    # Add a vertical line at x=0.1 to show the separation between -4 to 0 and 0.1 to 1
    plt.axvline(x=0.1, color='#ee8d0b', linestyle='dashed', linewidth=2.5)

    # Add labels and title
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    # plt.title('Distribution of reputation values')

    # Save the plot
    plt.savefig(f'{results_dir}/reputation_histogram.pdf', dpi=400, bbox_inches='tight')
    plt.savefig(f'{results_dir}/reputation_histogram.png', dpi=400, bbox_inches='tight')

    # Show the plot
    plt.show()

    # Plot client selection frequency for top 20 most selected clients
    top_clients = client_summary_df.head(20)

    plt.figure(figsize=(14, 6))
    plt.grid(True, color='#CCCCCC', linestyle='--', alpha=0.7)

    # Create bar plot
    bars = plt.bar(range(len(top_clients)), top_clients['selection_count'], 
                   color="#4371C4", edgecolor='black', linewidth=1.5)

    # Color bars by average utility
    from matplotlib import cm
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=top_clients['avg_utility'].min(), vmax=top_clients['avg_utility'].max())
    colors_gradient = cm.RdYlGn(norm(top_clients['avg_utility'].values))
    for bar, color in zip(bars, colors_gradient):
        bar.set_color(color)

    plt.xlabel('Client ID', fontsize=12, fontweight='semibold')
    plt.ylabel('Number of Times Selected', fontsize=12, fontweight='semibold')
    plt.title('Top 20 Most Frequently Selected Clients (colored by average utility)', fontsize=12)
    plt.xticks(range(len(top_clients)), [str(x) for x in top_clients['client_id'].values], rotation=45)

    # Add colorbar
    sm = cm.ScalarMappable(cmap=cm.RdYlGn, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Average Utility Score', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{results_dir}/client_selection_frequency.pdf', dpi=400, bbox_inches='tight')
    plt.savefig(f'{results_dir}/client_selection_frequency.png', dpi=400, bbox_inches='tight')
    plt.show()

    print("\n" + "="*80)
    print("FLASH-RL training completed successfully!")
    print(f"Device used: {device}")
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    print("="*80)

    # Print final performance summary
    print("\nFinal Performance Summary:")
    print("-" * 80)
    final_acc = results['Accuracy'][-1]
    final_acc_value = final_acc.cpu().numpy() if isinstance(final_acc, torch.Tensor) else final_acc
    max_acc = max([acc.cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in results['Accuracy']])
    print(f"Final Test Accuracy: {final_acc_value*100:.2f}%")
    print(f"Best Test Accuracy: {max_acc*100:.2f}%")
    print(f"Final Test Loss: {results['Loss'][-1]:.4f}")
    print(f"Total Training Time: {results['Timesum'][-1]:.2f} seconds")
    print(f"Average Time per Round: {sum(results['Timeurounds'])/len(results['Timeurounds']):.2f} seconds")
    print("-" * 80)
    print(f"\nAll results and plots saved to: {results_dir}/")
    print("="*80)

    # Store results for comparison across experiments
    all_experiment_results[CLIENTS_PER_ROUND] = {
        'experiment_id': EXPERIMENT_ID,
        'final_accuracy': final_acc_value,
        'best_accuracy': max_acc,
        'final_loss': results['Loss'][-1],
        'total_time': results['Timesum'][-1],
        'results_dir': results_dir
    }

# ==============================================================================
# Final Summary: Compare all experiments
# ==============================================================================

print("\n" + "="*80)
print("ALL EXPERIMENTS COMPLETED - SUMMARY")
print("="*80)

print(f"\n{'Clients/Round':<15} {'Best Accuracy':<15} {'Final Accuracy':<15} {'Final Loss':<12} {'Total Time':<12}")
print("-" * 70)

for clients_per_round, exp_results in all_experiment_results.items():
    print(f"{clients_per_round:<15} {exp_results['best_accuracy']*100:.2f}%{'':<8} {exp_results['final_accuracy']*100:.2f}%{'':<8} {exp_results['final_loss']:.4f}{'':<6} {exp_results['total_time']:.1f}s")

print("-" * 70)
print(f"\nResults saved in:")
for clients_per_round, exp_results in all_experiment_results.items():
    print(f"  - {exp_results['results_dir']}/")
print("="*80)
