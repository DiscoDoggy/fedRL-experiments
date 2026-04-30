#!/usr/bin/env python3
"""
Non-IID Federated Reinforcement Learning Main Script

This script implements federated learning with non-IID data distribution
using the new reward function and MNIST dataset with SimpleNN model.

Features:
- Non-IID data distribution (bias-based and shard-based)
- New reward function with balancing factors
- MNIST dataset with SimpleNN model
- RL-based client selection
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random
import logging
import os
import json
from torch.utils.data import DataLoader

# Import project modules
from client import Client
from server import Server
from environment import FL_Environment
from dqn_agent import DQN_Agent
from models import SimpleNN, SimpleMNISTCNN, ResNetFed, SimpleCIFAR10CNN
from non_iid_distributor import NonIIDDataDistributor, NonIIDAnalyzer, create_non_iid_config
from dirchlet_partitioner import direchlet_partition, plot_stacked_client_class_distributions

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def save_dqn(agent: DQN_Agent, path):
    check_pt = {
        'model': agent.model.state_dict(),
        'target_model': agent.target_model.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'hparams': {
            'state_size': agent.model.fc1.in_features,
            'action_size': agent.model.fc4.out_features,
            'lr': agent.optimizer.param_groups[0]['lr'],
            'gamma': 0.9
        }
    }

    torch.save(check_pt, path)

def save_server_model(server: Server, path='global_model.pt'):
    check_pt = {
        'model_state': server.global_model.state_dict()
    }

    torch.save(check_pt, path)
                      
def main():
    """Main function to run non-IID federated learning simulation."""
    
    # ==================== CONFIGURATION ====================

    clients_per_round: list[int] = [5, 10, 20, 30]

    try:
        for k in clients_per_round: 
            num_clients = 100
            # k = 60  # clients per round
            num_classes = 10
            num_rounds = 200
            
            # Balancing factors for the new reward function
            alpha = 0.5  # KL divergence balancing factor
            beta = 0.3   # Participation frequency balancing factor

            results_parent_path: str = f'results_for_runs_cifar_ddqn_with_larger_network_and_richer_state'
            per_run_path: str = f'{num_clients}_clients_{k}_per_round_cifar'
            full_run_results_dir_path: str = results_parent_path + '/' + per_run_path  
            run_logs_path = full_run_results_dir_path + '/' + 'logs'
            run_plots_path = full_run_results_dir_path + '/' + 'plots'
            run_json_results_path = full_run_results_dir_path + '/' + 'run_results.json'
            run_client_class_dist_plot = full_run_results_dir_path + '/' + 'client_dist_plots'
            os.makedirs(full_run_results_dir_path, exist_ok=True)
            os.makedirs(run_plots_path, exist_ok=True)
            os.makedirs(run_client_class_dist_plot, exist_ok=True)
            
            fh = logging.FileHandler(run_logs_path)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
    
            logger.addHandler(fh)
            
            # Model selection
            use_simple_nn = True  # Set to True for SimpleNN, False for SimpleMNISTCNN
            
            logging.info("Starting Non-IID Federated Learning Simulation")
            # logging.info(f"   Strategy: {non_iid_strategy}")
            # logging.info(f"   Bias level: {bias_level}" if non_iid_strategy == 'bias' else f"   Shards: {num_shards}")
            logging.info(f"   Alpha (KL): {alpha}, Beta (freq): {beta}")
            
            # ==================== DATASET PREPARATION ====================
            
            # CIFAR-10 transforms
            transform_train_cifar = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            
            transform_test_cifar = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            
            # Load CIFAR-10 dataset
            cifar_dataset = torchvision.datasets.CIFAR10(
                root="./data", train=True, download=True, transform=transform_train_cifar
            )
            cifar_test_dataset = torchvision.datasets.CIFAR10(
                root="./data", train=False, download=True, transform=transform_test_cifar
            )
            
            logging.info(f"Loaded CIFAR-10 dataset: {len(cifar_dataset)} train, {len(cifar_test_dataset)} test")
            
            # ==================== NON-IID DATA DISTRIBUTION ====================

            logging.info("Performing dirchlet ")
            
            # client_datasets = direchlet_partition(
            #     dataset=cifar_dataset,
            #     num_clients=num_clients,
            #     num_classes=10,
            #     alpha=0.1,    
            #     seed=42,
            #     min_size_per_client=1
            # )

            # plot_stacked_client_class_distributions(
            #     client_datasets,
            #     num_classes=10,
            #     out_path=run_client_class_dist_plot + f'/client_class_dist_{k}.pdf'
            # )

            logging.info('perorming bias data partitioning')
            distributor = NonIIDDataDistributor(
                dataset=cifar_dataset,
                num_clients=num_clients
            )
            # list of client datasets
            client_datasets = distributor.bias_based_distribution()[0] 

            # ==================== FEDERATED LEARNING SETUP ====================
            
            # Create clients with non-IID data
            clients = []
            for i in range(num_clients):
                logging.info(f"Creating client {i + 1}")
                client = Client(i, client_datasets[i], num_classes)
                
                # Use ResNetFed for CIFAR-10
                # client.model = ResNetFed().to(client.device)
                client.model = ResNetFed().to(client.device)
                client.optimizer = torch.optim.Adam(client.model.parameters(), lr=0.001, weight_decay=1e-4)
                client.criterion = torch.nn.CrossEntropyLoss()
                clients.append(client)
            
            logging.info(f"👥 Created {num_clients} clients with Resnetfed models")
            
            # Create server with SimpleCIFAR10CNN
            server = Server(cifar_test_dataset, num_classes)
            # server.global_model = ResNetFed().to(server.device)
            server.global_model = ResNetFed().to(server.device)
            
            logging.info("Initialized server with resnetfed model")
            
            # Create FL environment with new reward function parameters
            # Calculate global class distribution from all client datasets
            global_class_counts = np.zeros(10)
            client_sizes_list = []
            client_distributions_list = []
            for client_dataset in client_datasets:
                labels = [client_dataset.dataset[idx][1] for idx in client_dataset.indices]
                client_class_counts = np.zeros(10)
                for label in labels:
                    client_class_counts[label] += 1
                global_class_counts += client_class_counts
                client_sizes_list.append(len(client_dataset))
                client_distributions_list.append(client_class_counts / max(client_class_counts.sum(), 1e-8))
            global_class_dist = global_class_counts / np.sum(global_class_counts)
            
            env = FL_Environment(
                num_clients=num_clients,
                global_class_dist=global_class_dist,
                clients_per_round=k,
                client_sizes=client_sizes_list,
                client_distributions=client_distributions_list,
                alpha=alpha,
                beta=beta
            )
            
            # Create DQN agent — state: class_dist(10) + norm_freq(100) + norm_sizes(100) = 210
            agent = DQN_Agent(state_size=num_classes + num_clients * 2, action_size=num_clients)
            
            logging.info(f"🤖 Initialized DQN agent and FL environment (α={alpha}, β={beta})")
            
            # ==================== FEDERATED TRAINING ====================
            
            rewards = []
            accuracies = []
            losses = []  # Track global losses
            participation_freq = {}  # Track client participation frequency
            client_accuracies = {}
            
            logging.info("Starting federated learning training...")
            
            for epoch in range(num_rounds):
                logging.info(f"--- Round {epoch + 1}/{num_rounds} ---")
                
                # Get current state
                state = env.get_state(participation_freq)

                # # update greedy epsilon decay
                # agent.episode = epoch
                # agent.update_epsilon()
                
                # DQN agent selects clients
                selected_client_idxs = agent.select_clients(state, num_clients, k)
                selected_clients = [clients[i] for i in selected_client_idxs]
                logging.info(f"DQN Selected clients for training: {selected_client_idxs}")
                
                # Update participation frequency 
                for client_idx in selected_client_idxs:
                    participation_freq[client_idx] = participation_freq.get(client_idx, 0) + 1
                
                # Store previous accuracy and loss for reward calculation
                prev_acc, prev_loss = server.evaluate()
                
                # Local training
                client_models = []
                client_training_metrics = []
                for client_id in selected_client_idxs:
                    logging.info(f"   Training client {client_id}...")
                    training_result = clients[client_id].train(epochs=3)  # Local epochs
                    client_models.append(training_result['model_state'])
                    client_training_metrics.append({
                        'client_id': client_id,
                        'final_loss': training_result['final_loss'],
                        'final_accuracy': training_result['final_accuracy'],
                        'losses': training_result['losses'],
                        'accuracies': training_result['accuracies']
                    })
                
                # Log local training summary
                logging.info("   Local training summary:")
                for metrics in client_training_metrics:
                    logging.info(f"     Client {metrics['client_id']}: Final Loss: {metrics['final_loss']:.4f}, Final Accuracy: {metrics['final_accuracy']:.4f}")
                
                # Server aggregates models using FedAvg
                server.aggregate_models(client_models)
                
                # Update all client models with the aggregated global model
                global_state_dict = server.global_model.state_dict()
                for client in clients:
                    client.model.load_state_dict(global_state_dict)
                
                # Evaluate global model
                current_acc, current_loss = server.evaluate()
                accuracies.append(current_acc)
                losses.append(current_loss)

                # evaluate client contribution
                for client_idx in selected_client_idxs:
                    contribution = (current_acc - prev_acc) / max(prev_acc, 1e-8)
                    if client_idx not in client_accuracies:
                        client_accuracies[client_idx] = [contribution]
                    else:
                        client_accuracies[client_idx].append(contribution)

                # Log global model performance
                logging.info(f"   Global model - Accuracy: {current_acc:.4f}, Loss: {current_loss:.4f}")
                
                # Calculate reward using the new reward function
                # We need to calculate the reward for each selected client and then average
                total_reward = 0
                for client_idx in selected_client_idxs:
                    # Get client's class distribution
                    client_dataset = client_datasets[client_idx]
                    labels = [client_dataset.dataset[idx][1] for idx in client_dataset.indices]
                    client_class_counts = np.zeros(10)
                    for label in labels:
                        client_class_counts[label] += 1
                    client_class_dist = client_class_counts / np.sum(client_class_counts)
                    
                    # Get client participation frequency (initialize to 1 for first round)
                    client_part_freq = participation_freq.get(client_idx, 1)
                    
                    # Get client dataset size
                    client_size = len(client_dataset)
                    
                    # Calculate reward for this client
                    client_reward = env.compute_reward(
                        prev_acc=prev_acc,
                        new_acc=current_acc,
                        client_class_dist=client_class_dist,
                        client_part_freq=client_part_freq,
                        client_size=client_size,
                        round=epoch
                    )
                    total_reward += client_reward
                
                # Average reward across selected clients
                reward = total_reward / len(selected_client_idxs) if selected_client_idxs else 0
                rewards.append(reward)
                
                # Get next state for DQN training
                next_state = env.get_state(participation_freq)
                
                # Train DQN agent with experience
                agent.train(state, selected_client_idxs, reward, next_state)

                # Periodically update target network (Double DQN)
                if (epoch + 1) % agent.target_update_freq == 0:
                    agent.update_target()
                    logging.info(f"   Updated target network (round {epoch + 1})")
                
                logging.info(f"   Global accuracy: {current_acc:.4f}, Reward: {reward:.4f}")
            
            save_dqn(agent, 'dqn_checkpoint.pt')
            save_server_model(server, 'global_model_checkpoint.pt')
            
            # ==================== RESULTS ====================
            
            logging.info("Training completed!")
            logging.info(f"   Final accuracy: {accuracies[-1]:.4f}")
            logging.info(f"   Final loss: {losses[-1]:.4f}")
            logging.info(f"   Average reward: {np.mean(rewards):.4f}")
            logging.info(f"   Best accuracy: {max(accuracies):.4f}")
            logging.info(f"   Lowest loss: {min(losses):.4f}")
            
            # Plot results
            plt.figure(figsize=(18, 12))
            
            # Plot 1: Global accuracy over rounds
            plt.subplot(2, 4, 1)
            plt.plot(range(1, num_rounds + 1), accuracies, 'b-', linewidth=2, marker='o')
            plt.title(f'Global Model Accuracy Over Rounds on CIFAR-10 with {num_clients} Clients')
            plt.xlabel('Round')
            plt.ylabel('Accuracy')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            
            # Plot 2: Global loss over rounds
            plt.subplot(2, 4, 2)
            plt.plot(range(1, num_rounds + 1), losses, 'orange', linewidth=2, marker='d')
            plt.title(f'Global Model Loss Over Rounds on CIFAR-10 with {num_clients} Clients')
            plt.xlabel('Round')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            
            # Plot 3: Rewards over rounds
            plt.subplot(2, 4, 3)
            plt.plot(range(1, num_rounds + 1), rewards, 'r-', linewidth=2, marker='s')
            plt.title('Rewards (Modified Function)')
            plt.xlabel('Round')
            plt.ylabel('Reward')
            plt.grid(True, alpha=0.3)
            
            # Plot 4: Client participation frequency
            plt.subplot(2, 4, 4)
            participation_counts = [participation_freq.get(i, 0) for i in range(num_clients)]
            plt.bar(range(num_clients), participation_counts, color='green', alpha=0.7)
            plt.title('Client Participation Frequency')
            plt.xlabel('Client ID')
            plt.ylabel('Participation Count')
            plt.grid(True, alpha=0.3)
            
            # Plot 5: Summary statistics
            # plt.subplot(2, 4, 5)
            # stats_labels = ['Mean KL Div', 'Mean Entropy', 'Std KL Div']
            # stats_values = [
            #     analysis_results['heterogeneity_metrics']['mean_kl_divergence'],
            #     analysis_results['heterogeneity_metrics']['mean_entropy'],
            #     analysis_results['heterogeneity_metrics']['std_kl_divergence']
            # ]
            # plt.bar(stats_labels, stats_values, color='purple', alpha=0.7)
            # plt.title('Data Distribution Statistics')
            # plt.ylabel('Value')
            # plt.xticks(rotation=45)
            # plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{run_plots_path}/federated_learning_results.png', dpi=300, bbox_inches='tight')
            logging.info("Results visualization saved as 'federated_learning_results.png'")
            
            # Save individual plots in separate folder
            # import os
            # os.makedirs('individual_plots', exist_ok=True)
            
            # Individual Plot 1: Global accuracy over rounds
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, num_rounds + 1), accuracies, 'b-', linewidth=2, marker='o')
            plt.title(f'Global Model Accuracy Over Rounds on CIFAR-10 with {num_clients} Clients')
            plt.xlabel('Round')
            plt.ylabel('Accuracy')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(f'{run_plots_path}/global_accuracy.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Individual Plot 2: Global loss over rounds
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, num_rounds + 1), losses, 'orange', linewidth=2, marker='d')
            plt.title(f'Global Model Loss Over Rounds on CIFAR-10 with {num_clients} Clients')
            plt.xlabel('Round')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{run_plots_path}/global_loss.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Individual Plot 3: Rewards over rounds
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, num_rounds + 1), rewards, 'r-', linewidth=2, marker='s')
            plt.title('Rewards (Modified Function)')
            plt.xlabel('Round')
            plt.ylabel('Reward')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{run_plots_path}/rewards.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Individual Plot 4: Client participation frequency
            plt.figure(figsize=(10, 6))
            participation_counts = [participation_freq.get(i, 0) for i in range(num_clients)]
            plt.bar(range(num_clients), participation_counts, color='green', alpha=0.7)
            plt.title('Client Participation Frequency')
            plt.xlabel('Client ID')
            plt.ylabel('Participation Count')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{run_plots_path}/client_participation.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Individual Plot 5: Data distribution statistics
            # plt.figure(figsize=(10, 6))
            # stats_labels = ['Mean KL Div', 'Mean Entropy', 'Std KL Div']
            # stats_values = [
            #     analysis_results['heterogeneity_metrics']['mean_kl_divergence'],
            #     analysis_results['heterogeneity_metrics']['mean_entropy'],
            #     analysis_results['heterogeneity_metrics']['std_kl_divergence']
            # ]
            # plt.bar(stats_labels, stats_values, color='purple', alpha=0.7)
            # plt.title('Data Distribution Statistics')
            # plt.ylabel('Value')
            # plt.xticks(rotation=45)
            # plt.grid(True, alpha=0.3)
            # plt.tight_layout()
            # plt.savefig(f'{run_plots_path}/data_distribution.png', dpi=300, bbox_inches='tight')
            # plt.close()
            
            logging.info("Individual plots saved in 'individual_plots' folder")
            plt.close()  # Close any remaining figures to free memory
            
            #aggregate client_accuracies
            processed_client_accs = {}
            for client_idx in client_accuracies:
                processed_client_accs[client_idx] = sum(client_accuracies[client_idx]) / len(client_accuracies[client_idx])

            # Save results
            results = {
                'accuracies': accuracies,
                'losses': losses,  # Include losses in saved results
                'rewards': rewards,
                'participation_frequency': participation_freq,  # Use local variable instead of env attribute
                # 'analysis_results': analysis_results,
                'processed_client_accuracies': processed_client_accs,
                'client_accuracies': client_accuracies,
                'config': {
                    'num_clients': num_clients,
                    'num_rounds': num_rounds,
                    'clients_per_round': k,
                    'non_iid_strategy': 'bias',
                    # 'bias_level': bias_level if non_iid_strategy == 'bias' else None,
                    # 'num_shards': num_shards if non_iid_strategy == 'shard' else None,
                    'alpha': alpha,
                    'beta': beta
                }
            }
            
            torch.save(results, f'{full_run_results_dir_path}/fedrl_non_iid_results.pt')
            # logging.info("Results saved to 'fedrl_non_iid_results.pt'")
            with open(run_json_results_path, 'w') as f: 
                json.dump(results, f, indent=4)
    
            logging.info(f'Results saved to {run_json_results_path}')
            
            fh.close()
        
    except Exception as e:
        logging.exception(f'Critical error occured: {e}')
        fh.close()
        raise

if __name__ == "__main__":
    main()