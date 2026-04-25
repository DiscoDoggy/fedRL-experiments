#!/usr/bin/env python3
"""
Comprehensive Non-IID Federated Learning Simulation

This single-file script provides a complete non-IID federated learning simulation
incorporating all the non-IID data distribution strategies from the FLSim project.

Features:
- Bias-based non-IID data distribution
- Shard-based non-IID data distribution  
- Multiple heterogeneity levels
- Data distribution visualization
- Comprehensive analysis and reporting

Author: Based on FLSim framework
"""

import logging
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import json
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class NonIIDDataDistributor:
    """Handles non-IID data distribution strategies."""
    
    def __init__(self, dataset, labels, num_clients):
        self.dataset = dataset
        self.labels = labels
        self.num_clients = num_clients
        self.client_data = {i: [] for i in range(num_clients)}
        
    def uniform_distribution(self, N, k):
        """Uniform distribution of N items into k groups."""
        dist = []
        avg = N / k
        for i in range(k):
            dist.append(int((i + 1) * avg) - int(i * avg))
        random.shuffle(dist)
        return dist
    
    def normal_distribution(self, N, k):
        """Normal distribution of N items into k groups."""
        dist = []
        for i in range(k):
            x = i - (k - 1) / 2
            dist.append(int(N * (np.exp(-x) / (np.exp(-x) + 1)**2)))
        remainder = N - sum(dist)
        dist = list(np.add(dist, self.uniform_distribution(remainder, k)))
        return dist
    
    def group_by_labels(self):
        """Group dataset by labels."""
        grouped_data = {label: [] for label in range(len(self.labels))}
        
        for idx, (data, label) in enumerate(self.dataset):
            grouped_data[label].append((data, label))
            
        return grouped_data
    
    def bias_based_distribution(self, primary_bias=0.8, secondary_bias=False, 
                               label_distribution='uniform'):
        """
        Distribute data using bias-based non-IID strategy.
        
        Args:
            primary_bias: Percentage of data from preferred label (e.g., 0.8 = 80%)
            secondary_bias: If True, remaining data goes to one random label
            label_distribution: 'uniform' or 'normal' for client label preferences
        """
        logging.info(f"🎯 Bias-based Non-IID Distribution")
        logging.info(f"   Primary bias: {primary_bias*100}%")
        logging.info(f"   Secondary bias: {secondary_bias}")
        logging.info(f"   Label distribution: {label_distribution}")
        
        grouped_data = self.group_by_labels()
        
        # Determine client label preferences
        if label_distribution == 'uniform':
            dist = self.uniform_distribution(self.num_clients, len(self.labels))
        else:
            dist = self.normal_distribution(self.num_clients, len(self.labels))
        
        # Assign preferences to clients
        client_preferences = []
        label_idx = 0
        for i, count in enumerate(dist):
            client_preferences.extend([i] * count)
        random.shuffle(client_preferences)
        
        # Calculate data per client
        total_samples = len(self.dataset)
        samples_per_client = total_samples // self.num_clients
        
        for client_id in range(self.num_clients):
            pref_label = client_preferences[client_id]
            
            # Calculate majority and minority portions
            majority_size = int(samples_per_client * primary_bias)
            minority_size = samples_per_client - majority_size
            
            # Add majority data (preferred label)
            available_majority = len(grouped_data[pref_label])
            majority_to_take = min(majority_size, available_majority)
            
            for _ in range(majority_to_take):
                if grouped_data[pref_label]:
                    self.client_data[client_id].append(grouped_data[pref_label].pop(0))
            
            # Add minority data
            if secondary_bias:
                # All minority data from one random label
                other_labels = [l for l in range(len(self.labels)) if l != pref_label]
                secondary_label = random.choice(other_labels)
                
                available_minority = len(grouped_data[secondary_label])
                minority_to_take = min(minority_size, available_minority)
                
                for _ in range(minority_to_take):
                    if grouped_data[secondary_label]:
                        self.client_data[client_id].append(grouped_data[secondary_label].pop(0))
            else:
                # Distribute minority data among all other labels
                other_labels = [l for l in range(len(self.labels)) if l != pref_label]
                minority_dist = self.uniform_distribution(minority_size, len(other_labels))
                
                for label_idx, count in enumerate(minority_dist):
                    label = other_labels[label_idx]
                    available = len(grouped_data[label])
                    to_take = min(count, available)
                    
                    for _ in range(to_take):
                        if grouped_data[label]:
                            self.client_data[client_id].append(grouped_data[label].pop(0))
        
        return self.client_data, client_preferences
    
    def shard_based_distribution(self, shards_per_client=2):
        """
        Distribute data using shard-based non-IID strategy.
        
        Args:
            shards_per_client: Number of shards each client receives
        """
        logging.info(f"📦 Shard-based Non-IID Distribution")
        logging.info(f"   Shards per client: {shards_per_client}")
        
        # Create shards
        total_shards = self.num_clients * shards_per_client
        shard_size = len(self.dataset) // total_shards
        
        # Flatten and shuffle data
        data_list = list(self.dataset)
        random.shuffle(data_list)
        
        # Create shards
        shards = []
        for i in range(total_shards):
            start_idx = i * shard_size
            end_idx = min((i + 1) * shard_size, len(data_list))
            shards.append(data_list[start_idx:end_idx])
        
        # Distribute shards to clients
        shard_idx = 0
        for client_id in range(self.num_clients):
            for _ in range(shards_per_client):
                if shard_idx < len(shards):
                    self.client_data[client_id].extend(shards[shard_idx])
                    shard_idx += 1
        
        logging.info(f"   Created {len(shards)} shards of size ~{shard_size}")
        return self.client_data, None

class SimpleNN(nn.Module):
    """Simple neural network for MNIST-like datasets."""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class FederatedClient:
    """Federated learning client."""
    
    def __init__(self, client_id, data, device='cpu'):
        self.client_id = client_id
        self.device = device
        self.data = data
        self.model = None
        
        # Create data loader
        if data:
            inputs = torch.stack([item[0] for item in data])
            targets = torch.tensor([item[1] for item in data])
            dataset = TensorDataset(inputs, targets)
            self.dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        else:
            self.dataloader = None
    
    def set_model(self, global_model):
        """Set client model from global model."""
        self.model = SimpleNN().to(self.device)
        self.model.load_state_dict(global_model.state_dict())
    
    def train(self, epochs=5, lr=0.01):
        """Train client model."""
        if not self.dataloader or not self.model:
            return None
            
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, (data, target) in enumerate(self.dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            total_loss += epoch_loss
        
        return total_loss / (epochs * len(self.dataloader))
    
    def get_model_weights(self):
        """Get model weights."""
        if self.model:
            return {name: param.data.clone() for name, param in self.model.named_parameters()}
        return None

class FederatedServer:
    """Federated learning server."""
    
    def __init__(self, num_clients, device='cpu'):
        self.num_clients = num_clients
        self.device = device
        self.global_model = SimpleNN().to(device)
        self.clients = []
        self.round_accuracies = []
        self.round_losses = []
    
    def add_client(self, client):
        """Add client to server."""
        self.clients.append(client)
    
    def federated_averaging(self, client_weights, client_sizes):
        """Perform federated averaging."""
        if not client_weights:
            return
            
        # Calculate total samples
        total_samples = sum(client_sizes)
        
        # Initialize averaged weights
        avg_weights = {}
        for name in client_weights[0].keys():
            avg_weights[name] = torch.zeros_like(client_weights[0][name])
        
        # Weighted averaging
        for i, weights in enumerate(client_weights):
            weight = client_sizes[i] / total_samples
            for name in weights.keys():
                avg_weights[name] += weights[name] * weight
        
        # Update global model
        self.global_model.load_state_dict(avg_weights)
    
    def evaluate(self, test_loader):
        """Evaluate global model."""
        if not test_loader:
            return 0.0
            
        self.global_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.global_model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy

class NonIIDAnalyzer:
    """Analyze and visualize non-IID data distributions."""
    
    def __init__(self, client_data, labels):
        self.client_data = client_data
        self.labels = labels
    
    def analyze_distribution(self):
        """Analyze data distribution across clients."""
        analysis = {
            'client_sizes': [],
            'label_distributions': [],
            'heterogeneity_metrics': {}
        }
        
        for client_id, data in self.client_data.items():
            # Client data size
            analysis['client_sizes'].append(len(data))
            
            # Label distribution for this client
            label_counts = Counter([item[1] for item in data])
            client_dist = [label_counts.get(i, 0) for i in range(len(self.labels))]
            analysis['label_distributions'].append(client_dist)
        
        # Calculate heterogeneity metrics
        analysis['heterogeneity_metrics'] = self._calculate_heterogeneity(
            analysis['label_distributions']
        )
        
        return analysis
    
    def _calculate_heterogeneity(self, label_distributions):
        """Calculate heterogeneity metrics."""
        distributions = np.array(label_distributions)
        
        # Normalize distributions
        normalized_dists = distributions / (distributions.sum(axis=1, keepdims=True) + 1e-8)
        
        # Calculate entropy for each client
        entropies = []
        for dist in normalized_dists:
            entropy = -np.sum(dist * np.log(dist + 1e-8))
            entropies.append(entropy)
        
        # Calculate KL divergence from uniform distribution
        uniform_dist = np.ones(len(self.labels)) / len(self.labels)
        kl_divergences = []
        
        for dist in normalized_dists:
            kl_div = np.sum(dist * np.log((dist + 1e-8) / (uniform_dist + 1e-8)))
            kl_divergences.append(kl_div)
        
        return {
            'mean_entropy': np.mean(entropies),
            'std_entropy': np.std(entropies),
            'mean_kl_divergence': np.mean(kl_divergences),
            'std_kl_divergence': np.std(kl_divergences),
            'client_entropies': entropies,
            'client_kl_divergences': kl_divergences
        }
    
    def visualize_distribution(self, save_path=None):
        """Visualize data distribution across clients."""
        analysis = self.analyze_distribution()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Non-IID Data Distribution Analysis', fontsize=16)
        
        # 1. Client data sizes
        axes[0, 0].bar(range(len(analysis['client_sizes'])), analysis['client_sizes'])
        axes[0, 0].set_title('Data Size per Client')
        axes[0, 0].set_xlabel('Client ID')
        axes[0, 0].set_ylabel('Number of Samples')
        
        # 2. Label distribution heatmap
        label_matrix = np.array(analysis['label_distributions']).T
        sns.heatmap(label_matrix, ax=axes[0, 1], cmap='YlOrRd', 
                   xticklabels=range(len(self.client_data)),
                   yticklabels=[f'Label {i}' for i in range(len(self.labels))])
        axes[0, 1].set_title('Label Distribution Heatmap')
        axes[0, 1].set_xlabel('Client ID')
        
        # 3. Entropy distribution
        entropies = analysis['heterogeneity_metrics']['client_entropies']
        axes[1, 0].hist(entropies, bins=20, alpha=0.7, color='skyblue')
        axes[1, 0].axvline(np.mean(entropies), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(entropies):.2f}')
        axes[1, 0].set_title('Client Entropy Distribution')
        axes[1, 0].set_xlabel('Entropy')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # 4. KL divergence distribution
        kl_divs = analysis['heterogeneity_metrics']['client_kl_divergences']
        axes[1, 1].hist(kl_divs, bins=20, alpha=0.7, color='lightcoral')
        axes[1, 1].axvline(np.mean(kl_divs), color='red', linestyle='--',
                          label=f'Mean: {np.mean(kl_divs):.2f}')
        axes[1, 1].set_title('KL Divergence from Uniform')
        axes[1, 1].set_xlabel('KL Divergence')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"📊 Visualization saved to: {save_path}")
        
        plt.show()
        
        return analysis

class ComprehensiveNonIIDSimulation:
    """Main simulation class for comprehensive non-IID federated learning."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"🖥️  Using device: {self.device}")
        
        # Initialize components
        self.dataset = None
        self.test_dataset = None
        self.distributor = None
        self.server = None
        self.analyzer = None
        
    def load_dataset(self, dataset_name='MNIST'):
        """Load dataset."""
        logging.info(f"📁 Loading {dataset_name} dataset...")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        if dataset_name == 'MNIST':
            train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST('./data', train=False, transform=transform)
            labels = [str(i) for i in range(10)]
        elif dataset_name == 'CIFAR10':
            train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)
            labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        self.dataset = train_dataset
        self.test_dataset = test_dataset
        self.labels = labels
        
        # Create test data loader
        self.test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        
        logging.info(f"   Training samples: {len(train_dataset)}")
        logging.info(f"   Test samples: {len(test_dataset)}")
        logging.info(f"   Classes: {len(labels)}")
    
    def setup_non_iid_distribution(self):
        """Setup non-IID data distribution."""
        self.distributor = NonIIDDataDistributor(
            self.dataset, self.labels, self.config['num_clients']
        )
        
        strategy = self.config['distribution_strategy']
        
        if strategy == 'bias':
            client_data, preferences = self.distributor.bias_based_distribution(
                primary_bias=self.config['bias_settings']['primary_bias'],
                secondary_bias=self.config['bias_settings']['secondary_bias'],
                label_distribution=self.config['bias_settings']['label_distribution']
            )
        elif strategy == 'shard':
            client_data, preferences = self.distributor.shard_based_distribution(
                shards_per_client=self.config['shard_settings']['shards_per_client']
            )
        else:
            raise ValueError(f"Unknown distribution strategy: {strategy}")
        
        return client_data, preferences
    
    def setup_federated_learning(self, client_data):
        """Setup federated learning components."""
        # Initialize server
        self.server = FederatedServer(self.config['num_clients'], self.device)
        
        # Create clients
        for client_id, data in client_data.items():
            client = FederatedClient(client_id, data, self.device)
            self.server.add_client(client)
        
        logging.info(f"🤝 Created {len(self.server.clients)} federated clients")
    
    def run_federated_training(self):
        """Run federated training simulation."""
        logging.info("🚀 Starting Federated Training...")
        
        rounds = self.config['training']['rounds']
        clients_per_round = self.config['training']['clients_per_round']
        epochs_per_round = self.config['training']['epochs_per_round']
        
        for round_num in range(1, rounds + 1):
            logging.info(f"📍 Round {round_num}/{rounds}")
            
            # Select clients for this round
            selected_clients = random.sample(self.server.clients, 
                                           min(clients_per_round, len(self.server.clients)))
            
            # Send global model to selected clients
            client_weights = []
            client_sizes = []
            round_losses = []
            
            for client in selected_clients:
                client.set_model(self.server.global_model)
                loss = client.train(epochs=epochs_per_round)
                
                if loss is not None:
                    weights = client.get_model_weights()
                    if weights:
                        client_weights.append(weights)
                        client_sizes.append(len(client.data))
                        round_losses.append(loss)
            
            # Perform federated averaging
            if client_weights:
                self.server.federated_averaging(client_weights, client_sizes)
            
            # Evaluate global model
            accuracy = self.server.evaluate(self.test_loader)
            avg_loss = np.mean(round_losses) if round_losses else 0
            
            self.server.round_accuracies.append(accuracy)
            self.server.round_losses.append(avg_loss)
            
            logging.info(f"   Accuracy: {accuracy:.2f}% | Loss: {avg_loss:.4f}")
            
            # Early stopping
            if accuracy >= self.config['training'].get('target_accuracy', 100):
                logging.info(f"🎯 Target accuracy reached!")
                break
        
        logging.info("✅ Federated training completed!")
    
    def analyze_and_visualize(self):
        """Analyze and visualize results."""
        logging.info("📊 Analyzing results...")
        
        # Analyze data distribution
        self.analyzer = NonIIDAnalyzer(
            {i: client.data for i, client in enumerate(self.server.clients)},
            self.labels
        )
        
        # Create visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Data distribution analysis
        dist_analysis = self.analyzer.visualize_distribution(
            save_path=f"non_iid_distribution_{timestamp}.png"
        )
        
        # Training progress visualization
        self.plot_training_progress(save_path=f"training_progress_{timestamp}.png")
        
        return dist_analysis
    
    def plot_training_progress(self, save_path=None):
        """Plot training progress."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        rounds = range(1, len(self.server.round_accuracies) + 1)
        
        # Accuracy plot
        ax1.plot(rounds, self.server.round_accuracies, 'b-', marker='o')
        ax1.set_title('Global Model Accuracy')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Accuracy (%)')
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(rounds, self.server.round_losses, 'r-', marker='s')
        ax2.set_title('Average Training Loss')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"📈 Training progress saved to: {save_path}")
        
        plt.show()
    
    def save_results(self):
        """Save simulation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"non_iid_results_{timestamp}.json"
        
        results = {
            'config': self.config,
            'final_accuracy': self.server.round_accuracies[-1] if self.server.round_accuracies else 0,
            'round_accuracies': self.server.round_accuracies,
            'round_losses': self.server.round_losses,
            'total_rounds': len(self.server.round_accuracies),
            'timestamp': timestamp
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"💾 Results saved to: {results_file}")
        return results_file
    
    def run_complete_simulation(self):
        """Run complete non-IID simulation."""
        logging.info("🎬 Starting Comprehensive Non-IID Simulation")
        logging.info("=" * 60)
        
        # Load dataset
        self.load_dataset(self.config['dataset'])
        
        # Setup non-IID distribution
        client_data, preferences = self.setup_non_iid_distribution()
        
        # Setup federated learning
        self.setup_federated_learning(client_data)
        
        # Run training
        self.run_federated_training()
        
        # Analyze and visualize
        analysis = self.analyze_and_visualize()
        
        # Save results
        results_file = self.save_results()
        
        # Print summary
        self.print_summary(analysis)
        
        logging.info("🎉 Simulation completed successfully!")
        return results_file, analysis

    def print_summary(self, analysis):
        """Print simulation summary."""
        print("\n" + "="*60)
        print("📋 SIMULATION SUMMARY")
        print("="*60)
        print(f"Dataset: {self.config['dataset']}")
        print(f"Distribution Strategy: {self.config['distribution_strategy']}")
        print(f"Number of Clients: {self.config['num_clients']}")
        print(f"Total Rounds: {len(self.server.round_accuracies)}")
        print(f"Final Accuracy: {self.server.round_accuracies[-1]:.2f}%" if self.server.round_accuracies else "N/A")
        
        print(f"\n📊 Heterogeneity Metrics:")
        metrics = analysis['heterogeneity_metrics']
        print(f"Mean Entropy: {metrics['mean_entropy']:.3f} ± {metrics['std_entropy']:.3f}")
        print(f"Mean KL Divergence: {metrics['mean_kl_divergence']:.3f} ± {metrics['std_kl_divergence']:.3f}")
        
        print("="*60)

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'non_iid_simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )

def create_sample_configs():
    """Create sample configuration dictionaries."""
    
    # High heterogeneity bias-based config
    high_bias_config = {
        'dataset': 'MNIST',
        'num_clients': 50,
        'distribution_strategy': 'bias',
        'bias_settings': {
            'primary_bias': 0.9,  # 90% bias - very heterogeneous
            'secondary_bias': True,
            'label_distribution': 'normal'
        },
        'training': {
            'rounds': 30,
            'clients_per_round': 10,
            'epochs_per_round': 3,
            'target_accuracy': 85.0
        }
    }
    
    # Moderate heterogeneity shard-based config
    moderate_shard_config = {
        'dataset': 'MNIST',
        'num_clients': 100,
        'distribution_strategy': 'shard',
        'shard_settings': {
            'shards_per_client': 2
        },
        'training': {
            'rounds': 50,
            'clients_per_round': 20,
            'epochs_per_round': 5,
            'target_accuracy': 90.0
        }
    }
    
    # Extreme heterogeneity config
    extreme_config = {
        'dataset': 'MNIST',
        'num_clients': 20,
        'distribution_strategy': 'shard',
        'shard_settings': {
            'shards_per_client': 1  # Only 1 shard per client - extreme non-IID
        },
        'training': {
            'rounds': 100,
            'clients_per_round': 5,
            'epochs_per_round': 10,
            'target_accuracy': 75.0
        }
    }
    
    return {
        'high_bias': high_bias_config,
        'moderate_shard': moderate_shard_config,
        'extreme': extreme_config
    }

def main():
    """Main function to run comprehensive non-IID simulation."""
    setup_logging()
    
    print("🎯 Comprehensive Non-IID Federated Learning Simulation")
    print("=" * 60)
    
    # Get sample configurations
    configs = create_sample_configs()
    
    print("Available configurations:")
    for i, (name, config) in enumerate(configs.items(), 1):
        print(f"{i}. {name}: {config['distribution_strategy']} strategy")
    
    # Let user choose configuration or run all
    choice = input("\nEnter configuration number (1-3) or 'all' to run all: ").strip().lower()
    
    if choice == 'all':
        # Run all configurations
        for name, config in configs.items():
            print(f"\n🚀 Running {name} configuration...")
            simulation = ComprehensiveNonIIDSimulation(config)
            simulation.run_complete_simulation()
            print(f"✅ {name} configuration completed!\n")
    else:
        try:
            choice_idx = int(choice) - 1
            config_name = list(configs.keys())[choice_idx]
            config = configs[config_name]
            
            print(f"\n🚀 Running {config_name} configuration...")
            simulation = ComprehensiveNonIIDSimulation(config)
            simulation.run_complete_simulation()
            
        except (ValueError, IndexError):
            print("❌ Invalid choice. Running default high_bias configuration...")
            simulation = ComprehensiveNonIIDSimulation(configs['high_bias'])
            simulation.run_complete_simulation()

if __name__ == "__main__":
    main()