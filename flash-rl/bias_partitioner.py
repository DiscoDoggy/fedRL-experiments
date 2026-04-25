"""
Bias-based data partitioner for non-IID federated learning.
Creates partitions where each client has 80% from a dominant class 
and 20% uniformly distributed across other classes.
"""

import numpy as np
import random
from collections import defaultdict


def uniform_distribution(N, k):
    """Uniform distribution of 'N' items into 'k' groups."""
    dist = []
    avg = N / k
    for i in range(k):
        dist.append(int((i + 1) * avg) - int(i * avg))
    random.shuffle(dist)
    return dist


def bias_partition_cifar10(dataset, num_clients, primary_bias=0.8, num_classes=10, seed=42):
    """
    Create bias-based non-IID partition of CIFAR-10 dataset.
    
    Args:
        dataset: PyTorch dataset (e.g., CIFAR10)
        num_clients: Number of clients
        primary_bias: Fraction of data from dominant class (default: 0.8 = 80%)
        num_classes: Number of classes in dataset (default: 10 for CIFAR-10)
        seed: Random seed for reproducibility
        
    Returns:
        client_dict: Dictionary mapping client_id -> list of sample indices
        client_preferences: List of dominant class for each client
    """
    return bias_partition(dataset, num_clients, primary_bias, num_classes, seed)


def bias_partition_mnist(dataset, num_clients, primary_bias=0.8, num_classes=10, seed=42):
    """
    Create bias-based non-IID partition of MNIST dataset.
    
    Args:
        dataset: PyTorch dataset (e.g., MNIST)
        num_clients: Number of clients
        primary_bias: Fraction of data from dominant class (default: 0.8 = 80%)
        num_classes: Number of classes in dataset (default: 10 for MNIST)
        seed: Random seed for reproducibility
        
    Returns:
        client_dict: Dictionary mapping client_id -> list of sample indices
        client_preferences: List of dominant class for each client
    """
    return bias_partition(dataset, num_clients, primary_bias, num_classes, seed)


def bias_partition(dataset, num_clients, primary_bias=0.8, num_classes=10, seed=42):
    """
    Create bias-based non-IID partition of any dataset.
    
    Args:
        dataset: PyTorch dataset (e.g., CIFAR10, MNIST)
        num_clients: Number of clients
        primary_bias: Fraction of data from dominant class (default: 0.8 = 80%)
        num_classes: Number of classes in dataset (default: 10)
        seed: Random seed for reproducibility
        
    Returns:
        client_dict: Dictionary mapping client_id -> list of sample indices
        client_preferences: List of dominant class for each client
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Group indices by label
    targets = np.array(dataset.targets)
    grouped_indices = {label: [] for label in range(num_classes)}
    
    for idx, label in enumerate(targets):
        grouped_indices[label].append(idx)
    
    # Shuffle each group
    for label in grouped_indices:
        random.shuffle(grouped_indices[label])
    
    # Assign preferences uniformly across clients
    client_preferences = []
    dist = uniform_distribution(num_clients, num_classes)
    for i, count in enumerate(dist):
        client_preferences.extend([i] * count)
    random.shuffle(client_preferences)
    
    # Calculate samples per client
    total_samples = len(dataset)
    samples_per_client = total_samples // num_clients
    
    # Create client partitions
    client_dict = {}
    used_indices = {label: [] for label in range(num_classes)}
    
    for client_id in range(num_clients):
        pref_label = client_preferences[client_id]
        client_indices = []
        
        # Calculate majority and minority portions
        majority_size = int(samples_per_client * primary_bias)
        minority_size = samples_per_client - majority_size
        
        # Add majority data (preferred label)
        available_majority = grouped_indices[pref_label]
        if len(available_majority) < majority_size:
            # Not enough unused data, reuse from used pile
            grouped_indices[pref_label].extend(used_indices[pref_label])
            used_indices[pref_label] = []
            available_majority = grouped_indices[pref_label]
        
        majority_to_take = min(majority_size, len(available_majority))
        taken_majority = available_majority[:majority_to_take]
        client_indices.extend(taken_majority)
        used_indices[pref_label].extend(taken_majority)
        del grouped_indices[pref_label][:majority_to_take]
        
        # Add minority data distributed uniformly across other labels
        other_labels = [l for l in range(num_classes) if l != pref_label]
        minority_dist = uniform_distribution(minority_size, len(other_labels))
        
        for label_idx, count in enumerate(minority_dist):
            label = other_labels[label_idx]
            available = grouped_indices[label]
            
            if len(available) < count:
                # Reuse from used pile if needed
                grouped_indices[label].extend(used_indices[label])
                used_indices[label] = []
                available = grouped_indices[label]
            
            to_take = min(count, len(available))
            taken = available[:to_take]
            client_indices.extend(taken)
            used_indices[label].extend(taken)
            del grouped_indices[label][:to_take]
        
        # Shuffle client's data
        random.shuffle(client_indices)
        client_dict[client_id] = client_indices
    
    return client_dict, client_preferences


def create_partition_report(dataset, client_dict, num_classes=10, output_file=None):
    """
    Create a partition report showing data distribution across clients.
    
    Args:
        dataset: PyTorch dataset
        client_dict: Dictionary mapping client_id -> list of sample indices
        num_classes: Number of classes
        output_file: Optional CSV file path to save report
        
    Returns:
        DataFrame with class distribution per client
    """
    import pandas as pd
    
    targets = np.array(dataset.targets)
    
    rows = []
    for client_id in sorted(client_dict.keys()):
        indices = client_dict[client_id]
        client_labels = targets[indices]
        
        # Count samples per class
        class_counts = {}
        for cls in range(num_classes):
            class_counts[f'class{cls}'] = int(np.sum(client_labels == cls))
        
        row = {
            'client': client_id,
            'Amount': len(indices),
            **class_counts
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if output_file:
        # Add header row explaining the format
        with open(output_file, 'w') as f:
            f.write(f"# Data partition with {len(client_dict)} clients and {num_classes} classes\n")
        df.to_csv(output_file, mode='a', index=False)
    
    return df
