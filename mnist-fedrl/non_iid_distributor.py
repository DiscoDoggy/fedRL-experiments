#!/usr/bin/env python3
"""
Non-IID Data Distribution Module for Federated Learning

This module provides comprehensive non-IID data distribution strategies
for federated learning simulations, including bias-based and shard-based approaches.

Author: Based on comprehensive_non_iid_simulation.py
"""

import logging
import random
import numpy as np
from collections import defaultdict, Counter
import torch
from torch.utils.data import Subset
import warnings
warnings.filterwarnings('ignore')

class NonIIDDataDistributor:
    """Handles non-IID data distribution strategies for federated learning."""
    
    def __init__(self, dataset, num_clients, num_classes=10):
        self.dataset = dataset
        self.num_clients = num_clients
        self.num_classes = num_classes
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
        """Group dataset indices by labels."""
        grouped_indices = {label: [] for label in range(self.num_classes)}
        
        for idx, (data, label) in enumerate(self.dataset):
            grouped_indices[label].append(idx)
            
        return grouped_indices
    
    def bias_based_distribution(self, primary_bias=0.8, secondary_bias=False, 
                               label_distribution='uniform'):
        """
        Distribute data using bias-based non-IID strategy.
        
        Args:
            primary_bias: Percentage of data from preferred label (e.g., 0.8 = 80%)
            secondary_bias: If True, remaining data goes to one random label
            label_distribution: 'uniform' or 'normal' for client label preferences
            
        Returns:
            List of Subset objects for each client
        """
        logging.info(f"🎯 Bias-based Non-IID Distribution")
        logging.info(f"   Primary bias: {primary_bias*100}%")
        logging.info(f"   Secondary bias: {secondary_bias}")
        logging.info(f"   Label distribution: {label_distribution}")
        
        grouped_indices = self.group_by_labels()
        
        # Determine client label preferences
        if label_distribution == 'uniform':
            dist = self.uniform_distribution(self.num_clients, self.num_classes)
        else:
            dist = self.normal_distribution(self.num_clients, self.num_classes)
        
        # Assign preferences to clients
        client_preferences = []
        for i, count in enumerate(dist):
            client_preferences.extend([i] * count)
        random.shuffle(client_preferences)
        
        # Calculate data per client
        total_samples = len(self.dataset)
        samples_per_client = total_samples // self.num_clients
        
        client_subsets = []
        
        for client_id in range(self.num_clients):
            pref_label = client_preferences[client_id]
            client_indices = []
            
            # Calculate majority and minority portions
            majority_size = int(samples_per_client * primary_bias)
            minority_size = samples_per_client - majority_size
            
            # Add majority data (preferred label)
            available_majority = len(grouped_indices[pref_label])
            majority_to_take = min(majority_size, available_majority)
            
            for _ in range(majority_to_take):
                if grouped_indices[pref_label]:
                    client_indices.append(grouped_indices[pref_label].pop(0))
            
            # Add minority data
            if secondary_bias:
                # All minority data from one random label
                other_labels = [l for l in range(self.num_classes) if l != pref_label]
                secondary_label = random.choice(other_labels)
                
                available_minority = len(grouped_indices[secondary_label])
                minority_to_take = min(minority_size, available_minority)
                
                for _ in range(minority_to_take):
                    if grouped_indices[secondary_label]:
                        client_indices.append(grouped_indices[secondary_label].pop(0))
            else:
                # Distribute minority data among all other labels
                other_labels = [l for l in range(self.num_classes) if l != pref_label]
                minority_dist = self.uniform_distribution(minority_size, len(other_labels))
                
                for label_idx, count in enumerate(minority_dist):
                    label = other_labels[label_idx]
                    available = len(grouped_indices[label])
                    to_take = min(count, available)
                    
                    for _ in range(to_take):
                        if grouped_indices[label]:
                            client_indices.append(grouped_indices[label].pop(0))
            
            # Create subset for this client
            client_subsets.append(Subset(self.dataset, client_indices))
        
        logging.info(f"   Created {len(client_subsets)} client datasets")
        return client_subsets, client_preferences
    
    def shard_based_distribution(self, shards_per_client=2):
        """
        Distribute data using shard-based non-IID strategy.
        
        Args:
            shards_per_client: Number of shards each client receives
            
        Returns:
            List of Subset objects for each client
        """
        logging.info(f"📦 Shard-based Non-IID Distribution")
        logging.info(f"   Shards per client: {shards_per_client}")
        
        # Create shards
        total_shards = self.num_clients * shards_per_client
        shard_size = len(self.dataset) // total_shards
        
        # Create indices and shuffle them
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        
        # Create shards
        shards = []
        for i in range(total_shards):
            start_idx = i * shard_size
            end_idx = min((i + 1) * shard_size, len(indices))
            shards.append(indices[start_idx:end_idx])
        
        # Distribute shards to clients
        client_subsets = []
        shard_idx = 0
        
        for client_id in range(self.num_clients):
            client_indices = []
            for _ in range(shards_per_client):
                if shard_idx < len(shards):
                    client_indices.extend(shards[shard_idx])
                    shard_idx += 1
            
            # Create subset for this client
            client_subsets.append(Subset(self.dataset, client_indices))
        
        logging.info(f"   Created {len(shards)} shards of size ~{shard_size}")
        logging.info(f"   Created {len(client_subsets)} client datasets")
        return client_subsets, None

class NonIIDAnalyzer:
    """Analyze and visualize non-IID data distributions."""
    
    def __init__(self, client_datasets, num_classes=10):
        self.client_datasets = client_datasets
        self.num_classes = num_classes
    
    def analyze_distribution(self):
        """Analyze data distribution across clients."""
        analysis = {
            'client_sizes': [],
            'label_distributions': [],
            'heterogeneity_metrics': {}
        }
        
        for client_dataset in self.client_datasets:
            # Client data size
            analysis['client_sizes'].append(len(client_dataset))
            
            # Label distribution for this client
            labels = [client_dataset.dataset[idx][1] for idx in client_dataset.indices]
            label_counts = Counter(labels)
            client_dist = [label_counts.get(i, 0) for i in range(self.num_classes)]
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
        uniform_dist = np.ones(self.num_classes) / self.num_classes
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

def create_non_iid_config(strategy='bias', heterogeneity='moderate'):
    """
    Create non-IID configuration presets.
    
    Args:
        strategy: 'bias' or 'shard'
        heterogeneity: 'low', 'moderate', 'high', 'extreme'
    """
    configs = {
        'bias': {
            'low': {
                'distribution_strategy': 'bias',
                'bias_settings': {
                    'primary_bias': 0.6,
                    'secondary_bias': False,
                    'label_distribution': 'uniform'
                }
            },
            'moderate': {
                'distribution_strategy': 'bias',
                'bias_settings': {
                    'primary_bias': 0.7,
                    'secondary_bias': False,
                    'label_distribution': 'uniform'
                }
            },
            'high': {
                'distribution_strategy': 'bias',
                'bias_settings': {
                    'primary_bias': 0.8,
                    'secondary_bias': True,
                    'label_distribution': 'normal'
                }
            },
            'extreme': {
                'distribution_strategy': 'bias',
                'bias_settings': {
                    'primary_bias': 0.9,
                    'secondary_bias': True,
                    'label_distribution': 'normal'
                }
            }
        },
        'shard': {
            'low': {
                'distribution_strategy': 'shard',
                'shard_settings': {
                    'shards_per_client': 4
                }
            },
            'moderate': {
                'distribution_strategy': 'shard',
                'shard_settings': {
                    'shards_per_client': 3
                }
            },
            'high': {
                'distribution_strategy': 'shard',
                'shard_settings': {
                    'shards_per_client': 2
                }
            },
            'extreme': {
                'distribution_strategy': 'shard',
                'shard_settings': {
                    'shards_per_client': 1
                }
            }
        }
    }
    
    return configs[strategy][heterogeneity]