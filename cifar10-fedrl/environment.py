import numpy as np
import torch.nn.functional as F
from scipy.stats import entropy

class FL_Environment:
    """Federated Learning Environment for RL-based client selection."""
    def __init__(self, num_clients, global_class_dist, clients_per_round=10, client_sizes=None, client_distributions=None, alpha=0.3, beta=0.2):
        self.num_clients = num_clients
        self.global_class_dist = global_class_dist
        self.clients_per_round = clients_per_round
        # Balancing factors for the reward function
        self.alpha = alpha  # KL divergence balancing factor
        self.beta = beta    # Participation frequency balancing factor

        # Precompute static state features
        if client_sizes is not None:
            sizes = np.array(client_sizes, dtype=np.float64)
            self.norm_client_sizes = sizes / sizes.sum()
        else:
            self.norm_client_sizes = np.zeros(num_clients)

        if client_distributions is not None:
            self.client_kl_divs = np.array([
                self.compute_kl_divergence(cd, global_class_dist)
                for cd in client_distributions
            ])
            max_kl = self.client_kl_divs.max()
            self.norm_kl_divs = self.client_kl_divs / max_kl if max_kl > 0 else self.client_kl_divs
        else:
            self.client_kl_divs = np.zeros(num_clients)
            self.norm_kl_divs = np.zeros(num_clients)

    def get_state(self, participation_freq=None):
        """Return 210-dim state: global class dist (10) + norm participation freq (100) + norm client sizes (100)."""
        if participation_freq is not None:
            freq_array = np.array([participation_freq.get(i, 0) for i in range(self.num_clients)], dtype=np.float64)
            total = freq_array.sum()
            norm_freq = freq_array / total if total > 0 else freq_array
        else:
            norm_freq = np.zeros(self.num_clients)
        return np.concatenate([self.global_class_dist, norm_freq, self.norm_client_sizes])

    def compute_kl_divergence(self, client_dist, global_dist):
        """
        Compute KL divergence between client and global distributions.
        KL(P||Q) = sum(P * log(P/Q))
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        client_dist = np.array(client_dist) + epsilon
        global_dist = np.array(global_dist) + epsilon
        
        # Normalize distributions
        client_dist = client_dist / np.sum(client_dist)
        global_dist = global_dist / np.sum(global_dist)
        
        # Compute KL divergence
        kl_div = np.sum(client_dist * np.log(client_dist / global_dist))
        return kl_div

    def compute_reward(self, prev_acc, new_acc, client_class_dist, client_part_freq, client_size, round):
        """
        Calculate reward using the new formula:
        rt = ΔAcc / [(1 + β × fc) × (1 + α × DKL(Pc || Pg)) × log(1 + |Dc|)]
        
        Where:
        - ΔAcc: Accuracy improvement (new_acc - prev_acc)
        - fc: Client participation frequency
        - DKL(Pc || Pg): KL divergence between client and global distributions
        - |Dc|: Client dataset size
        - α, β: Balancing factors
        """
        # Calculate accuracy improvement
        delta_acc = new_acc - prev_acc
        
        # Calculate KL divergence between client and global distributions
        kl_divergence = self.compute_kl_divergence(client_class_dist, self.global_class_dist)
        
        # Apply the new reward formula
        # rt = ΔAcc / [(1 + β × fc) × (1 + α × DKL(Pc || Pg)) × log(1 + |Dc|)]
        participation_factor = 1 + self.beta * client_part_freq
        kl_factor = 1 + self.alpha * kl_divergence
        size_factor = np.log(1 + client_size)
        
        denominator = participation_factor * kl_factor * size_factor
        
        # Avoid division by zero
        if denominator == 0:
            denominator = 1e-8
            
        reward = delta_acc / denominator
        # if round == 0:
        #     return 0
        
        # reward = delta_acc / max(prev_acc, 1e-8)
        
        return reward

    def step(self, selected_client_indexes, prev_acc, new_acc, client_distributions, participation_freq, client_sizes):
        """Simulate FL training step and compute reward."""
        rewards = []
        for c in selected_client_indexes:
            reward = self.compute_reward(
                prev_acc, 
                new_acc, 
                client_distributions[c], 
                participation_freq[c], 
                client_sizes[c]
            )
            rewards.append(reward)
        
        return np.mean(rewards)
