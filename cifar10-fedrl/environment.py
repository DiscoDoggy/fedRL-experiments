import numpy as np
import torch.nn.functional as F
from scipy.stats import entropy

class FL_Environment:
    """Federated Learning Environment for RL-based client selection."""
    def __init__(self, num_clients, global_class_dist, alpha=0.3, beta=0.2):
        self.num_clients = num_clients
        self.global_class_dist = global_class_dist
        # Balancing factors for the reward function
        self.alpha = alpha  # KL divergence balancing factor
        self.beta = beta    # Participation frequency balancing factor

    def get_state(self):
        return self.global_class_dist

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
