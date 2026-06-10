import numpy as np


class FL_Environment:
    """Federated Learning Environment for RL-based client selection.

    Parameters
    ----------
    reward_formula : str
        'full'   — rt = ΔAcc / [(1 + β·fc)(1 + α·DKL)(log(1 + |Dc|))]
                   Used by the CIFAR-10 experiment.
        'simple' — rt = ΔAcc / max(prev_acc, ε)
                   Used by the MNIST experiment.
    """

    def __init__(self, num_clients, global_class_dist, alpha=0.3, beta=0.2,
                 reward_formula: str = 'full'):
        self.num_clients = num_clients
        self.global_class_dist = global_class_dist
        self.alpha = alpha   # KL divergence balancing factor
        self.beta = beta     # Participation frequency balancing factor
        if reward_formula not in ('full', 'simple'):
            raise ValueError("reward_formula must be 'full' or 'simple'")
        self.reward_formula = reward_formula

    def get_state(self):
        return self.global_class_dist

    def compute_kl_divergence(self, client_dist, global_dist):
        """KL(client_dist || global_dist)."""
        epsilon = 1e-8
        client_dist = np.array(client_dist) + epsilon
        global_dist = np.array(global_dist) + epsilon
        client_dist = client_dist / np.sum(client_dist)
        global_dist = global_dist / np.sum(global_dist)
        return float(np.sum(client_dist * np.log(client_dist / global_dist)))

    def compute_reward(self, prev_acc, new_acc, client_class_dist,
                       client_part_freq, client_size):
        """Compute per-client reward.

        The formula used depends on ``self.reward_formula``:

        'full':
            rt = ΔAcc / [(1 + β·fc) · (1 + α·DKL(Pc‖Pg)) · log(1 + |Dc|)]

        'simple':
            rt = ΔAcc / max(prev_acc, ε)
        """
        delta_acc = new_acc - prev_acc

        if self.reward_formula == 'simple':
            return delta_acc / max(prev_acc, 1e-8)

        # 'full' formula
        kl_divergence = self.compute_kl_divergence(client_class_dist,
                                                    self.global_class_dist)
        participation_factor = 1 + self.beta * client_part_freq
        kl_factor = 1 + self.alpha * kl_divergence
        size_factor = np.log(1 + client_size)
        denominator = participation_factor * kl_factor * size_factor
        if denominator == 0:
            denominator = 1e-8
        return delta_acc / denominator

    def step(self, selected_client_indexes, prev_acc, new_acc,
             client_distributions, participation_freq, client_sizes):
        """Compute mean reward over selected clients (convenience method)."""
        rewards = [
            self.compute_reward(
                prev_acc, new_acc,
                client_distributions[c],
                participation_freq[c],
                client_sizes[c],
            )
            for c in selected_client_indexes
        ]
        return float(np.mean(rewards))
