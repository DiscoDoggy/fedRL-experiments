import numpy as np


class FL_Environment:
    """Federated Learning Environment for RL-based client selection.

    Parameters
    ----------
    reward_formula : str
        'full'       — rt = ΔAcc / [(1 + β·fc)(1 + α·DKL)(log(1 + |Dc|))]
        'simple'     — rt = ΔAcc / max(prev_acc, ε)
        'fairness'   — rt = (ΔAcc / prev_acc) · (1 + γ·max(0, ā − aᵢ)) − β·fc − α·DKL
        'per_client' — rt_i = (local_delta / prev_acc) · (1 + γ·max(0, ā − aᵢ)) − β·fc_i
        'kl_capped'  — rt_i = (local_delta / prev_acc) · (1 + γ·max(0, ā − aᵢ)) − α·DKL − β·fc_i
        'rank_ema'   — rt_i = (local_delta / prev_acc) · (1 + γ·(1 − rank_MA_i)) − β·fc_i
                        where rank_MA_i is a smoothed percentile rank (0=best, 1=worst)
                        maintained via exponential moving average across rounds.
    gamma : float
        Fairness pressure. Higher values push the agent more aggressively
        toward under-performing clients. Default 2.0.
    """

    def __init__(self, num_clients, global_class_dist, alpha=0.3, beta=0.2,
                 reward_formula: str = 'full', gamma: float = 2.0):
        self.num_clients = num_clients
        self.global_class_dist = global_class_dist
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        valid_formulas = ('full', 'simple', 'fairness', 'per_client', 'kl_capped', 'rank_ema')
        if reward_formula not in valid_formulas:
            raise ValueError(f"reward_formula must be one of {valid_formulas}")
        self.reward_formula = reward_formula
        # Rank EMA tracking for 'rank_ema' formula
        self.client_rank_ema = np.full(num_clients, 0.5, dtype=np.float32)
        self.rank_ema_alpha = 0.3  # EMA smoothing factor

    def get_state(self):
        return self.global_class_dist

    def update_history(self, per_client_accs):
        """Update rank EMA with latest per-client accuracies.

        Call this each round after evaluation, before computing rewards.
        """
        if self.reward_formula != 'rank_ema':
            return
        sorted_indices = np.argsort(per_client_accs)
        # Normalized rank: 0 = worst client, 1 = best client
        ranks = np.zeros(self.num_clients, dtype=np.float32)
        for rank_pos, client_idx in enumerate(sorted_indices):
            ranks[client_idx] = rank_pos / max(self.num_clients - 1, 1)
        a = self.rank_ema_alpha
        self.client_rank_ema = a * ranks + (1 - a) * self.client_rank_ema

    def compute_kl_divergence(self, client_dist, global_dist):
        """KL(client_dist || global_dist)."""
        epsilon = 1e-8
        client_dist = np.array(client_dist) + epsilon
        global_dist = np.array(global_dist) + epsilon
        client_dist = client_dist / np.sum(client_dist)
        global_dist = global_dist / np.sum(global_dist)
        return float(np.sum(client_dist * np.log(client_dist / global_dist)))

    def compute_reward(self, prev_acc, new_acc, client_class_dist,
                       client_part_freq, client_size,
                       client_acc=None, mean_acc=None,
                       client_local_delta=None, client_id=None):
        """Compute per-client reward.

        The formula used depends on ``self.reward_formula``:

        'full':
            rt = ΔAcc / [(1 + β·fc) · (1 + α·DKL(Pc‖Pg)) · log(1 + |Dc|)]

        'simple':
            rt = ΔAcc / max(prev_acc, ε)

        'fairness':
            rt = (ΔAcc / prev_acc) · (1 + γ · max(0, mean_acc − client_acc))
                 − β·fc − α·DKL(Pc‖Pg)
            Subtractive penalties (previously multiplicative). Requires client_acc
            and mean_acc to be supplied.

        'per_client':
            rt_i = (local_delta / prev_acc) · (1 + γ · max(0, ā − aᵢ))  −  β · fc_i
            Requires client_acc, mean_acc, and client_local_delta.

        'kl_capped':
            rt_i = (local_delta / prev_acc) · (1 + γ · max(0, ā − aᵢ))
                   − α · DKL(Pc‖Pg)  −  β · fc_i
            Requires client_acc, mean_acc, and client_local_delta.

        'rank_ema':
            rt_i = (local_delta / prev_acc) · (1 + γ · (1 − rank_MA_i)) − β · fc_i
            Uses exponential moving average of percentile rank (0=best, 1=worst)
            instead of single-round accuracy gap. Requires client_acc, mean_acc,
            client_local_delta, and client_id.
        """
        delta_acc = new_acc - prev_acc

        if self.reward_formula == 'simple':
            return delta_acc / max(prev_acc, 1e-8)

        kl_divergence = self.compute_kl_divergence(client_class_dist,
                                                    self.global_class_dist)
        participation_factor = 1 + self.beta * client_part_freq
        kl_factor = 1 + self.alpha * kl_divergence

        if self.reward_formula == 'fairness':
            # Amplify reward for below-average clients to close accuracy gap
            # Subtractive penalties (not multiplicative) to keep reward signal strong
            if client_acc is None or mean_acc is None:
                raise ValueError(
                    "'fairness' reward requires client_acc and mean_acc"
                )
            gap = max(0.0, mean_acc - client_acc)
            base = (delta_acc / max(prev_acc, 1e-8)) * (1 + self.gamma * gap)
            return base - self.beta * client_part_freq - self.alpha * kl_divergence

        if self.reward_formula in ('per_client', 'kl_capped', 'rank_ema'):
            if any(v is None for v in (client_acc, mean_acc, client_local_delta)):
                raise ValueError(
                    f"'{self.reward_formula}' reward requires "
                    f"client_acc, mean_acc, and client_local_delta"
                )
            gap = max(0.0, mean_acc - client_acc)
            if self.reward_formula == 'rank_ema':
                fairness_boost = 1 + self.gamma * (1 - self.client_rank_ema[client_id])
            else:
                fairness_boost = 1 + self.gamma * gap
            base = (client_local_delta / max(prev_acc, 1e-8)) * fairness_boost
            penalty = self.beta * client_part_freq
            if self.reward_formula == 'kl_capped':
                penalty += self.alpha * kl_divergence
            return base - penalty

        # 'full' formula
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
