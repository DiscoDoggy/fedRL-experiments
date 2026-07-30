import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np


class DQN(nn.Module):
    """Neural Network for Q-learning."""

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, action_size)

    def forward(self, state):
        return self.fc2(torch.relu(self.fc1(state)))


class DQN_Agent:
    """DQN Agent for client selection.

    Parameters
    ----------
    use_target_network : bool
        False — Single DQN: the online network is used for both action
                selection and target computation (CIFAR-10 setting).
        True  — Double DQN: a separate frozen target network is used for
                target computation (MNIST setting).
    """

    def __init__(self, state_size, action_size, use_target_network: bool = False):
        self.use_target_network = use_target_network
        self.model = DQN(state_size, action_size)
        if use_target_network:
            self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()
        self.epsilon = 0.5
        self.update_rate = 10
        self.round_count = 0

    def select_clients(self, state, num_clients, k=60):
        """Select k clients using epsilon-greedy strategy."""
        if np.random.rand() < self.epsilon:
            return random.sample(range(num_clients), k)
        with torch.no_grad():
            q_values = self.model(torch.tensor(state, dtype=torch.float32))
            return q_values.argsort(descending=True)[:k].tolist()

    def update_target_network(self):
        """Sync target network to online network (hard copy)."""
        if self.use_target_network:
            self.target_model.load_state_dict(self.model.state_dict())

    def train(self, state, action, reward, next_state):
        """Train the Q-network on one (s, a, r, s') transition."""
        self.round_count += 1
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        if self.use_target_network:
            # Double DQN: online network selects action, target network evaluates it
            next_action = self.model(next_state).argmax()
            target_q = reward + 0.9 * self.target_model(next_state)[next_action]
        else:
            target_q = reward + 0.9 * self.model(next_state).max()

        q_value = self.model(state)[action]
        loss = self.loss_fn(q_value, torch.tensor(target_q))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
