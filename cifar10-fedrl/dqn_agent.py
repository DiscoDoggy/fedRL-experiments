import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class DQN(nn.Module):
    """Neural Network for Q-learning."""
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class DQN_Agent:
    """DQN Agent for Client Selection (Double DQN)."""
    def __init__(self, state_size, action_size):
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss — more stable with noisy rewards

        self.epsilon = 0.5
        self.target_update_freq = 10  # sync target network every N rounds

    def update_target(self):
        """Copy online network weights to target network."""
        self.target_model.load_state_dict(self.model.state_dict())

    def select_clients(self, state, num_clients, k=60):
        """Select clients using epsilon-greedy strategy."""
        if np.random.rand() < self.epsilon:
            return random.sample(range(num_clients), k)
        with torch.no_grad():
            q_values = self.model(torch.tensor(state, dtype=torch.float32))
            return q_values.argsort(descending=True)[:k].tolist()

    def train(self, state, action, reward, next_state):
        """Train the Q-network (Double DQN).

        Online network selects the best action for next_state,
        target network evaluates Q-value of that action.
        """
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        # Double DQN: online net picks the action, target net evaluates it
        with torch.no_grad():
            best_action = self.model(next_state).argmax().item()
            target_q = reward + 0.9 * self.target_model(next_state)[best_action]

        q_value = self.model(state)[action]

        loss = self.loss_fn(q_value, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
