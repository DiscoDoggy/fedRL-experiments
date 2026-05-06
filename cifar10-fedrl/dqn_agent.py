import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class ReplayBuffer:
    """Fixed-size circular replay buffer."""
    def __init__(self, capacity=2000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((np.array(state, dtype=np.float32),
                            int(action),
                            float(reward),
                            np.array(next_state, dtype=np.float32)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (np.stack(states),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.stack(next_states))

    def __len__(self):
        return len(self.buffer)

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
    """DQN Agent for Client Selection (Double DQN + Replay Buffer)."""
    def __init__(self, state_size, action_size):
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        # Epsilon-greedy with linear decay
        self.epsilon = 0.8
        self.epsilon_min = 0.05
        self.epsilon_decay_rounds = 150
        self.round = 0

        self.target_update_freq = 10

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=2000)
        self.batch_size = 32
        self.min_buffer_size = 64  # wait until buffer has enough transitions

    def update_target(self):
        """Copy online network weights to target network."""
        self.target_model.load_state_dict(self.model.state_dict())

    def update_epsilon(self):
        """Linear epsilon decay."""
        frac = min(self.round / self.epsilon_decay_rounds, 1.0)
        self.epsilon = self.epsilon - frac * (self.epsilon - self.epsilon_min)

    def push(self, state, action, reward, next_state):
        """Store a single (s, a, r, s') transition."""
        self.replay_buffer.push(state, action, reward, next_state)

    def select_clients(self, state, num_clients, k=60):
        """Select clients using epsilon-greedy strategy."""
        if np.random.rand() < self.epsilon:
            return random.sample(range(num_clients), k)
        with torch.no_grad():
            q_values = self.model(torch.tensor(state, dtype=torch.float32))
            return q_values.argsort(descending=True)[:k].tolist()

    def train(self):
        """Sample a mini-batch from replay buffer and train (Double DQN)."""
        if len(self.replay_buffer) < self.min_buffer_size:
            return

        states, actions, rewards, next_states = self.replay_buffer.sample(self.batch_size)

        states      = torch.tensor(states,      dtype=torch.float32)
        actions     = torch.tensor(actions,     dtype=torch.long)
        rewards     = torch.tensor(rewards,     dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        # Double DQN: online net selects action, target net evaluates Q-value
        with torch.no_grad():
            best_actions = self.model(next_states).argmax(dim=1)          # [B]
            target_qs = rewards + 0.9 * self.target_model(next_states) \
                            .gather(1, best_actions.unsqueeze(1)).squeeze(1)  # [B]

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]

        loss = self.loss_fn(q_values, target_qs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
