"""
Simple Deep Q-Network (DQN) Agent for Stock Trading.
Uses a lightweight 2-layer Neural Network that is easy to explain to faculty.
"""

import collections
import random
from typing import Deque, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SimpleQNetwork(nn.Module):
    """
    Lightweight 2-layer Neural Network approximating Q(s, a).
    Input (2 features) -> Hidden (32) -> Hidden (32) -> Output (3 actions)
    """

    def __init__(self, state_dim: int = 2, action_dim: int = 3):
        super(SimpleQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class ReplayBuffer:
    """
    Replay buffer storing transitions: (state, action, reward, next_state, done).
    Breaks correlation in time series data.
    """

    def __init__(self, capacity: int = 10000):
        self.buffer: Deque[Tuple] = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network (DQN) with Experience Replay and Target Network.
    """

    def __init__(
        self,
        state_dim: int = 2,
        action_dim: int = 3,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.95,
        batch_size: int = 32,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = torch.device("cpu")

        # 1. Main Policy Network
        self.policy_net = SimpleQNetwork(state_dim, action_dim).to(self.device)
        # 2. Target Network for stable Q-targets
        self.target_net = SimpleQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(capacity=10000)

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action choice."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self) -> float:
        """Sample mini-batch and do one gradient descent step."""
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size, self.device
        )

        # Current Q prediction: Q(s, a)
        current_q = self.policy_net(states).gather(1, actions)

        # Target Q from target network: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + (1.0 - dones) * (self.gamma * max_next_q)

        # Mean Squared Error Loss
        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically soft-update target network
        for target_p, policy_p in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_p.data.copy_(0.01 * policy_p.data + 0.99 * target_p.data)

        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        torch.save(self.policy_net.state_dict(), filepath)

    def load(self, filepath: str):
        self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
