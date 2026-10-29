"""
Simple Tabular Q-Learning Agent for Stock Trading.
Uses a 6-State Q-Table that is intuitive and easy to present to faculty.
"""

from typing import Dict, Tuple
import numpy as np


class QLearningAgent:
    """
    Classical Tabular Q-Learning.

    States (Total 6):
        Market Trend: Down (-1), Flat (0), Up (+1)
        Position    : In Cash (0), Holding Stock (1)
        -> 3 x 2 = 6 total states!

    Actions (Total 3):
        0: HOLD
        1: BUY
        2: SELL
    """

    STATE_NAMES = {
        (-1, 0): "Market DOWN  | In CASH ",
        (-1, 1): "Market DOWN  | In STOCK",
        ( 0, 0): "Market FLAT  | In CASH ",
        ( 0, 1): "Market FLAT  | In STOCK",
        ( 1, 0): "Market UP    | In CASH ",
        ( 1, 1): "Market UP    | In STOCK",
    }
    ACTION_NAMES = ["HOLD", "BUY", "SELL"]

    def __init__(
        self,
        action_dim: int = 3,
        alpha: float = 0.1,         # Learning rate
        gamma: float = 0.95,        # Discount factor
        epsilon: float = 1.0,       # Initial exploration rate
        epsilon_min: float = 0.01,  # Minimum exploration rate
        epsilon_decay: float = 0.985,  # Decay rate per episode
    ):
        self.action_dim = action_dim
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        # Initialize Q-table: dictionary mapping (trend, position) -> [Q(HOLD), Q(BUY), Q(SELL)]
        self.q_table: Dict[Tuple[int, int], np.ndarray] = {
            state: np.zeros(self.action_dim, dtype=np.float32)
            for state in self.STATE_NAMES.keys()
        }

    def discretize_state(self, obs: np.ndarray) -> Tuple[int, int]:
        """Maps 2D observation [trend, position] to (trend_int, position_int)."""
        trend = int(round(obs[0]))
        position = int(round(obs[1]))
        # Clamp to valid ranges
        trend = max(-1, min(1, trend))
        position = max(0, min(1, position))
        return (trend, position)

    def act(self, obs: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        state = self.discretize_state(obs)

        if training and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_dim))

        q_values = self.q_table[state]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return int(np.random.choice(best_actions))

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """
        Standard Bellman equation update:
        Q(s, a) <- Q(s, a) + alpha * [reward + gamma * max_a' Q(s', a') - Q(s, a)]
        """
        state = self.discretize_state(obs)
        next_state = self.discretize_state(next_obs)

        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            max_next_q = np.max(self.q_table[next_state])
            target = reward + (self.gamma * max_next_q)

        td_error = target - current_q
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def print_q_table(self):
        """Prints the entire 6-state Q-table in a clean format to show faculty."""
        print("\n" + "=" * 65)
        print("               LEARNED TABULAR Q-TABLE (6 STATES)")
        print("=" * 65)
        print(f"{'State Description':<26} | {'Q(HOLD)':>10} | {'Q(BUY)':>10} | {'Q(SELL)':>10} | {'Best Action':<10}")
        print("-" * 65)
        for state, name in self.STATE_NAMES.items():
            q_vals = self.q_table[state]
            best_act = self.ACTION_NAMES[np.argmax(q_vals)]
            print(
                f"{name:<26} | {q_vals[0]:10.2f} | {q_vals[1]:10.2f} | {q_vals[2]:10.2f} | -> {best_act:<7}"
            )
        print("=" * 65 + "\n")

    def save(self, filepath: str):
        np.save(filepath, self.q_table, allow_pickle=True)

    def load(self, filepath: str):
        self.q_table = np.load(filepath, allow_pickle=True).item()
