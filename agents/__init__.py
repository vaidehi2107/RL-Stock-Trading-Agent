"""
RL Agents package containing Tabular Q-Learning and Deep Q-Network (DQN).
"""
from agents.q_learning_agent import QLearningAgent

try:
    from agents.dqn_agent import DQNAgent
    __all__ = ["QLearningAgent", "DQNAgent"]
except ImportError:
    __all__ = ["QLearningAgent"]
