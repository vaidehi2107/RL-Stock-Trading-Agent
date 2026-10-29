"""
Simple Training and Evaluation Pipeline for Tabular Q-Learning.
Prints the learned 6-state Q-table so you can explain it to faculty.
"""

import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from agents.q_learning_agent import QLearningAgent
from data.download_data import load_or_fetch_stock_data, train_test_split_stock
from env.stock_trading_env import StockTradingEnv
from utils.metrics import calculate_portfolio_metrics, format_metrics_table
from utils.visualizer import plot_portfolio_comparison, plot_trading_actions, plot_training_progress


def train_q_learning(
    train_df: pd.DataFrame,
    episodes: int = 50,
    alpha: float = 0.1,
    gamma: float = 0.95,
    save_path: str = "models/q_learning_model.npy",
) -> Tuple[QLearningAgent, List[float], List[float]]:
    """Trains Tabular Q-Learning on training data."""
    env = StockTradingEnv(train_df)
    agent = QLearningAgent(action_dim=env.action_dim, alpha=alpha, gamma=gamma)

    rewards_history = []
    epsilons_history = []

    print("=" * 60)
    print(f"Training Simple Tabular Q-Learning ({episodes} episodes)...")
    print("=" * 60)

    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.act(obs, training=True)
            next_obs, reward, done, _ = env.step(action)
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward

        agent.decay_epsilon()
        rewards_history.append(total_reward)
        epsilons_history.append(agent.epsilon)

        if ep % 10 == 0 or ep == episodes:
            avg_rew = np.mean(rewards_history[-10:])
            print(f"Episode {ep:2d}/{episodes} | Avg Reward: {avg_rew:7.2f} | Epsilon: {agent.epsilon:.3f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)

    # Print the clean 6-state Q-table to show faculty!
    agent.print_q_table()

    return agent, rewards_history, epsilons_history


def evaluate_agent(
    agent: QLearningAgent,
    test_df: pd.DataFrame,
) -> Tuple[Dict[str, float], List[float], List[Dict]]:
    """Evaluates agent on test data with greedy policy."""
    env = StockTradingEnv(test_df)
    obs = env.reset()
    done = False

    while not done:
        action = agent.act(obs, training=False)
        obs, _, done, _ = env.step(action)

    metrics = calculate_portfolio_metrics(env.portfolio_history, env.trades_history)
    return metrics, env.portfolio_history, env.trades_history


def run_buy_and_hold(test_df: pd.DataFrame, initial_balance: float = 10000.0):
    """Buy and hold baseline."""
    prices = test_df['Close'].values
    day1_price = prices[0]
    shares = int(initial_balance // (day1_price * 1.001))
    cash = initial_balance - (shares * day1_price * 1.001)
    portfolio_values = [cash + (shares * p) for p in prices]
    metrics = calculate_portfolio_metrics(portfolio_values, [{"step": 0, "type": "BUY", "shares": shares}])
    return metrics, portfolio_values


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Tabular Q-Learning on a stock")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker (e.g., TSLA, MSFT, NVDA)")
    parser.add_argument("--episodes", type=int, default=50, help="Training episodes")
    args = parser.parse_args()

    ticker_clean = args.ticker.replace("^", "").replace(".", "_")
    save_model_path = f"models/q_learning_{ticker_clean}.npy"

    df = load_or_fetch_stock_data(ticker=args.ticker)
    train_df, test_df = train_test_split_stock(df, train_ratio=0.8)

    agent, rewards, epsilons = train_q_learning(train_df, episodes=args.episodes, save_path=save_model_path)

    plot_training_progress(rewards, epsilons, save_path="results/q_learning_training.png")

    ql_metrics, ql_portfolio, ql_trades = evaluate_agent(agent, test_df)
    bh_metrics, bh_portfolio = run_buy_and_hold(test_df)

    print(f"\nOUT-OF-SAMPLE TEST PERFORMANCE ({args.ticker}):")
    print(format_metrics_table({"Q-Learning": ql_metrics, "Buy & Hold": bh_metrics}))

    plot_trading_actions(test_df, ql_trades, title=f"Q-Learning Trades ({args.ticker})", save_path="results/q_learning_trades.png")
    plot_portfolio_comparison(
        {"Q-Learning": ql_portfolio, "Buy & Hold": bh_portfolio},
        title=f"Portfolio: Q-Learning vs Buy & Hold ({args.ticker})",
        save_path="results/q_learning_equity.png"
    )
