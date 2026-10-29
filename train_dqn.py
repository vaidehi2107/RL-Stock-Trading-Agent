"""
Simple Training and Evaluation Pipeline for Deep Q-Network (DQN).
Uses a 2-feature input and 2-layer Neural Network.
"""

import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from agents.dqn_agent import DQNAgent
from data.download_data import load_or_fetch_stock_data, train_test_split_stock
from env.stock_trading_env import StockTradingEnv
from utils.metrics import calculate_portfolio_metrics, format_metrics_table
from utils.visualizer import plot_portfolio_comparison, plot_trading_actions, plot_training_progress


def train_dqn(
    train_df: pd.DataFrame,
    episodes: int = 30,
    learning_rate: float = 0.001,
    gamma: float = 0.95,
    save_path: str = "models/dqn_model.pt",
) -> Tuple[DQNAgent, List[float], List[float]]:
    """Trains Deep Q-Network Agent on training data."""
    env = StockTradingEnv(train_df)
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=32,
    )

    rewards_history = []
    epsilons_history = []

    print("=" * 60)
    print(f"Training Simple Deep Q-Network (DQN) for {episodes} episodes...")
    print(f"Network: Input(2) -> Hidden(32) -> Hidden(32) -> Output(3)")
    print("=" * 60)

    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            action = agent.act(obs, training=True)
            next_obs, reward, done, _ = env.step(action)
            agent.remember(obs, action, reward, next_obs, done)

            if step_count % 4 == 0:
                agent.train_step()

            obs = next_obs
            total_reward += reward
            step_count += 1

        agent.decay_epsilon()
        rewards_history.append(total_reward)
        epsilons_history.append(agent.epsilon)

        if ep % 5 == 0 or ep == episodes:
            avg_rew = np.mean(rewards_history[-5:])
            print(
                f"Episode {ep:2d}/{episodes} | Avg Reward: {avg_rew:7.2f} | "
                f"Epsilon: {agent.epsilon:.3f} | Net Worth: ${env.net_worth:,.2f}"
            )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    print(f"[+] Saved trained DQN model weights to {save_path}")

    return agent, rewards_history, epsilons_history


def evaluate_dqn(
    agent: DQNAgent,
    test_df: pd.DataFrame,
) -> Tuple[Dict[str, float], List[float], List[Dict]]:
    """Evaluates trained DQN on test data."""
    env = StockTradingEnv(test_df)
    obs = env.reset()
    done = False

    while not done:
        action = agent.act(obs, training=False)
        obs, _, done, _ = env.step(action)

    metrics = calculate_portfolio_metrics(env.portfolio_history, env.trades_history)
    return metrics, env.portfolio_history, env.trades_history


def run_buy_and_hold(test_df: pd.DataFrame, initial_balance: float = 10000.0):
    prices = test_df['Close'].values
    day1_price = prices[0]
    shares = int(initial_balance // (day1_price * 1.001))
    cash = initial_balance - (shares * day1_price * 1.001)
    portfolio_values = [cash + (shares * p) for p in prices]
    metrics = calculate_portfolio_metrics(portfolio_values, [{"step": 0, "type": "BUY", "shares": shares}])
    return metrics, portfolio_values


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train DQN on a stock")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker (e.g., TSLA, MSFT, NVDA)")
    parser.add_argument("--episodes", type=int, default=30, help="Training episodes")
    args = parser.parse_args()

    ticker_clean = args.ticker.replace("^", "").replace(".", "_")
    save_model_path = f"models/dqn_{ticker_clean}.pt"

    df = load_or_fetch_stock_data(ticker=args.ticker)
    train_df, test_df = train_test_split_stock(df, train_ratio=0.8)

    agent, rewards, epsilons = train_dqn(train_df, episodes=args.episodes, save_path=save_model_path)

    plot_training_progress(rewards, epsilons, title=f"DQN Training Progress ({args.ticker})", save_path="results/dqn_training.png")

    dqn_metrics, dqn_portfolio, dqn_trades = evaluate_dqn(agent, test_df)
    bh_metrics, bh_portfolio = run_buy_and_hold(test_df)

    print(f"\nDQN OUT-OF-SAMPLE TEST PERFORMANCE ({args.ticker}):")
    print(format_metrics_table({"DQN": dqn_metrics, "Buy & Hold": bh_metrics}))

    plot_trading_actions(test_df, dqn_trades, title=f"DQN Trades ({args.ticker})", save_path="results/dqn_trades.png")
    plot_portfolio_comparison(
        {"DQN": dqn_portfolio, "Buy & Hold": bh_portfolio},
        title=f"Portfolio: DQN vs Buy & Hold ({args.ticker})",
        save_path="results/dqn_equity.png"
    )
