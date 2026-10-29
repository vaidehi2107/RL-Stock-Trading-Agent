"""
Comparative evaluation script: Tabular Q-Learning vs Deep Q-Network (DQN) vs Buy & Hold.
Runs all agents on identical test data, prints metrics and the 6-state Q-table.
"""

import os
from typing import Dict, List, Tuple
import pandas as pd
from agents.dqn_agent import DQNAgent
from agents.q_learning_agent import QLearningAgent
from data.download_data import load_or_fetch_stock_data, train_test_split_stock
from env.stock_trading_env import StockTradingEnv
from utils.metrics import calculate_portfolio_metrics, format_metrics_table
from utils.visualizer import plot_portfolio_comparison, plot_trading_actions


def evaluate_trading_agent(
    agent,
    test_df: pd.DataFrame,
) -> Tuple[Dict[str, float], List[float], List[Dict]]:
    """Evaluates an agent (greedy mode) on test environment."""
    env = StockTradingEnv(test_df)
    obs = env.reset()
    done = False

    while not done:
        action = agent.act(obs, training=False)
        obs, _, done, _ = env.step(action)

    metrics = calculate_portfolio_metrics(env.portfolio_history, env.trades_history)
    return metrics, env.portfolio_history, env.trades_history


def run_buy_and_hold(test_df: pd.DataFrame, initial_balance: float = 10000.0):
    """Executes Buy & Hold baseline."""
    prices = test_df['Close'].values
    day1_price = prices[0]
    shares = int(initial_balance // (day1_price * 1.001))
    cash = initial_balance - (shares * day1_price * 1.001)
    portfolio_values = [cash + (shares * p) for p in prices]
    dummy_trades = [{"step": 0, "type": "BUY", "shares": shares}]
    metrics = calculate_portfolio_metrics(portfolio_values, dummy_trades)
    return metrics, portfolio_values


def run_comparison(
    ticker: str = "AAPL",
    ql_model_path: str = None,
    dqn_model_path: str = None,
    output_dir: str = "results",
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    ticker_clean = ticker.replace("^", "").replace(".", "_")

    if ql_model_path is None:
        ql_model_path = f"models/q_learning_{ticker_clean}.npy"
    if dqn_model_path is None:
        dqn_model_path = f"models/dqn_{ticker_clean}.pt"

    df = load_or_fetch_stock_data(ticker=ticker)
    train_df, test_df = train_test_split_stock(df, train_ratio=0.8)

    # 1. Q-Learning
    ql_agent = QLearningAgent(action_dim=3)
    if os.path.exists(ql_model_path):
        print(f"[*] Loading Q-Learning model from {ql_model_path}...")
        ql_agent.load(ql_model_path)
    else:
        print(f"[!] Q-Learning model not found for {ticker}. Training new model...")
        from train_qlearning import train_q_learning
        ql_agent, _, _ = train_q_learning(train_df, episodes=50, save_path=ql_model_path)

    # Print the 6-state Q-table
    ql_agent.print_q_table()

    # 2. DQN
    dqn_agent = DQNAgent(state_dim=2, action_dim=3)
    if os.path.exists(dqn_model_path):
        print(f"[*] Loading DQN model from {dqn_model_path}...")
        dqn_agent.load(dqn_model_path)
    else:
        print(f"[!] DQN model not found for {ticker}. Training new model...")
        from train_dqn import train_dqn
        dqn_agent, _, _ = train_dqn(train_df, episodes=30, save_path=dqn_model_path)
        from train_dqn import train_dqn
        dqn_agent, _, _ = train_dqn(train_df, episodes=30, save_path=dqn_model_path)

    # 3. Evaluate both agents and baseline
    print("[*] Evaluating strategies on out-of-sample test data...")
    ql_metrics, ql_portfolio, ql_trades = evaluate_trading_agent(ql_agent, test_df)
    dqn_metrics, dqn_portfolio, dqn_trades = evaluate_trading_agent(dqn_agent, test_df)
    bh_metrics, bh_portfolio = run_buy_and_hold(test_df)

    metrics_comparison = {
        "Q-Learning (Tabular)": ql_metrics,
        "Deep Q-Network (DQN)": dqn_metrics,
        "Buy & Hold (Baseline)": bh_metrics,
    }

    print("\n" + "=" * 70)
    print("           FINAL OUT-OF-SAMPLE BACKTEST COMPARISON")
    print("=" * 70)
    print(format_metrics_table(metrics_comparison))
    print("=" * 70)

    # 4. Generate Visualizations
    portfolio_curves = {
        "Q-Learning": ql_portfolio,
        "Deep Q-Network": dqn_portfolio,
        "Buy & Hold": bh_portfolio,
    }
    plot_portfolio_comparison(
        portfolio_curves,
        title=f"Net Worth Growth: RL Agents vs Buy & Hold ({ticker})",
        save_path=os.path.join(output_dir, "comparison_equity_curve.png"),
    )

    plot_trading_actions(
        test_df,
        ql_trades,
        title=f"Q-Learning Trading Signals on {ticker}",
        save_path=os.path.join(output_dir, "q_learning_trades.png"),
    )

    plot_trading_actions(
        test_df,
        dqn_trades,
        title=f"DQN Trading Signals on {ticker}",
        save_path=os.path.join(output_dir, "dqn_trades.png"),
    )

    print(f"\n[+] Visualizations successfully saved to '{output_dir}/'.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare RL Agents vs Buy & Hold")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (e.g., TSLA, MSFT, GOOGL, NVDA)")
    args = parser.parse_args()
    run_comparison(ticker=args.ticker)
