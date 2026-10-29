"""
Master CLI Entry Point for Reinforcement Learning Stock Trading System.
Supports Tabular Q-Learning, Deep Q-Network (DQN), and Benchmarking against Buy & Hold.
"""

import argparse
import os
import sys
from data.download_data import load_or_fetch_stock_data, train_test_split_stock
from env.stock_trading_env import StockTradingEnv
from utils.metrics import format_metrics_table
from utils.visualizer import plot_portfolio_comparison, plot_trading_actions, plot_training_progress


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reinforcement Learning Stock Trading Agent (Q-Learning & DQN)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "train-ql", "train-dqn", "compare"],
        help="Execution mode: all (train both & compare), train-ql, train-dqn, or compare",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock ticker symbol (e.g., AAPL, MSFT, NVDA, SPY)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force generation of synthetic GBM market data (offline mode)",
    )
    parser.add_argument(
        "--ql-episodes",
        type=int,
        default=50,
        help="Training episodes for Tabular Q-Learning (default: 50)",
    )
    parser.add_argument(
        "--dqn-episodes",
        type=int,
        default=30,
        help="Training episodes for Deep Q-Network (default: 30)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save plots and reports",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("\n" + "=" * 70)
    print("      REINFORCEMENT LEARNING STOCK TRADING SYSTEM")
    print(f"      Mode: {args.mode.upper()} | Ticker: {args.ticker}")
    print("=" * 70 + "\n")

    # Step 1: Load and prepare data
    df = load_or_fetch_stock_data(ticker=args.ticker, force_synthetic=args.synthetic)
    train_df, test_df = train_test_split_stock(df, train_ratio=0.8)

    print(f"[*] Total market data points: {len(df)}")
    print(f"[*] Training partition: {len(train_df)} trading days")
    print(f"[*] Out-of-sample test partition: {len(test_df)} trading days\n")

    ql_model_path = "models/q_learning_model.npy"
    dqn_model_path = "models/dqn_model.pt"

    from train_qlearning import evaluate_agent as eval_ql, run_buy_and_hold, train_q_learning
    from train_dqn import evaluate_dqn, train_dqn

    # Mode: train-ql
    if args.mode in ["all", "train-ql"]:
        print("[STEP 1/3] Training Tabular Q-Learning Agent...")
        ql_agent, ql_rewards, ql_epsilons = train_q_learning(
            train_df=train_df,
            episodes=args.ql_episodes,
            save_path=ql_model_path,
        )
        plot_training_progress(
            ql_rewards,
            ql_epsilons,
            title="Tabular Q-Learning Training Curve",
            save_path=os.path.join(args.output_dir, "q_learning_training.png"),
        )

    # Mode: train-dqn
    if args.mode in ["all", "train-dqn"]:
        print("\n[STEP 2/3] Training Deep Q-Network (DQN) Agent...")
        dqn_agent, dqn_rewards, dqn_epsilons = train_dqn(
            train_df=train_df,
            episodes=args.dqn_episodes,
            save_path=dqn_model_path,
        )
        plot_training_progress(
            dqn_rewards,
            dqn_epsilons,
            title="Deep Q-Network Training Curve",
            save_path=os.path.join(args.output_dir, "dqn_training.png"),
        )

    # Mode: compare / evaluation
    if args.mode in ["all", "compare"]:
        print("\n[STEP 3/3] Evaluating and Comparing Strategies on Test Data...")
        from agents.q_learning_agent import QLearningAgent
        from agents.dqn_agent import DQNAgent

        # Load models
        ql_agent = QLearningAgent(action_dim=3)
        ql_agent.load(ql_model_path)

        dqn_agent = DQNAgent(state_dim=2, action_dim=3)
        dqn_agent.load(dqn_model_path)

        # Print Q-Table
        ql_agent.print_q_table()

        # Evaluate
        ql_metrics, ql_portfolio, ql_trades = eval_ql(ql_agent, test_df)
        dqn_metrics, dqn_portfolio, dqn_trades = evaluate_dqn(dqn_agent, test_df)
        bh_metrics, bh_portfolio = run_buy_and_hold(test_df)

        comparison_table = {
            "Q-Learning (Tabular)": ql_metrics,
            "Deep Q-Network (DQN)": dqn_metrics,
            "Buy & Hold (Baseline)": bh_metrics,
        }

        print("\n" + "=" * 70)
        print("                 OUT-OF-SAMPLE TEST PERFORMANCE")
        print("=" * 70)
        print(format_metrics_table(comparison_table))
        print("=" * 70)

        # Plot equity curves
        plot_portfolio_comparison(
            {
                "Q-Learning": ql_portfolio,
                "Deep Q-Network": dqn_portfolio,
                "Buy & Hold": bh_portfolio,
            },
            title=f"Comparative Net Worth Growth ({args.ticker} Test Period)",
            save_path=os.path.join(args.output_dir, "comparison_equity_curve.png"),
        )

        # Plot individual trading action charts
        plot_trading_actions(
            test_df,
            ql_trades,
            title=f"Q-Learning Trading Execution on {args.ticker}",
            save_path=os.path.join(args.output_dir, "q_learning_trades.png"),
        )
        plot_trading_actions(
            test_df,
            dqn_trades,
            title=f"DQN Trading Execution on {args.ticker}",
            save_path=os.path.join(args.output_dir, "dqn_trades.png"),
        )

        print(f"\n[+] Visualizations successfully saved to '{args.output_dir}/'!")


if __name__ == "__main__":
    main()
