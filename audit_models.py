"""
Audit all trained ticker models and check performance metrics.
"""
import os
import pandas as pd
from data.download_data import load_or_fetch_stock_data, train_test_split_stock
from env.stock_trading_env import StockTradingEnv
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from utils.metrics import calculate_portfolio_metrics

tickers = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'GOOGL', 'SPY', 'TATAMOTORS.NS']

print(f"{'Ticker':<14} | {'Market Return':<14} | {'Q-Learning Ret':<16} | {'DQN Ret':<14} | {'QL Orders':<10} | {'DQN Orders':<10}")
print("-" * 90)

for ticker in tickers:
    clean = ticker.replace('^', '').replace('.', '_')
    ql_path = f"models/q_learning_{clean}.npy"
    dqn_path = f"models/dqn_{clean}.pt"
    
    if not os.path.exists(ql_path) or not os.path.exists(dqn_path):
        print(f"{ticker:<14} | Models missing!")
        continue
        
    df = load_or_fetch_stock_data(ticker=ticker)
    train_df, test_df = train_test_split_stock(df, train_ratio=0.8)
    
    # Buy & Hold
    prices = test_df['Close'].values
    day1_p = prices[0]
    bh_shares = int(10000.0 // (day1_p * 1.001))
    bh_cash = 10000.0 - (bh_shares * day1_p * 1.001)
    bh_vals = [bh_cash + (bh_shares * p) for p in prices]
    bh_metrics = calculate_portfolio_metrics(bh_vals, [{'step': 0, 'type': 'BUY', 'shares': bh_shares}])
    
    # Q-Learning
    ql = QLearningAgent(action_dim=3)
    ql.load(ql_path)
    env_ql = StockTradingEnv(test_df)
    obs = env_ql.reset()
    done = False
    while not done:
        a = ql.act(obs, training=False)
        obs, _, done, _ = env_ql.step(a)
    ql_metrics = calculate_portfolio_metrics(env_ql.portfolio_history, env_ql.trades_history)
    
    # DQN
    dqn = DQNAgent(state_dim=2, action_dim=3)
    dqn.load(dqn_path)
    env_dqn = StockTradingEnv(test_df)
    obs = env_dqn.reset()
    done = False
    while not done:
        a = dqn.act(obs, training=False)
        obs, _, done, _ = env_dqn.step(a)
    dqn_metrics = calculate_portfolio_metrics(env_dqn.portfolio_history, env_dqn.trades_history)
    
    bh_ret = f"{bh_metrics['Total Return (%)']:+.2f}%"
    ql_ret = f"{ql_metrics['Total Return (%)']:+.2f}%"
    dqn_ret = f"{dqn_metrics['Total Return (%)']:+.2f}%"
    ql_orders = f"{ql_metrics['Buy Orders']}B/{ql_metrics['Sell Orders']}S"
    dqn_orders = f"{dqn_metrics['Buy Orders']}B/{dqn_metrics['Sell Orders']}S"
    
    print(f"{ticker:<14} | {bh_ret:<14} | {ql_ret:<16} | {dqn_ret:<14} | {ql_orders:<10} | {dqn_orders:<10}")
