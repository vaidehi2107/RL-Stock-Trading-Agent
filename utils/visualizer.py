"""
Visualization utilities for trading actions, equity curves, and training progress.
"""

import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_trading_actions(
    df: pd.DataFrame,
    trades: List[Dict],
    title: str = "Trading Execution (Buy / Sell Signals)",
    save_path: Optional[str] = None,
):
    """
    Plots the stock price with Buy and Sell markers overlaid.
    """
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Close'], label='Close Price', color='#2c3e50', alpha=0.75, linewidth=1.5)

    buy_dates, buy_prices = [], []
    sell_dates, sell_prices = [], []

    for t in trades:
        step_idx = t['step']
        if step_idx < len(df):
            date_val = df.index[step_idx]
            price = t['price']
            if t['type'] == 'BUY':
                buy_dates.append(date_val)
                buy_prices.append(price)
            elif t['type'] == 'SELL':
                sell_dates.append(date_val)
                sell_prices.append(price)

    if buy_dates:
        plt.scatter(buy_dates, buy_prices, marker='^', color='#2ecc71', s=120, label='Buy Action', zorder=5)
    if sell_dates:
        plt.scatter(sell_dates, sell_prices, marker='v', color='#e74c3c', s=120, label='Sell Action', zorder=5)

    plt.title(title, fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Date / Step', fontsize=11)
    plt.ylabel('Stock Price ($)', fontsize=11)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_portfolio_comparison(
    portfolio_histories: Dict[str, List[float]],
    title: str = "Portfolio Value Comparison: RL Agents vs Benchmark",
    save_path: Optional[str] = None,
):
    """
    Plots the portfolio net worth curves over time for multiple agents/benchmarks.
    """
    plt.figure(figsize=(14, 6))
    colors = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6', '#e74c3c']

    for i, (name, values) in enumerate(portfolio_histories.items()):
        color = colors[i % len(colors)]
        style = '--' if 'Hold' in name else '-'
        plt.plot(values, label=name, color=color, linestyle=style, linewidth=2.0)

    plt.title(title, fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Trading Day', fontsize=11)
    plt.ylabel('Portfolio Net Worth ($)', fontsize=11)
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_training_progress(
    rewards_history: List[float],
    epsilons: Optional[List[float]] = None,
    window: int = 10,
    title: str = "Training Progress (Episode Rewards & Epsilon)",
    save_path: Optional[str] = None,
):
    """
    Plots training reward curves (raw and rolling average) alongside epsilon exploration decay.
    """
    episodes = np.arange(1, len(rewards_history) + 1)
    rewards_s = pd.Series(rewards_history)
    rolling_mean = rewards_s.rolling(window=window, min_periods=1).mean()

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(episodes, rewards_history, alpha=0.3, color='#3498db', label='Raw Episode Reward')
    ax1.plot(episodes, rolling_mean, color='#1f77b4', linewidth=2.0, label=f'{window}-Ep Rolling Avg Reward')
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Total Reward', color='#1f77b4', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.grid(True, linestyle='--', alpha=0.4)

    if epsilons and len(epsilons) == len(rewards_history):
        ax2 = ax1.twinx()
        ax2.plot(episodes, epsilons, color='#e67e22', linestyle=':', linewidth=1.8, label='Epsilon (Exploration)')
        ax2.set_ylabel('Epsilon', color='#e67e22', fontsize=11)
        ax2.tick_params(axis='y', labelcolor='#e67e22')

    plt.title(title, fontsize=13, fontweight='bold', pad=12)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
