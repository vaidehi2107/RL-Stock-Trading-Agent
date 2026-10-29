"""
Simple Stock Trading Environment.
Designed for clarity and ease of academic presentation.

- State (2 values): [Market Trend (-1, 0, 1), Position Status (0 or 1)]
- Actions (3 discrete): 0 = HOLD, 1 = BUY, 2 = SELL
- Reward: Daily change in total net worth percentage
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


class StockTradingEnv:
    """
    A simplified single-stock trading environment.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_fee_pct: float = 0.001,  # 0.1% fee per transaction
    ):
        self.df = df.reset_index(drop=True)
        self.initial_balance = float(initial_balance)
        self.transaction_fee_pct = float(transaction_fee_pct)

        self.prices = self.df['Close'].values.astype(np.float64)
        self.trends = self.df['Trend'].values.astype(np.int32)           # -1 (Down), 0 (Flat), 1 (Up)
        self.returns = self.df['Daily_Return'].values.astype(np.float32)  # Continuous return
        self.n_steps = len(self.df)

        # State dimensions: [Trend, Position] -> 2 features
        self.state_dim = 2
        self.action_dim = 3  # 0: HOLD, 1: BUY, 2: SELL

        # Internal state variables
        self.current_step = 0
        self.cash = self.initial_balance
        self.shares = 0
        self.cost_basis = 0.0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance

        # Track history for plotting
        self.portfolio_history: List[float] = []
        self.trades_history: List[Dict[str, Any]] = []

    def reset(self) -> np.ndarray:
        """Reset the environment to start a new episode."""
        self.current_step = 0
        self.cash = self.initial_balance
        self.shares = 0
        self.cost_basis = 0.0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance

        self.portfolio_history = [self.initial_balance]
        self.trades_history = []

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """
        Returns a simple 2D state vector:
        1. Trend: Market direction (-1 = Down, 0 = Flat, 1 = Up)
        2. Position: 1 if currently holding stock, 0 if in cash
        """
        trend = float(self.trends[self.current_step])
        position = 1.0 if self.shares > 0 else 0.0
        return np.array([trend, position], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one action in the market:
        - 0 (HOLD): Do nothing
        - 1 (BUY) : Use cash to buy shares
        - 2 (SELL): Liquidate shares to cash
        """
        current_price = self.prices[self.current_step]
        executed_trade = None

        # ACTION 1: BUY
        if action == 1 and self.cash > current_price:
            effective_cash = self.cash / (1.0 + self.transaction_fee_pct)
            shares_to_buy = int(effective_cash // current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                fee = cost * self.transaction_fee_pct
                self.cash -= (cost + fee)
                self.shares += shares_to_buy
                self.cost_basis = current_price
                executed_trade = {
                    "step": self.current_step,
                    "type": "BUY",
                    "price": current_price,
                    "shares": shares_to_buy,
                }

        # ACTION 2: SELL
        elif action == 2 and self.shares > 0:
            revenue = self.shares * current_price
            fee = revenue * self.transaction_fee_pct
            net_revenue = revenue - fee
            pnl = net_revenue - (self.shares * self.cost_basis)

            self.cash += net_revenue
            executed_trade = {
                "step": self.current_step,
                "type": "SELL",
                "price": current_price,
                "shares": self.shares,
                "pnl": pnl,
            }
            self.shares = 0
            self.cost_basis = 0.0

        if executed_trade is not None:
            self.trades_history.append(executed_trade)

        # Move to next trading day
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1

        # Calculate updated portfolio net worth
        next_price = self.prices[self.current_step]
        self.prev_net_worth = self.net_worth
        self.net_worth = self.cash + (self.shares * next_price)
        self.portfolio_history.append(self.net_worth)

        # Reward = Daily percentage change in total net worth
        reward = ((self.net_worth - self.prev_net_worth) / (self.prev_net_worth + 1e-9)) * 100.0

        next_obs = self._get_observation()
        info = {
            "step": self.current_step,
            "net_worth": self.net_worth,
            "shares": self.shares,
            "cash": self.cash,
        }

        return next_obs, float(reward), done, info
