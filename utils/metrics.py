"""
Financial performance evaluation metrics for backtesting RL trading agents.
"""

from typing import Dict, List
import numpy as np
import pandas as pd


def calculate_portfolio_metrics(
    portfolio_values: List[float],
    trades: List[Dict],
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = 252,
) -> Dict[str, float]:
    """
    Computes standard quantitative finance metrics given portfolio value history and trades.

    Args:
        portfolio_values: Daily total portfolio net worth over time.
        trades: List of executed trades with 'type', 'price', 'shares', 'pnl'.
        risk_free_rate: Annualized risk-free rate (e.g., 0.0 or 0.02 for 2%).
        trading_days_per_year: Typically 252 for daily equity market.

    Returns:
        Dictionary of performance metrics.
    """
    values = np.array(portfolio_values, dtype=np.float64)
    if len(values) < 2:
        return {
            "Initial Balance": values[0] if len(values) > 0 else 0.0,
            "Final Balance": values[-1] if len(values) > 0 else 0.0,
            "Total Return (%)": 0.0,
            "CAGR (%)": 0.0,
            "Annualized Volatility (%)": 0.0,
            "Sharpe Ratio": 0.0,
            "Max Drawdown (%)": 0.0,
            "Total Trades": 0,
            "Win Rate (%)": 0.0,
            "Profit Factor": 0.0,
        }

    initial_val = values[0]
    final_val = values[-1]
    total_return = (final_val - initial_val) / initial_val

    # Daily returns
    daily_returns = np.diff(values) / values[:-1]
    n_days = len(values)

    # Annualized CAGR
    if initial_val > 0 and final_val > 0 and n_days > 1:
        cagr = (final_val / initial_val) ** (trading_days_per_year / n_days) - 1.0
    else:
        cagr = -1.0

    # Annualized Volatility
    daily_vol = np.std(daily_returns)
    annualized_vol = daily_vol * np.sqrt(trading_days_per_year)

    # Sharpe Ratio
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / trading_days_per_year) - 1.0
    excess_returns = daily_returns - daily_rf
    if daily_vol > 1e-9:
        sharpe_ratio = (np.mean(excess_returns) / daily_vol) * np.sqrt(trading_days_per_year)
    else:
        sharpe_ratio = 0.0

    # Max Drawdown
    cumulative_max = np.maximum.accumulate(values)
    drawdowns = (cumulative_max - values) / cumulative_max
    max_drawdown = np.max(drawdowns)

    # Trade statistics
    buy_orders = sum(1 for t in trades if t.get("type") == "BUY")
    sell_orders = sum(1 for t in trades if t.get("type") == "SELL")
    closed_pnls = [t.get("pnl", 0.0) for t in trades if t.get("type") == "SELL" and "pnl" in t]
    total_closed = len(closed_pnls)

    if total_closed > 0:
        winning_trades = sum(1 for p in closed_pnls if p > 0)
        win_rate = (winning_trades / total_closed) * 100.0
        gross_profit = sum(p for p in closed_pnls if p > 0)
        gross_loss = abs(sum(p for p in closed_pnls if p < 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-9 else (99.0 if gross_profit > 0 else 0.0)
    else:
        win_rate = 0.0
        profit_factor = 0.0

    return {
        "Initial Balance": float(initial_val),
        "Final Balance": float(final_val),
        "Total Return (%)": float(total_return * 100.0),
        "CAGR (%)": float(cagr * 100.0),
        "Annualized Volatility (%)": float(annualized_vol * 100.0),
        "Sharpe Ratio": float(sharpe_ratio),
        "Max Drawdown (%)": float(max_drawdown * 100.0),
        "Buy Orders": int(buy_orders),
        "Sell Orders": int(sell_orders),
        "Win Rate (%)": float(win_rate),
        "Profit Factor": float(profit_factor),
    }


def format_metrics_table(metrics_dict: Dict[str, Dict[str, float]]) -> str:
    """
    Format comparison metrics dictionary into an aligned markdown table.
    """
    keys = list(next(iter(metrics_dict.values())).keys())
    models = list(metrics_dict.keys())

    header = "| Metric | " + " | ".join(models) + " |"
    separator = "|:---| " + " | ".join([":---:"] * len(models)) + " |"
    rows = [header, separator]

    for k in keys:
        row = f"| **{k}** |"
        for m in models:
            val = metrics_dict[m].get(k, 0.0)
            if isinstance(val, float):
                if "%" in k:
                    val_str = f"{val:+.2f}%"
                elif "Ratio" in k or "Factor" in k:
                    val_str = f"{val:.2f}"
                elif "Balance" in k:
                    val_str = f"${val:,.2f}"
                else:
                    val_str = f"{val:.2f}"
            else:
                val_str = str(val)
            row += f" {val_str} |"
        rows.append(row)

    return "\n".join(rows)
