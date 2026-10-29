"""
Stock data loader and synthetic market data generator.
Fetches historical stock prices using Yahoo Finance (yfinance) or generates
realistic Geometric Brownian Motion (GBM) stock series for offline use.
"""

import os
from typing import Tuple
import numpy as np
import pandas as pd
from utils.indicators import prepare_stock_features


def generate_synthetic_stock_data(
    start_date: str = "2020-01-01",
    days: int = 1260,  # ~5 years of trading days
    initial_price: float = 150.0,
    annual_drift: float = 0.12,  # 12% annual drift
    annual_volatility: float = 0.25,  # 25% annual volatility
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic synthetic daily stock prices using Geometric Brownian Motion (GBM).
    dSt = mu * St * dt + sigma * St * dWt
    """
    np.random.seed(seed)
    dt = 1.0 / 252.0  # Daily time step
    mu = annual_drift
    sigma = annual_volatility

    # Generate dates excluding weekends
    dates = pd.bdate_range(start=start_date, periods=days)

    # Brownian shock
    shocks = np.random.normal(0, np.sqrt(dt), size=days)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * shocks

    # Add realistic market regimes (a bull run, a dip, a recovery)
    regime_cycle = np.sin(np.linspace(0, 4 * np.pi, days)) * 0.003
    log_returns += regime_cycle

    price_path = initial_price * np.exp(np.cumsum(log_returns))

    # Construct OHLCV DataFrame
    df = pd.DataFrame(index=dates)
    df['Close'] = price_path

    # Daily intraday variations
    high_mult = 1.0 + np.abs(np.random.normal(0, 0.008, size=days))
    low_mult = 1.0 - np.abs(np.random.normal(0, 0.008, size=days))
    open_mult = 1.0 + np.random.normal(0, 0.004, size=days)

    df['Open'] = df['Close'] * open_mult
    df['High'] = np.maximum(df['Open'], df['Close']) * high_mult
    df['Low'] = np.minimum(df['Open'], df['Close']) * low_mult
    df['Volume'] = (np.random.lognormal(mean=14, sigma=0.5, size=days)).astype(int)

    return df


def load_or_fetch_stock_data(
    ticker: str = "AAPL",
    start_date: str = "2020-01-01",
    end_date: str = "2024-01-01",
    csv_fallback_path: str = None,
    force_synthetic: bool = False,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Loads stock data:
    1. If cached CSV exists for the given ticker, load cached CSV instantly.
    2. If not, fetch from Yahoo Finance and cache to data/{ticker}.csv.
    3. If yfinance fails or offline, fall back to synthetic data.
    """
    if csv_fallback_path is None:
        # Check ticker-specific file first, then fallback to sample_stock_data.csv if AAPL
        ticker_clean = ticker.replace("^", "").replace(".", "_")
        csv_fallback_path = f"data/{ticker_clean}.csv"
        if not os.path.exists(csv_fallback_path) and ticker == "AAPL" and os.path.exists("data/sample_stock_data.csv"):
            csv_fallback_path = "data/sample_stock_data.csv"

    df = None

    if os.path.exists(csv_fallback_path) and not force_download and not force_synthetic:
        print(f"[*] Loading cached data from {csv_fallback_path} for {ticker}...")
        df = pd.read_csv(csv_fallback_path, index_col=0, parse_dates=True)

    if df is None and not force_synthetic:
        try:
            import yfinance as yf
            print(f"[*] Fetching {ticker} from Yahoo Finance ({start_date} to {end_date})...")
            downloaded = yf.download(ticker, start=start_date, end=end_date, progress=False, timeout=10)
            if not downloaded.empty and len(downloaded) > 50:
                if isinstance(downloaded.columns, pd.MultiIndex):
                    downloaded = downloaded.xs(ticker, axis=1, level=1) if ticker in downloaded.columns.levels[1] else downloaded
                    if isinstance(downloaded.columns, pd.MultiIndex):
                        downloaded.columns = [c[0] for c in downloaded.columns]
                df = downloaded
                os.makedirs(os.path.dirname(csv_fallback_path), exist_ok=True)
                df.to_csv(csv_fallback_path)
                print(f"[+] Successfully fetched {len(df)} rows for {ticker} and saved to {csv_fallback_path}.")
        except Exception as e:
            print(f"[!] yfinance fetch failed for {ticker} ({e}). Generating fallback data.")

    if df is None:
        print(f"[*] Generating synthetic market data for {ticker}...")
        df = generate_synthetic_stock_data(start_date=start_date, days=1260)
        os.makedirs(os.path.dirname(csv_fallback_path), exist_ok=True)
        df.to_csv(csv_fallback_path)
        print(f"[+] Saved synthetic dataset to {csv_fallback_path}")

    # Calculate indicators
    df = prepare_stock_features(df)
    return df


def train_test_split_stock(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/test split.
    Sequential split is essential in financial time series to prevent lookahead bias.
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


if __name__ == "__main__":
    data = load_or_fetch_stock_data()
    print("Features summary:")
    print(data.info())
    print("\nSample records:")
    print(data[['Close', 'SMA_30', 'RSI', 'MACD', 'Bollinger_PctB']].head())
