"""
Simple feature extraction for stock trading.
Keeps only the most intuitive signals so it's easy to explain to faculty.
"""

import numpy as np
import pandas as pd


def prepare_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches stock dataframe with simple, intuitive features:
    1. Daily Return (% price change from previous day)
    2. 5-Day Simple Moving Average (SMA_5) to capture short-term trend
    3. Trend Signal:
       +1 : Price > SMA_5 by more than 0.5% (Uptrend / Bullish)
        0 : Price near SMA_5 within [-0.5%, +0.5%] (Sideways / Neutral)
       -1 : Price < SMA_5 by more than 0.5% (Downtrend / Bearish)
    """
    data = df.copy()
    close = data['Close']

    # 1. Daily return
    data['Daily_Return'] = close.pct_change().fillna(0.0)

    # 2. Short-term trend: 5-day Moving Average
    data['SMA_5'] = close.rolling(window=5, min_periods=1).mean()

    # 3. % Distance from SMA_5
    ratio = (close - data['SMA_5']) / (data['SMA_5'] + 1e-9)

    # 4. Discrete Trend Signal: -1 (Down), 0 (Flat), +1 (Up)
    conditions = [
        ratio < -0.005,           # Down > 0.5%
        (ratio >= -0.005) & (ratio <= 0.005), # Flat
        ratio > 0.005            # Up > 0.5%
    ]
    choices = [-1, 0, 1]
    data['Trend'] = np.select(conditions, choices, default=0)

    return data
