"""
Divergence Detection Module
Detects bullish and bearish RSI divergences — a key reversal signal.
"""

import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger("divergence")


def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Detect RSI divergence.

    Bullish divergence: price makes lower low, RSI makes higher low -> reversal up
    Bearish divergence: price makes higher high, RSI makes lower high -> reversal down
    """
    n = len(df)
    bull_div = np.zeros(n)
    bear_div = np.zeros(n)

    if "rsi_14" not in df.columns:
        logger.warning("  RSI not found, skipping divergence detection")
        df["rsi_bull_divergence"] = 0
        df["rsi_bear_divergence"] = 0
        return df

    close = df["close"].values
    rsi = df["rsi_14"].values

    for i in range(lookback, n):
        window_close = close[i - lookback:i + 1]
        window_rsi = rsi[i - lookback:i + 1]

        if np.any(np.isnan(window_rsi)):
            continue

        # Find local minimums for bullish divergence
        # Current bar is near the low of the lookback period
        if close[i] <= np.percentile(window_close, 10):
            # Find the previous low in the window
            prev_low_idx = np.argmin(window_close[:-5]) if len(window_close) > 5 else 0
            prev_low_price = window_close[prev_low_idx]
            prev_low_rsi = window_rsi[prev_low_idx]

            # Bullish divergence: lower price low but higher RSI low
            if close[i] < prev_low_price and rsi[i] > prev_low_rsi:
                bull_div[i] = 1

        # Find local maximums for bearish divergence
        if close[i] >= np.percentile(window_close, 90):
            prev_high_idx = np.argmax(window_close[:-5]) if len(window_close) > 5 else 0
            prev_high_price = window_close[prev_high_idx]
            prev_high_rsi = window_rsi[prev_high_idx]

            # Bearish divergence: higher price high but lower RSI high
            if close[i] > prev_high_price and rsi[i] < prev_high_rsi:
                bear_div[i] = 1

    df["rsi_bull_divergence"] = bull_div
    df["rsi_bear_divergence"] = bear_div

    bull_count = int(bull_div.sum())
    bear_count = int(bear_div.sum())
    logger.info(f"  Divergence: {bull_count} bullish, {bear_count} bearish RSI divergences")

    return df
