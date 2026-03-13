"""
Wyckoff Features Module
Detects Wyckoff market phases and patterns:
- Accumulation / Distribution detection
- Spring and Upthrust detection
- Effort vs Result analysis
"""

import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger("wyckoff")


def detect_wyckoff_phases(df: pd.DataFrame, range_lookback: int = 50) -> pd.DataFrame:
    """
    Detect Wyckoff market phases:
    0 = Unknown/Transitional
    1 = Accumulation (range after downtrend - smart money buying)
    2 = Markup (uptrend)
    3 = Distribution (range after uptrend - smart money selling)
    4 = Markdown (downtrend)
    """
    n = len(df)
    phase = np.zeros(n)

    if "atr_14" not in df.columns:
        df["wyckoff_phase"] = 0
        return df

    close = df["close"].values
    atr = df["atr_14"].values

    for i in range(range_lookback, n):
        window = close[i - range_lookback:i + 1]
        window_range = np.max(window) - np.min(window)
        window_atr = np.nanmean(atr[i - range_lookback:i + 1])

        if window_atr == 0 or np.isnan(window_atr):
            continue

        # Normalized range: small = ranging, large = trending
        norm_range = window_range / (window_atr * range_lookback)

        # Trend direction in window
        first_quarter = np.mean(window[:range_lookback // 4])
        last_quarter = np.mean(window[-range_lookback // 4:])
        trend_direction = last_quarter - first_quarter

        # Classify phase
        if norm_range < 0.15:  # Tight range
            if trend_direction < -window_atr:
                phase[i] = 1  # Accumulation (range after decline)
            elif trend_direction > window_atr:
                phase[i] = 3  # Distribution (range after rally)
            else:
                phase[i] = 1 if close[i] < np.mean(window) else 3
        else:  # Trending
            if trend_direction > 0:
                phase[i] = 2  # Markup
            else:
                phase[i] = 4  # Markdown

    df["wyckoff_phase"] = phase

    # Phase as one-hot for ML
    for p in [1, 2, 3, 4]:
        df[f"wyckoff_phase_{p}"] = (phase == p).astype(int)

    logger.info(f"  Wyckoff phases detected")
    return df


def detect_springs_upthrusts(df: pd.DataFrame, lookback: int = 30) -> pd.DataFrame:
    """
    Detect Wyckoff Springs and Upthrusts.

    Spring: price briefly breaks BELOW support, then quickly recovers (bullish).
    Upthrust: price briefly breaks ABOVE resistance, then quickly reverses (bearish).
    """
    n = len(df)
    springs = np.zeros(n)
    upthrusts = np.zeros(n)

    for i in range(lookback + 2, n):
        window_lows = df["low"].iloc[i - lookback:i]
        window_highs = df["high"].iloc[i - lookback:i]

        support = window_lows.min()
        resistance = window_highs.max()

        # Spring: low goes below support but close recovers above it
        if df["low"].iloc[i] < support and df["close"].iloc[i] > support:
            # Confirm with volume or next candle
            if df["close"].iloc[i] > df["open"].iloc[i]:  # Bullish close
                springs[i] = 1

        # Upthrust: high goes above resistance but close falls below it
        if df["high"].iloc[i] > resistance and df["close"].iloc[i] < resistance:
            if df["close"].iloc[i] < df["open"].iloc[i]:  # Bearish close
                upthrusts[i] = 1

    df["wyckoff_spring"] = springs
    df["wyckoff_upthrust"] = upthrusts

    spring_count = int(springs.sum())
    upthrust_count = int(upthrusts.sum())
    logger.info(f"  Wyckoff: {spring_count} springs, {upthrust_count} upthrusts detected")

    return df


def add_effort_vs_result(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wyckoff Effort vs Result analysis.
    Effort = volume, Result = price movement.
    High effort + low result = potential reversal.
    Low effort + high result = strong move.
    """
    if "volume" not in df.columns or "atr_14" not in df.columns:
        df["effort_vs_result"] = np.nan
        return df

    # Price movement (result)
    price_move = abs(df["close"] - df["open"])
    avg_move = price_move.rolling(20).mean()

    # Volume (effort)
    avg_vol = df["volume"].rolling(20).mean()

    # Normalized effort and result
    safe_avg_move = avg_move.replace(0, np.nan)
    safe_avg_vol = avg_vol.replace(0, np.nan)

    norm_result = price_move / safe_avg_move
    norm_effort = df["volume"] / safe_avg_vol

    # Effort vs Result ratio
    # > 1 = more effort than result (divergence, potential reversal)
    # < 1 = more result than effort (strong conviction)
    safe_norm_result = norm_result.replace(0, np.nan)
    df["effort_vs_result"] = norm_effort / safe_norm_result

    logger.info(f"  Effort vs Result analysis added")
    return df


def compute_all_wyckoff(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all Wyckoff features."""
    df = detect_wyckoff_phases(df)
    df = detect_springs_upthrusts(df)
    df = add_effort_vs_result(df)
    return df
