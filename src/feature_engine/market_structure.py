"""
Market Structure Module
Detects swing highs/lows, BOS (Break of Structure), CHOCH (Change of Character),
Fair Value Gaps (FVGs), Order Blocks, and Liquidity Zones.
These are the building blocks for ICT and Wyckoff strategies.
"""

import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger("market_structure")


def detect_swing_points(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """
    Detect swing highs and swing lows WITHOUT lookahead bias.

    A swing high at bar i requires that bars i-lookback..i-1 and i+1..i+lookback
    are all lower. In live trading this can only be CONFIRMED after `lookback`
    bars have passed. To avoid lookahead bias in backtesting we record the
    confirmed swing at bar i+lookback (the bar when the confirmation is complete),
    not at bar i. Forward-fill then propagates the level to all future bars.
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    # swing_high[k] / swing_low[k] is set at the CONFIRMATION bar (i + lookback),
    # storing the actual swing price. This matches live-trading behaviour.
    swing_high = np.full(n, np.nan)
    swing_low  = np.full(n, np.nan)

    for i in range(lookback, n - lookback):
        confirm_idx = i + lookback   # bar at which confirmation is available

        # Check swing high: bar i is higher than all neighbours
        is_swing_high = all(highs[i] > highs[i - j] for j in range(1, lookback + 1)) and \
                        all(highs[i] > highs[i + j] for j in range(1, lookback + 1))
        if is_swing_high:
            swing_high[confirm_idx] = highs[i]

        # Check swing low
        is_swing_low = all(lows[i] < lows[i - j] for j in range(1, lookback + 1)) and \
                       all(lows[i] < lows[i + j] for j in range(1, lookback + 1))
        if is_swing_low:
            swing_low[confirm_idx] = lows[i]

    df["swing_high"] = swing_high
    df["swing_low"]  = swing_low

    # Forward-fill: each bar knows the LAST CONFIRMED swing level
    df["last_swing_high"] = df["swing_high"].ffill()
    df["last_swing_low"]  = df["swing_low"].ffill()

    # Distance from price to last confirmed swing points
    df["dist_to_swing_high"] = (df["close"] - df["last_swing_high"]) / df["close"] * 100
    df["dist_to_swing_low"]  = (df["close"] - df["last_swing_low"])  / df["close"] * 100

    sh_count = df["swing_high"].notna().sum()
    sl_count = df["swing_low"].notna().sum()
    logger.info(f"  Swing points: {sh_count} highs, {sl_count} lows detected (no-lookahead)")

    return df


def detect_bos_choch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Break of Structure (BOS) and Change of Character (CHOCH).

    BOS = price breaks a swing point IN the direction of the current trend
          (bullish trend breaks above swing high = BOS up)
    CHOCH = price breaks a swing point AGAINST the current trend
            (bullish trend breaks below swing low = CHOCH = potential reversal)
    """
    n = len(df)
    bos = np.zeros(n)        # +1 = bullish BOS, -1 = bearish BOS
    choch = np.zeros(n)      # +1 = bullish CHOCH, -1 = bearish CHOCH
    trend = np.zeros(n)      # +1 = uptrend, -1 = downtrend

    last_sh = np.nan
    last_sl = np.nan
    current_trend = 0  # 0 = undefined

    for i in range(1, n):
        # Update swing points
        if pd.notna(df["swing_high"].iloc[i]):
            last_sh = df["swing_high"].iloc[i]
        if pd.notna(df["swing_low"].iloc[i]):
            last_sl = df["swing_low"].iloc[i]

        if pd.isna(last_sh) or pd.isna(last_sl):
            trend[i] = current_trend
            continue

        close = df["close"].iloc[i]
        prev_close = df["close"].iloc[i - 1]

        # Break above swing high
        if close > last_sh and prev_close <= last_sh:
            if current_trend >= 0:
                bos[i] = 1  # Bullish BOS (continuation)
            else:
                choch[i] = 1  # Bullish CHOCH (reversal from downtrend)
            current_trend = 1

        # Break below swing low
        elif close < last_sl and prev_close >= last_sl:
            if current_trend <= 0:
                bos[i] = -1  # Bearish BOS (continuation)
            else:
                choch[i] = -1  # Bearish CHOCH (reversal from uptrend)
            current_trend = -1

        trend[i] = current_trend

    df["bos"] = bos
    df["choch"] = choch
    df["market_trend"] = trend

    bos_count = (np.abs(bos) > 0).sum()
    choch_count = (np.abs(choch) > 0).sum()
    logger.info(f"  Structure: {bos_count} BOS, {choch_count} CHOCH events detected")

    return df


def detect_fvg(df: pd.DataFrame, min_gap_atr: float = 0.3) -> pd.DataFrame:
    """
    Detect Fair Value Gaps (FVGs) - also called imbalances.
    A bullish FVG: candle[i-1].high < candle[i+1].low (gap between them)
    A bearish FVG: candle[i-1].low > candle[i+1].high (gap between them)
    """
    n = len(df)
    fvg_bull = np.zeros(n)
    fvg_bear = np.zeros(n)
    fvg_bull_top = np.full(n, np.nan)
    fvg_bull_bot = np.full(n, np.nan)
    fvg_bear_top = np.full(n, np.nan)
    fvg_bear_bot = np.full(n, np.nan)

    atr = df["atr_14"].values if "atr_14" in df.columns else np.ones(n)

    for i in range(1, n - 1):
        # Bullish FVG: gap up
        gap_up = df["low"].iloc[i + 1] - df["high"].iloc[i - 1]
        if gap_up > 0 and (pd.isna(atr[i]) or gap_up > atr[i] * min_gap_atr):
            fvg_bull[i] = 1
            fvg_bull_top[i] = df["low"].iloc[i + 1]
            fvg_bull_bot[i] = df["high"].iloc[i - 1]

        # Bearish FVG: gap down
        gap_down = df["low"].iloc[i - 1] - df["high"].iloc[i + 1]
        if gap_down > 0 and (pd.isna(atr[i]) or gap_down > atr[i] * min_gap_atr):
            fvg_bear[i] = 1
            fvg_bear_top[i] = df["low"].iloc[i - 1]
            fvg_bear_bot[i] = df["high"].iloc[i + 1]

    df["fvg_bullish"] = fvg_bull
    df["fvg_bearish"] = fvg_bear
    df["fvg_bull_top"] = fvg_bull_top
    df["fvg_bull_bot"] = fvg_bull_bot
    df["fvg_bear_top"] = fvg_bear_top
    df["fvg_bear_bot"] = fvg_bear_bot

    # Nearest unfilled FVG (forward-filled for reference)
    df["nearest_fvg_bull_top"] = df["fvg_bull_top"].ffill()
    df["nearest_fvg_bull_bot"] = df["fvg_bull_bot"].ffill()
    df["nearest_fvg_bear_top"] = df["fvg_bear_top"].ffill()
    df["nearest_fvg_bear_bot"] = df["fvg_bear_bot"].ffill()

    bull_count = int(fvg_bull.sum())
    bear_count = int(fvg_bear.sum())
    logger.info(f"  FVGs: {bull_count} bullish, {bear_count} bearish gaps detected")

    return df


def detect_order_blocks(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Detect Order Blocks (OBs).
    Bullish OB: the last bearish candle before a strong bullish move up.
    Bearish OB: the last bullish candle before a strong bearish move down.
    """
    n = len(df)
    ob_bull = np.zeros(n)
    ob_bear = np.zeros(n)
    ob_bull_high = np.full(n, np.nan)
    ob_bull_low = np.full(n, np.nan)
    ob_bear_high = np.full(n, np.nan)
    ob_bear_low = np.full(n, np.nan)

    atr = df["atr_14"].values if "atr_14" in df.columns else np.ones(n)

    for i in range(2, n):
        body_prev = df["close"].iloc[i - 1] - df["open"].iloc[i - 1]
        body_curr = df["close"].iloc[i] - df["open"].iloc[i]

        safe_atr = atr[i] if pd.notna(atr[i]) and atr[i] > 0 else 1

        # Bullish OB: previous candle was bearish, current candle is strongly bullish
        if body_prev < 0 and body_curr > safe_atr * 1.0:
            ob_bull[i - 1] = 1
            ob_bull_high[i - 1] = df["high"].iloc[i - 1]
            ob_bull_low[i - 1] = df["low"].iloc[i - 1]

        # Bearish OB: previous candle was bullish, current candle is strongly bearish
        if body_prev > 0 and body_curr < -safe_atr * 1.0:
            ob_bear[i - 1] = 1
            ob_bear_high[i - 1] = df["high"].iloc[i - 1]
            ob_bear_low[i - 1] = df["low"].iloc[i - 1]

    df["ob_bullish"] = ob_bull
    df["ob_bearish"] = ob_bear
    df["ob_bull_high"] = ob_bull_high
    df["ob_bull_low"] = ob_bull_low
    df["ob_bear_high"] = ob_bear_high
    df["ob_bear_low"] = ob_bear_low

    # Forward-fill nearest OB levels
    df["nearest_ob_bull_high"] = df["ob_bull_high"].ffill()
    df["nearest_ob_bull_low"] = df["ob_bull_low"].ffill()
    df["nearest_ob_bear_high"] = df["ob_bear_high"].ffill()
    df["nearest_ob_bear_low"] = df["ob_bear_low"].ffill()

    bull_count = int(ob_bull.sum())
    bear_count = int(ob_bear.sum())
    logger.info(f"  Order Blocks: {bull_count} bullish, {bear_count} bearish OBs detected")

    return df


def detect_liquidity_zones(df: pd.DataFrame, window: int = 20, tolerance_pct: float = 0.1) -> pd.DataFrame:
    """
    Detect liquidity zones — areas where equal highs/lows cluster.
    These are where stop-losses accumulate, making them targets for smart money.
    """
    n = len(df)
    liq_above = np.zeros(n)  # Liquidity pool above price
    liq_below = np.zeros(n)  # Liquidity pool below price

    highs = df["high"].values
    lows = df["low"].values
    close = df["close"].values

    for i in range(window, n):
        # Look for equal highs in the window (within tolerance)
        recent_highs = highs[i - window:i]
        max_high = np.max(recent_highs)
        tolerance = max_high * tolerance_pct / 100

        equal_high_count = np.sum(np.abs(recent_highs - max_high) < tolerance)
        if equal_high_count >= 2 and close[i] < max_high:
            liq_above[i] = equal_high_count

        # Look for equal lows
        recent_lows = lows[i - window:i]
        min_low = np.min(recent_lows)
        tolerance = min_low * tolerance_pct / 100

        equal_low_count = np.sum(np.abs(recent_lows - min_low) < tolerance)
        if equal_low_count >= 2 and close[i] > min_low:
            liq_below[i] = equal_low_count

    df["liquidity_above"] = liq_above
    df["liquidity_below"] = liq_below

    logger.info(f"  Liquidity zones mapped")
    return df


def compute_all_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all market structure features."""
    df = detect_swing_points(df)
    df = detect_bos_choch(df)
    df = detect_fvg(df)
    df = detect_order_blocks(df)
    df = detect_liquidity_zones(df)
    return df
