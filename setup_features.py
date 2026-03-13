"""
XAUUSD AI Trading System - Feature Engine Setup
Run this to create the Feature Engine module files.

Usage:
    python setup_features.py
"""

import os

FILES = {}

# ============================================================
# FILE 1: src/feature_engine/indicators.py
# Technical indicators using pandas-ta
# ============================================================
FILES["src/feature_engine/indicators.py"] = r'''"""
Technical Indicators Module
Computes 30+ indicators across trend, momentum, volatility, and volume categories.
These feed directly into the ML model and strategy detectors.
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger("indicators")


def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trend-following indicators.
    These tell us: is gold going up, down, or sideways?
    """
    # --- Exponential Moving Averages ---
    # Fast (8) for short-term, Medium (21,50) for trend, Slow (200) for big picture
    df["ema_8"] = ta.ema(df["close"], length=8)
    df["ema_21"] = ta.ema(df["close"], length=21)
    df["ema_50"] = ta.ema(df["close"], length=50)
    df["ema_200"] = ta.ema(df["close"], length=200)

    # --- EMA Alignment Score ---
    # +1 if EMAs are stacked bullish (8>21>50>200), -1 if bearish
    def ema_alignment(row):
        score = 0
        if pd.notna(row["ema_8"]) and pd.notna(row["ema_21"]):
            score += 1 if row["ema_8"] > row["ema_21"] else -1
        if pd.notna(row["ema_21"]) and pd.notna(row["ema_50"]):
            score += 1 if row["ema_21"] > row["ema_50"] else -1
        if pd.notna(row["ema_50"]) and pd.notna(row["ema_200"]):
            score += 1 if row["ema_50"] > row["ema_200"] else -1
        return score

    df["ema_alignment"] = df.apply(ema_alignment, axis=1)

    # --- MACD (12, 26, 9) ---
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None:
        df = pd.concat([df, macd], axis=1)

    # --- ADX (Average Directional Index) ---
    # Measures trend strength (>25 = strong trend, <20 = no trend)
    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx is not None:
        df = pd.concat([df, adx], axis=1)

    # --- Parabolic SAR ---
    psar = ta.psar(df["high"], df["low"], df["close"])
    if psar is not None:
        df = pd.concat([df, psar], axis=1)

    # --- Supertrend ---
    st = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
    if st is not None:
        df = pd.concat([df, st], axis=1)

    # --- Price position relative to EMAs ---
    if "ema_21" in df.columns:
        df["price_vs_ema21"] = (df["close"] - df["ema_21"]) / df["ema_21"] * 100
    if "ema_200" in df.columns:
        df["price_vs_ema200"] = (df["close"] - df["ema_200"]) / df["ema_200"] * 100

    logger.info(f"  Trend indicators added: EMA(8,21,50,200), MACD, ADX, PSAR, Supertrend")
    return df


def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum indicators.
    These tell us: is the move strong or weakening?
    """
    # --- RSI (Relative Strength Index) ---
    df["rsi_14"] = ta.rsi(df["close"], length=14)
    df["rsi_7"] = ta.rsi(df["close"], length=7)

    # RSI zones
    if "rsi_14" in df.columns:
        df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
        df["rsi_oversold"] = (df["rsi_14"] < 30).astype(int)

    # --- Stochastic RSI ---
    stochrsi = ta.stochrsi(df["close"], length=14, rsi_length=14)
    if stochrsi is not None:
        df = pd.concat([df, stochrsi], axis=1)

    # --- CCI (Commodity Channel Index) ---
    df["cci_20"] = ta.cci(df["high"], df["low"], df["close"], length=20)

    # --- Williams %R ---
    df["willr_14"] = ta.willr(df["high"], df["low"], df["close"], length=14)

    # --- Rate of Change ---
    df["roc_10"] = ta.roc(df["close"], length=10)

    # --- Momentum ---
    df["mom_10"] = ta.mom(df["close"], length=10)

    # --- TSI (True Strength Index) ---
    tsi = ta.tsi(df["close"])
    if tsi is not None:
        df = pd.concat([df, tsi], axis=1)

    logger.info(f"  Momentum indicators added: RSI(7,14), StochRSI, CCI, Williams%R, ROC, MOM, TSI")
    return df


def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volatility indicators.
    These tell us: how much is gold moving? Is it expanding or contracting?
    Critical for stop-loss sizing and trade filtering.
    """
    # --- ATR (Average True Range) ---
    # THE most important indicator for risk management
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["atr_7"] = ta.atr(df["high"], df["low"], df["close"], length=7)

    # ATR as percentage of price (normalized volatility)
    if "atr_14" in df.columns:
        df["atr_pct"] = df["atr_14"] / df["close"] * 100

    # --- Bollinger Bands ---
    bbands = ta.bbands(df["close"], length=20, std=2.0)
    if bbands is not None:
        df = pd.concat([df, bbands], axis=1)

    # Bollinger Band width (squeeze detection)
    bb_upper = [c for c in df.columns if "BBU_" in c]
    bb_lower = [c for c in df.columns if "BBL_" in c]
    bb_mid = [c for c in df.columns if "BBM_" in c]
    if bb_upper and bb_lower and bb_mid:
        df["bb_width"] = (df[bb_upper[0]] - df[bb_lower[0]]) / df[bb_mid[0]] * 100
        df["bb_position"] = (df["close"] - df[bb_lower[0]]) / (df[bb_upper[0]] - df[bb_lower[0]])

    # --- Keltner Channels ---
    kc = ta.kc(df["high"], df["low"], df["close"], length=20, scalar=1.5)
    if kc is not None:
        df = pd.concat([df, kc], axis=1)

    # --- Donchian Channels ---
    dc = ta.donchian(df["high"], df["low"], lower_length=20, upper_length=20)
    if dc is not None:
        df = pd.concat([df, dc], axis=1)

    # --- Historical Volatility ---
    df["hvol_20"] = df["close"].pct_change().rolling(20).std() * np.sqrt(252) * 100

    # --- Candle Body Size (normalized) ---
    df["body_size"] = abs(df["close"] - df["open"])
    df["candle_range"] = df["high"] - df["low"]
    if "atr_14" in df.columns:
        df["body_atr_ratio"] = df["body_size"] / df["atr_14"]
        df["range_atr_ratio"] = df["candle_range"] / df["atr_14"]

    # --- Upper/Lower Wick Analysis ---
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    if "candle_range" in df.columns:
        safe_range = df["candle_range"].replace(0, np.nan)
        df["upper_wick_pct"] = df["upper_wick"] / safe_range
        df["lower_wick_pct"] = df["lower_wick"] / safe_range

    logger.info(f"  Volatility indicators added: ATR(7,14), BBands, Keltner, Donchian, HVol, Candle analysis")
    return df


def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume indicators.
    Volume confirms price moves — a breakout without volume is likely fake.
    """
    if "volume" not in df.columns or df["volume"].sum() == 0:
        logger.warning("  No volume data available, skipping volume indicators")
        return df

    # --- Volume SMA ---
    df["vol_sma_20"] = ta.sma(df["volume"], length=20)

    # --- Relative Volume ---
    if "vol_sma_20" in df.columns:
        safe_vol = df["vol_sma_20"].replace(0, np.nan)
        df["relative_volume"] = df["volume"] / safe_vol

    # --- OBV (On-Balance Volume) ---
    obv = ta.obv(df["close"], df["volume"])
    if obv is not None:
        df["obv"] = obv
        df["obv_ema"] = ta.ema(df["obv"], length=21)

    # --- MFI (Money Flow Index) - RSI with volume ---
    df["mfi_14"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)

    # --- AD Line (Accumulation/Distribution) ---
    ad = ta.ad(df["high"], df["low"], df["close"], df["volume"])
    if ad is not None:
        df["ad_line"] = ad

    # --- CMF (Chaikin Money Flow) ---
    cmf = ta.cmf(df["high"], df["low"], df["close"], df["volume"], length=20)
    if cmf is not None:
        df["cmf_20"] = cmf

    # --- Volume Trend ---
    # Is volume increasing or decreasing over last 10 bars?
    if "vol_sma_20" in df.columns:
        df["vol_trend"] = df["volume"].rolling(5).mean() / df["volume"].rolling(20).mean()

    logger.info(f"  Volume indicators added: Vol SMA, Relative Vol, OBV, MFI, AD, CMF")
    return df


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function: compute ALL technical indicators on a DataFrame.

    Args:
        df: DataFrame with columns [open, high, low, close, volume]

    Returns:
        DataFrame with all indicators added as new columns
    """
    original_cols = len(df.columns)

    df = add_trend_indicators(df)
    df = add_momentum_indicators(df)
    df = add_volatility_indicators(df)
    df = add_volume_indicators(df)

    new_cols = len(df.columns) - original_cols
    logger.info(f"  Total: {new_cols} indicator columns added")

    return df
'''

# ============================================================
# FILE 2: src/feature_engine/market_structure.py
# ICT-style market structure detection
# ============================================================
FILES["src/feature_engine/market_structure.py"] = r'''"""
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
    Detect swing highs and swing lows.
    A swing high = a candle whose high is higher than `lookback` candles on each side.
    A swing low = a candle whose low is lower than `lookback` candles on each side.
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)

    for i in range(lookback, n - lookback):
        # Check swing high
        is_swing_high = True
        for j in range(1, lookback + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_swing_high = False
                break
        if is_swing_high:
            swing_high[i] = highs[i]

        # Check swing low
        is_swing_low = True
        for j in range(1, lookback + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_swing_low = False
                break
        if is_swing_low:
            swing_low[i] = lows[i]

    df["swing_high"] = swing_high
    df["swing_low"] = swing_low

    # Forward-fill the last known swing levels (for S/R reference)
    df["last_swing_high"] = df["swing_high"].ffill()
    df["last_swing_low"] = df["swing_low"].ffill()

    # Distance from price to last swing points (useful for ML)
    df["dist_to_swing_high"] = (df["close"] - df["last_swing_high"]) / df["close"] * 100
    df["dist_to_swing_low"] = (df["close"] - df["last_swing_low"]) / df["close"] * 100

    sh_count = df["swing_high"].notna().sum()
    sl_count = df["swing_low"].notna().sum()
    logger.info(f"  Swing points: {sh_count} highs, {sl_count} lows detected")

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
'''

# ============================================================
# FILE 3: src/feature_engine/support_resistance.py
# ============================================================
FILES["src/feature_engine/support_resistance.py"] = r'''"""
Support & Resistance Module
Detects key horizontal price levels where gold tends to bounce or break.
"""

import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger("support_resistance")


def detect_sr_levels(df: pd.DataFrame, lookback: int = 50, min_touches: int = 2,
                     zone_pct: float = 0.15) -> pd.DataFrame:
    """
    Detect support and resistance levels based on swing point clustering.

    Args:
        df: DataFrame with swing_high and swing_low columns
        lookback: How many bars to look back for level detection
        min_touches: Minimum number of touches to confirm a level
        zone_pct: Zone width as percentage of price (0.15% for gold ~ $7.5 at $5000)
    """
    n = len(df)

    # Collect all swing points
    levels = []
    for i in range(n):
        if pd.notna(df["swing_high"].iloc[i]):
            levels.append(df["swing_high"].iloc[i])
        if pd.notna(df["swing_low"].iloc[i]):
            levels.append(df["swing_low"].iloc[i])

    if not levels:
        logger.warning("  No swing points found for S/R detection")
        df["nearest_support"] = np.nan
        df["nearest_resistance"] = np.nan
        df["sr_strength"] = 0
        return df

    # Cluster nearby levels into zones
    levels = sorted(levels)
    zones = []
    used = set()

    for i, level in enumerate(levels):
        if i in used:
            continue
        zone_levels = [level]
        tolerance = level * zone_pct / 100

        for j in range(i + 1, len(levels)):
            if j in used:
                continue
            if abs(levels[j] - level) < tolerance:
                zone_levels.append(levels[j])
                used.add(j)

        if len(zone_levels) >= min_touches:
            zones.append({
                "level": np.mean(zone_levels),
                "touches": len(zone_levels),
                "strength": len(zone_levels),
            })

    logger.info(f"  S/R: Found {len(zones)} confirmed zones (min {min_touches} touches)")

    # For each bar, find nearest support and resistance
    nearest_support = np.full(n, np.nan)
    nearest_resistance = np.full(n, np.nan)
    sr_strength = np.zeros(n)

    zone_levels_arr = np.array([z["level"] for z in zones])
    zone_strengths = np.array([z["strength"] for z in zones])

    for i in range(n):
        price = df["close"].iloc[i]

        if len(zone_levels_arr) == 0:
            continue

        # Support = nearest zone below price
        below = zone_levels_arr[zone_levels_arr < price]
        if len(below) > 0:
            idx = np.argmin(price - below)
            nearest_support[i] = below[idx]

        # Resistance = nearest zone above price
        above = zone_levels_arr[zone_levels_arr > price]
        if len(above) > 0:
            idx = np.argmin(above - price)
            nearest_resistance[i] = above[idx]

        # Strength = how close to any strong level?
        distances = np.abs(zone_levels_arr - price)
        closest_idx = np.argmin(distances)
        closest_dist_pct = distances[closest_idx] / price * 100
        if closest_dist_pct < zone_pct * 2:
            sr_strength[i] = zone_strengths[closest_idx]

    df["nearest_support"] = nearest_support
    df["nearest_resistance"] = nearest_resistance
    df["sr_strength"] = sr_strength

    # Distance to nearest S/R (as % of price)
    df["dist_to_support_pct"] = (df["close"] - df["nearest_support"]) / df["close"] * 100
    df["dist_to_resistance_pct"] = (df["nearest_resistance"] - df["close"]) / df["close"] * 100

    # S/R range (room to move)
    df["sr_range"] = df["nearest_resistance"] - df["nearest_support"]

    # Position within S/R range (0 = at support, 1 = at resistance)
    safe_range = df["sr_range"].replace(0, np.nan)
    df["sr_position"] = (df["close"] - df["nearest_support"]) / safe_range

    # Round number proximity (gold traders watch $50 and $100 levels)
    df["dist_to_round_50"] = df["close"] % 50
    df["near_round_50"] = (df["dist_to_round_50"].apply(lambda x: min(x, 50 - x)) < 5).astype(int)
    df["dist_to_round_100"] = df["close"] % 100
    df["near_round_100"] = (df["dist_to_round_100"].apply(lambda x: min(x, 100 - x)) < 5).astype(int)

    return df
'''

# ============================================================
# FILE 4: src/feature_engine/session_features.py
# ============================================================
FILES["src/feature_engine/session_features.py"] = r'''"""
Session Features Module
Gold behaves differently during different trading sessions.
This module adds session-based features critical for XAUUSD trading.
"""

import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger("sessions")


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trading session features.
    Gold is most active during London and New York sessions.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("  Index is not DatetimeIndex, skipping session features")
        return df

    hours = df.index.hour

    # --- Session identification (UTC) ---
    df["session_asia"] = ((hours >= 0) & (hours < 7)).astype(int)
    df["session_london"] = ((hours >= 7) & (hours < 12)).astype(int)
    df["session_ny"] = ((hours >= 12) & (hours < 17)).astype(int)
    df["session_overlap"] = ((hours >= 12) & (hours < 15)).astype(int)  # London-NY overlap
    df["session_off_hours"] = ((hours >= 17) | (hours < 0)).astype(int)

    # --- Time features (cyclical encoding for ML) ---
    # Encode hour as sin/cos so 23:00 and 00:00 are close together
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    # Day of week (0=Monday, 4=Friday)
    dow = df.index.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    # Is it Monday (gap risk) or Friday (position squaring)?
    df["is_monday"] = (dow == 0).astype(int)
    df["is_friday"] = (dow == 4).astype(int)

    # --- ICT Killzone Detection ---
    # London Killzone: 07:00-10:00 UTC (highest probability setups)
    df["killzone_london"] = ((hours >= 7) & (hours < 10)).astype(int)
    # New York Killzone: 12:00-15:00 UTC
    df["killzone_ny"] = ((hours >= 12) & (hours < 15)).astype(int)
    # Any killzone active
    df["killzone_active"] = ((df["killzone_london"] == 1) | (df["killzone_ny"] == 1)).astype(int)

    logger.info(f"  Session features added: sessions, killzones, time encoding")
    return df
'''

# ============================================================
# FILE 5: src/feature_engine/divergence.py
# ============================================================
FILES["src/feature_engine/divergence.py"] = r'''"""
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
'''

# ============================================================
# FILE 6: src/feature_engine/wyckoff_features.py
# ============================================================
FILES["src/feature_engine/wyckoff_features.py"] = r'''"""
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
'''

# ============================================================
# FILE 7: src/feature_engine/engine.py - MASTER ENGINE
# ============================================================
FILES["src/feature_engine/engine.py"] = r'''"""
Feature Engine - Master Module
Orchestrates all feature computation across all sub-modules.
This is the single entry point for turning raw candle data into ML-ready features.
"""

import pandas as pd
import time
from src.feature_engine.indicators import compute_all_indicators
from src.feature_engine.market_structure import compute_all_structure
from src.feature_engine.support_resistance import detect_sr_levels
from src.feature_engine.session_features import add_session_features
from src.feature_engine.divergence import detect_rsi_divergence
from src.feature_engine.wyckoff_features import compute_all_wyckoff
from src.utils.logger import setup_logger

logger = setup_logger("feature_engine")


class FeatureEngine:
    """
    Master feature computation engine.
    Takes raw OHLCV data and produces a fully-featured DataFrame
    ready for strategy detection and ML model training.
    """

    def __init__(self):
        pass

    def compute_features(self, df: pd.DataFrame, timeframe: str = "") -> pd.DataFrame:
        """
        Compute ALL features on a single timeframe DataFrame.

        Args:
            df: DataFrame with [open, high, low, close, volume] columns
            timeframe: Label for logging (e.g. "H1")

        Returns:
            DataFrame with 100+ feature columns added
        """
        start_time = time.time()
        original_cols = len(df.columns)
        label = f"[{timeframe}]" if timeframe else ""

        logger.info(f"{'='*60}")
        logger.info(f"Computing features {label} ({len(df)} candles)...")
        logger.info(f"{'='*60}")

        # Step 1: Technical Indicators (trend, momentum, volatility, volume)
        logger.info(f"Step 1/6: Technical Indicators {label}")
        df = compute_all_indicators(df)

        # Step 2: Market Structure (swing points, BOS, CHOCH, FVGs, OBs)
        logger.info(f"Step 2/6: Market Structure {label}")
        df = compute_all_structure(df)

        # Step 3: Support & Resistance
        logger.info(f"Step 3/6: Support & Resistance {label}")
        df = detect_sr_levels(df)

        # Step 4: Session Features
        logger.info(f"Step 4/6: Session Features {label}")
        df = add_session_features(df)

        # Step 5: RSI Divergence
        logger.info(f"Step 5/6: RSI Divergence {label}")
        df = detect_rsi_divergence(df)

        # Step 6: Wyckoff Features
        logger.info(f"Step 6/6: Wyckoff Features {label}")
        df = compute_all_wyckoff(df)

        # Summary
        elapsed = time.time() - start_time
        new_cols = len(df.columns) - original_cols
        null_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100

        logger.info(f"{'='*60}")
        logger.info(f"Feature Engine {label} COMPLETE:")
        logger.info(f"  Columns: {original_cols} -> {len(df.columns)} (+{new_cols} features)")
        logger.info(f"  Rows: {len(df)} candles")
        logger.info(f"  NaN: {null_pct:.1f}% (normal for warmup period)")
        logger.info(f"  Time: {elapsed:.2f} seconds")
        logger.info(f"{'='*60}")

        return df

    def compute_multi_timeframe(self, data: dict) -> dict:
        """
        Compute features for all timeframes.

        Args:
            data: Dictionary of {timeframe_name: DataFrame}

        Returns:
            Dictionary of {timeframe_name: featured_DataFrame}
        """
        featured_data = {}
        for tf_name, df in data.items():
            featured_data[tf_name] = self.compute_features(df.copy(), tf_name)

        return featured_data

    def get_feature_summary(self, df: pd.DataFrame) -> dict:
        """Get a summary of all computed features."""
        categories = {
            "Trend": ["ema_", "macd", "adx", "psar", "supertrend", "market_trend"],
            "Momentum": ["rsi_", "stochrsi", "cci_", "willr_", "roc_", "mom_", "tsi_"],
            "Volatility": ["atr_", "bb_", "BBU_", "BBL_", "BBM_", "kc_", "donchian", "hvol", "body_", "candle_", "wick"],
            "Volume": ["vol_", "relative_volume", "obv", "mfi_", "ad_line", "cmf_"],
            "Structure": ["swing_", "bos", "choch", "fvg_", "ob_", "liquidity_"],
            "S/R": ["nearest_support", "nearest_resistance", "sr_", "dist_to_", "round_"],
            "Session": ["session_", "killzone_", "hour_", "dow_", "is_monday", "is_friday"],
            "Divergence": ["divergence"],
            "Wyckoff": ["wyckoff_", "effort_vs_result"],
        }

        summary = {}
        for cat_name, prefixes in categories.items():
            cols = [c for c in df.columns if any(p in c for p in prefixes)]
            summary[cat_name] = len(cols)

        return summary
'''

# ============================================================
# FILE 8: run_features.py - Test script
# ============================================================
FILES["run_features.py"] = r'''"""
XAUUSD AI Trading System - Feature Engine Test
Run this to compute all features on your saved data.

Usage:
    python run_features.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collector.mt5_collector import MT5Connector, DataCollector
from src.feature_engine.engine import FeatureEngine
from src.utils.logger import setup_logger

logger = setup_logger("run_features")


def main():
    print("\n" + "=" * 70)
    print("  XAUUSD AI TRADING SYSTEM")
    print("  Phase 1 - Session 2: Feature Engine")
    print("=" * 70 + "\n")

    # --- Step 1: Load data ---
    logger.info("Step 1: Connecting to MT5 and fetching fresh data...")
    connector = MT5Connector()

    if not connector.connect():
        logger.error("Cannot connect to MT5!")
        return

    collector = DataCollector(connector)
    collector.check_symbol()
    data = collector.fetch_all_timeframes()
    connector.disconnect()

    if not data:
        logger.error("No data to process!")
        return

    # --- Step 2: Compute features ---
    logger.info("\nStep 2: Computing features for all timeframes...\n")
    engine = FeatureEngine()
    featured_data = engine.compute_multi_timeframe(data)

    # --- Step 3: Summary ---
    print("\n" + "=" * 70)
    print("  FEATURE ENGINE RESULTS")
    print("=" * 70)

    for tf_name, df in featured_data.items():
        summary = engine.get_feature_summary(df)
        total_features = sum(summary.values())

        print(f"\n  {tf_name}: {len(df)} candles x {len(df.columns)} columns ({total_features} features)")
        for cat, count in summary.items():
            if count > 0:
                print(f"    {cat:>12}: {count} features")

    # --- Step 4: Save featured data ---
    logger.info("\nStep 3: Saving featured data...")
    data_dir = PROJECT_ROOT / "data" / "historical"
    data_dir.mkdir(parents=True, exist_ok=True)

    for tf_name, df in featured_data.items():
        filepath = data_dir / f"XAUUSD_{tf_name}_features.parquet"
        df.to_parquet(filepath, engine="pyarrow")
        size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"  Saved {tf_name}: {len(df.columns)} columns -> {filepath.name} ({size_mb:.2f} MB)")

    # --- Step 5: Show sample data ---
    print("\n" + "=" * 70)
    print("  SAMPLE: Last 3 H1 candles with key features")
    print("=" * 70)

    h1 = featured_data.get("H1")
    if h1 is not None:
        key_cols = ["close", "ema_21", "rsi_14", "atr_14", "market_trend",
                    "wyckoff_phase", "sr_strength", "killzone_active"]
        available = [c for c in key_cols if c in h1.columns]
        print(h1[available].tail(3).to_string())

    print("\n" + "=" * 70)
    print("  Phase 1 - Session 2 COMPLETE!")
    print("  Featured data saved in: data/historical/")
    print("  Next: Build strategy detectors")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
'''


# ============================================================
# Create all files
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("  Feature Engine Setup")
    print("=" * 60 + "\n")

    for filepath, content in FILES.items():
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        print(f"  [FILE] {filepath}")

    print(f"\n  Created {len(FILES)} files.")
    print("\n  Now run:  python run_features.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
