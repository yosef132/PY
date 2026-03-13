"""
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
