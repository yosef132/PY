"""
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
