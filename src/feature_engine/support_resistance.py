"""
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
