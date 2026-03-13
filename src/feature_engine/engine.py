"""
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
