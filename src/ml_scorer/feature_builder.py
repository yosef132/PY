"""
ML Feature Builder
Extracts features from each signal + market context to build training data.
This converts a Signal object + DataFrame row into a flat feature vector for XGBoost.
"""

import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger("ml_features")


class FeatureBuilder:
    """Builds ML feature vectors from signals and market data."""

    # Features we extract from each signal's market context
    MARKET_FEATURES = [
        "rsi_14", "rsi_7", "ema_alignment", "atr_14", "atr_pct",
        "bb_width", "bb_position", "hvol_20",
        "market_trend", "bos", "choch",
        "sr_strength", "dist_to_support_pct", "dist_to_resistance_pct", "sr_position",
        "wyckoff_phase", "effort_vs_result",
        "relative_volume", "mfi_14", "cmf_20",
        "killzone_active", "session_london", "session_ny", "session_overlap",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "liquidity_above", "liquidity_below",
        "body_atr_ratio", "range_atr_ratio", "upper_wick_pct", "lower_wick_pct",
    ]

    # Features from the signal itself
    SIGNAL_FEATURES = [
        "confidence", "risk_reward",
    ]

    # Strategy one-hot encoding
    STRATEGY_NAMES = [
        "support_resistance", "rsi_divergence", "ict", "wyckoff", "crt", "sk_system"
    ]

    def get_feature_names(self) -> list:
        """Get all feature column names."""
        names = []
        names.extend(self.MARKET_FEATURES)
        names.extend(self.SIGNAL_FEATURES)
        names.append("is_buy")  # 1 for BUY, 0 for SELL
        names.extend([f"strat_{s}" for s in self.STRATEGY_NAMES])
        return names

    def extract_features(self, signal, df_row: pd.Series) -> dict:
        """
        Extract a feature vector from a signal + its market context row.

        Args:
            signal: Signal object from a strategy detector
            df_row: The DataFrame row at the signal's timestamp

        Returns:
            Dict of feature_name -> value
        """
        features = {}

        # Market context features
        for feat in self.MARKET_FEATURES:
            val = df_row.get(feat, np.nan)
            features[feat] = float(val) if pd.notna(val) else 0.0

        # Signal features
        features["confidence"] = signal.confidence
        features["risk_reward"] = signal.risk_reward

        # Direction
        features["is_buy"] = 1.0 if signal.direction == "BUY" else 0.0

        # Strategy one-hot
        for strat in self.STRATEGY_NAMES:
            features[f"strat_{strat}"] = 1.0 if signal.strategy_name == strat else 0.0

        return features

    def build_training_data(self, trades: list, featured_df: pd.DataFrame) -> tuple:
        """
        Build training dataset from completed trades.

        Args:
            trades: List of completed trade dicts (from backtester)
            featured_df: The featured DataFrame used during backtesting

        Returns:
            (X: np.array of features, y: np.array of labels, feature_names: list)
        """
        feature_names = self.get_feature_names()
        rows = []
        labels = []

        for trade in trades:
            # Find the closest row in the DataFrame
            entry_time = trade.get("entry_time")
            if entry_time is None:
                continue

            # Get the market context at entry time
            if entry_time in featured_df.index:
                df_row = featured_df.loc[entry_time]
            else:
                # Find nearest timestamp
                idx = featured_df.index.get_indexer([entry_time], method="nearest")[0]
                if idx < 0 or idx >= len(featured_df):
                    continue
                df_row = featured_df.iloc[idx]

            # Build feature vector
            features = {}
            for feat in self.MARKET_FEATURES:
                val = df_row.get(feat, np.nan)
                features[feat] = float(val) if pd.notna(val) else 0.0

            features["confidence"] = trade.get("confidence", 0.5)
            features["risk_reward"] = trade.get("risk_reward", 1.0)
            features["is_buy"] = 1.0 if trade["direction"] == "BUY" else 0.0

            for strat in self.STRATEGY_NAMES:
                features[f"strat_{strat}"] = 1.0 if trade.get("strategy") == strat else 0.0

            # Feature vector in correct order
            row = [features.get(name, 0.0) for name in feature_names]
            rows.append(row)

            # Label: 1 = profitable trade, 0 = losing trade
            labels.append(1 if trade["pnl"] > 0 else 0)

        X = np.array(rows)
        y = np.array(labels)

        logger.info(f"  Built training data: {len(rows)} samples, {len(feature_names)} features")
        logger.info(f"  Win rate in data: {y.mean()*100:.1f}%")

        return X, y, feature_names
