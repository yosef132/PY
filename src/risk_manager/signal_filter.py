"""
Signal Filter
First line of defense: filters raw strategy signals based on quality rules.
Reduces thousands of signals to only tradeable setups.
"""

import pandas as pd
import numpy as np
from src.utils.config import get_settings
from src.utils.logger import setup_logger

logger = setup_logger("signal_filter")


class SignalFilter:
    """
    Filters raw signals based on quality criteria.
    This runs BEFORE the Risk Manager to remove obviously bad signals.
    """

    def __init__(self):
        settings = get_settings()
        self.risk_config = settings.get("risk", {})
        self.session_config = settings.get("sessions", {})

        # Filter thresholds
        self.min_rr = self.risk_config.get("min_risk_reward", 2.0)
        self.min_confidence = 0.60
        self.max_signals_per_bar = 1  # Only 1 signal per candle
        self.min_sl_distance_pct = 0.05  # SL must be at least 0.05% from entry
        self.max_sl_distance_pct = 2.0   # SL can't be more than 2% from entry

    def filter_signals(self, signals: list, timeframe: str = "") -> list:
        """
        Filter raw signals and return only quality setups.
        """
        if not signals:
            return []

        original_count = len(signals)
        filtered = []

        for sig in signals:
            # --- Filter 1: Minimum Risk-Reward Ratio ---
            if sig.risk_reward < self.min_rr:
                continue

            # --- Filter 2: Minimum Confidence ---
            if sig.confidence < self.min_confidence:
                continue

            # --- Filter 3: SL Distance Sanity Check ---
            if sig.entry_price <= 0:
                continue

            sl_distance_pct = abs(sig.entry_price - sig.stop_loss) / sig.entry_price * 100

            # SL too tight = will get stopped out by noise
            if sl_distance_pct < self.min_sl_distance_pct:
                continue

            # SL too wide = risking too much per trade
            if sl_distance_pct > self.max_sl_distance_pct:
                continue

            # --- Filter 4: TP must be on the correct side ---
            if sig.direction == "BUY":
                if sig.take_profit <= sig.entry_price or sig.stop_loss >= sig.entry_price:
                    continue
            elif sig.direction == "SELL":
                if sig.take_profit >= sig.entry_price or sig.stop_loss <= sig.entry_price:
                    continue
            else:
                continue

            filtered.append(sig)

        # --- Filter 5: Deduplicate — max 1 signal per timestamp per direction ---
        seen = set()
        deduped = []
        for sig in filtered:
            key = (sig.timestamp, sig.direction)
            if key not in seen:
                seen.add(key)
                deduped.append(sig)

        # Sort by confidence (best first)
        deduped.sort(key=lambda s: s.confidence, reverse=True)

        logger.info(f"  [{timeframe}] Signal Filter: {original_count} -> {len(deduped)} "
                     f"(removed {original_count - len(deduped)} low-quality signals)")

        return deduped
