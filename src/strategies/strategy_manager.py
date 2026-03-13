"""
Strategy Manager
Orchestrates all strategy detectors and collects signals.
"""

import pandas as pd
from src.strategies.support_resistance_strategy import SupportResistanceStrategy
from src.strategies.rsi_divergence_strategy import RSIDivergenceStrategy
from src.strategies.ict_strategy import ICTStrategy
from src.strategies.wyckoff_strategy import WyckoffStrategy
from src.strategies.crt_strategy import CRTStrategy
from src.strategies.sk_system_strategy import SKSystemStrategy
from src.utils.config import get_strategies
from src.utils.logger import setup_logger

logger = setup_logger("strategy_manager")


class StrategyManager:
    """Manages all strategy detectors and collects signals."""

    def __init__(self):
        config = get_strategies().get("strategies", {})

        self.strategies = [
            SupportResistanceStrategy(config.get("support_resistance", {})),
            RSIDivergenceStrategy(config.get("rsi_divergence", {})),
            ICTStrategy(config.get("ict", {})),
            WyckoffStrategy(config.get("wyckoff", {})),
            CRTStrategy(config.get("crt", {})),
            SKSystemStrategy(config.get("sk_system", {})),
        ]

        enabled = [s.name for s in self.strategies if s.enabled]
        disabled = [s.name for s in self.strategies if not s.enabled]
        logger.info(f"Strategies loaded: {len(enabled)} enabled, {len(disabled)} disabled")
        logger.info(f"  Enabled: {', '.join(enabled)}")
        if disabled:
            logger.info(f"  Disabled: {', '.join(disabled)}")

    def scan_all(self, df: pd.DataFrame, timeframe: str = "") -> list:
        """Run all enabled strategies on a DataFrame and collect signals."""
        all_signals = []

        for strategy in self.strategies:
            if not strategy.enabled:
                continue
            try:
                signals = strategy.detect_signals(df, timeframe)
                all_signals.extend(signals)
            except Exception as e:
                logger.error(f"Error in {strategy.name}: {e}")

        # Sort by confidence (highest first)
        all_signals.sort(key=lambda s: s.confidence, reverse=True)

        return all_signals

    def scan_multi_timeframe(self, featured_data: dict) -> dict:
        """Scan all timeframes and collect signals."""
        all_signals = {}

        for tf_name, df in featured_data.items():
            signals = self.scan_all(df, tf_name)
            all_signals[tf_name] = signals
            if signals:
                logger.info(f"  {tf_name}: {len(signals)} total signals (top: {signals[0]})")

        return all_signals
