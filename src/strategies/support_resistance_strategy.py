"""
Strategy 1: Support & Resistance
Detects bounces off confirmed S/R levels with confirmation candle patterns.
Best for: Range-bound markets, counter-trend entries at strong levels.
"""

import pandas as pd
import numpy as np
from src.strategies.base import BaseStrategy, Signal
from src.utils.logger import setup_logger

logger = setup_logger("strategy_sr")


class SupportResistanceStrategy(BaseStrategy):
    name = "support_resistance"

    def detect_signals(self, df: pd.DataFrame, timeframe: str = "") -> list:
        if not self.enabled:
            return []

        signals = []
        n = len(df)

        for i in range(50, n):
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            atr = self._get_atr(df, i)

            # Skip if no S/R data
            if pd.isna(row.get("nearest_support")) or pd.isna(row.get("nearest_resistance")):
                continue

            support = row["nearest_support"]
            resistance = row["nearest_resistance"]
            close = row["close"]
            low = row["low"]
            high = row["high"]
            strength = row.get("sr_strength", 0)

            # --- BOUNCE OFF SUPPORT (BUY) ---
            # Price touches support zone AND closes above it with bullish candle
            dist_to_support = close - support
            if (0 < dist_to_support < atr * 0.5      # Close is near support
                and low <= support + atr * 0.2         # Low touched the zone
                and close > row["open"]                 # Bullish candle
                and strength >= 2):                     # Level has been tested 2+ times

                confidence = min(0.5 + strength * 0.1, 0.85)

                # Check for confirmation: RSI oversold or divergence
                if row.get("rsi_14", 50) < 35:
                    confidence += 0.1
                if row.get("rsi_bull_divergence", 0) == 1:
                    confidence += 0.15

                sl = support - atr * 1.5
                tp = support + (resistance - support) * 0.7  # Target 70% of range

                signals.append(Signal(
                    direction="BUY",
                    strategy_name=self.name,
                    confidence=min(confidence, 0.95),
                    timeframe=timeframe,
                    entry_price=close,
                    stop_loss=sl,
                    take_profit=tp,
                    timestamp=df.index[i],
                    reason=f"Bounce off support {support:.0f} (strength={strength:.0f})",
                    tags=["sr_bounce", "support"]
                ))

            # --- REJECTION FROM RESISTANCE (SELL) ---
            dist_to_resistance = resistance - close
            if (0 < dist_to_resistance < atr * 0.5
                and high >= resistance - atr * 0.2
                and close < row["open"]                 # Bearish candle
                and strength >= 2):

                confidence = min(0.5 + strength * 0.1, 0.85)

                if row.get("rsi_14", 50) > 65:
                    confidence += 0.1
                if row.get("rsi_bear_divergence", 0) == 1:
                    confidence += 0.15

                sl = resistance + atr * 1.5
                tp = resistance - (resistance - support) * 0.7

                signals.append(Signal(
                    direction="SELL",
                    strategy_name=self.name,
                    confidence=min(confidence, 0.95),
                    timeframe=timeframe,
                    entry_price=close,
                    stop_loss=sl,
                    take_profit=tp,
                    timestamp=df.index[i],
                    reason=f"Rejection from resistance {resistance:.0f} (strength={strength:.0f})",
                    tags=["sr_rejection", "resistance"]
                ))

        logger.info(f"  [{timeframe}] S&R Strategy: {len(signals)} signals")
        return signals
