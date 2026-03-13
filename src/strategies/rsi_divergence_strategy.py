"""
Strategy 2: RSI Divergence
Detects bullish and bearish RSI divergences with price action confirmation.
Best for: Catching reversals at exhaustion points.
"""

import pandas as pd
import numpy as np
from src.strategies.base import BaseStrategy, Signal
from src.utils.logger import setup_logger

logger = setup_logger("strategy_rsi_div")


class RSIDivergenceStrategy(BaseStrategy):
    name = "rsi_divergence"

    def detect_signals(self, df: pd.DataFrame, timeframe: str = "") -> list:
        if not self.enabled:
            return []

        signals = []
        n = len(df)

        for i in range(2, n):
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            atr = self._get_atr(df, i)

            # --- BULLISH RSI DIVERGENCE ---
            if row.get("rsi_bull_divergence", 0) == 1:
                confidence = 0.55

                # Confirmation: bullish candle after divergence
                if row["close"] > row["open"]:
                    confidence += 0.1

                # Stronger if RSI was in oversold zone
                rsi = row.get("rsi_14", 50)
                if rsi < 30:
                    confidence += 0.15
                elif rsi < 40:
                    confidence += 0.05

                # Stronger near support
                if row.get("sr_strength", 0) >= 2:
                    confidence += 0.1

                # Killzone bonus
                if row.get("killzone_active", 0) == 1:
                    confidence += 0.05

                sl = row["low"] - atr * 1.5
                tp = row["close"] + atr * 3.0

                signals.append(Signal(
                    direction="BUY",
                    strategy_name=self.name,
                    confidence=min(confidence, 0.95),
                    timeframe=timeframe,
                    entry_price=row["close"],
                    stop_loss=sl,
                    take_profit=tp,
                    timestamp=df.index[i],
                    reason=f"Bullish RSI divergence (RSI={rsi:.1f})",
                    tags=["rsi_divergence", "bullish"]
                ))

            # --- BEARISH RSI DIVERGENCE ---
            if row.get("rsi_bear_divergence", 0) == 1:
                confidence = 0.55

                if row["close"] < row["open"]:
                    confidence += 0.1

                rsi = row.get("rsi_14", 50)
                if rsi > 70:
                    confidence += 0.15
                elif rsi > 60:
                    confidence += 0.05

                if row.get("sr_strength", 0) >= 2:
                    confidence += 0.1

                if row.get("killzone_active", 0) == 1:
                    confidence += 0.05

                sl = row["high"] + atr * 1.5
                tp = row["close"] - atr * 3.0

                signals.append(Signal(
                    direction="SELL",
                    strategy_name=self.name,
                    confidence=min(confidence, 0.95),
                    timeframe=timeframe,
                    entry_price=row["close"],
                    stop_loss=sl,
                    take_profit=tp,
                    timestamp=df.index[i],
                    reason=f"Bearish RSI divergence (RSI={rsi:.1f})",
                    tags=["rsi_divergence", "bearish"]
                ))

        logger.info(f"  [{timeframe}] RSI Divergence: {len(signals)} signals")
        return signals
