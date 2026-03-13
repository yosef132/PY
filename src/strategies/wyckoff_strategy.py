"""
Strategy 4: Wyckoff Method
Detects Springs (buy at false breakdown) and Upthrusts (sell at false breakout).
Combined with phase analysis and effort-vs-result confirmation.
Best for: Catching major reversals at accumulation/distribution boundaries.
"""

import pandas as pd
import numpy as np
from src.strategies.base import BaseStrategy, Signal
from src.utils.logger import setup_logger

logger = setup_logger("strategy_wyckoff")


class WyckoffStrategy(BaseStrategy):
    name = "wyckoff"

    def detect_signals(self, df: pd.DataFrame, timeframe: str = "") -> list:
        if not self.enabled:
            return []

        signals = []
        n = len(df)

        for i in range(2, n):
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            atr = self._get_atr(df, i)
            phase = row.get("wyckoff_phase", 0)

            # --- WYCKOFF SPRING (BUY) ---
            # Spring = false breakdown below support in accumulation phase
            if row.get("wyckoff_spring", 0) == 1:
                confidence = 0.60

                # Accumulation phase = highest probability spring
                if phase == 1:
                    confidence += 0.15
                elif phase == 0:
                    confidence += 0.05

                # Effort vs Result: low effort on the spring = strong signal
                evr = row.get("effort_vs_result", 1.0)
                if pd.notna(evr) and evr < 0.8:
                    confidence += 0.10

                # Volume confirmation
                rel_vol = row.get("relative_volume", 1.0)
                if pd.notna(rel_vol) and rel_vol > 1.5:
                    confidence += 0.05

                # RSI oversold confirmation
                if row.get("rsi_14", 50) < 35:
                    confidence += 0.05

                sl = row["low"] - atr * 1.5
                tp = row["close"] + atr * 4.0

                # Target the top of the range if available
                resistance = row.get("nearest_resistance", np.nan)
                if pd.notna(resistance) and resistance > row["close"]:
                    tp = resistance

                signals.append(Signal(
                    direction="BUY",
                    strategy_name=self.name,
                    confidence=min(confidence, 0.95),
                    timeframe=timeframe,
                    entry_price=row["close"],
                    stop_loss=sl,
                    take_profit=tp,
                    timestamp=df.index[i],
                    reason=f"Wyckoff Spring (phase={phase:.0f})",
                    tags=["wyckoff", "spring", f"phase_{phase:.0f}"]
                ))

            # --- WYCKOFF UPTHRUST (SELL) ---
            if row.get("wyckoff_upthrust", 0) == 1:
                confidence = 0.60

                # Distribution phase = highest probability upthrust
                if phase == 3:
                    confidence += 0.15
                elif phase == 0:
                    confidence += 0.05

                evr = row.get("effort_vs_result", 1.0)
                if pd.notna(evr) and evr < 0.8:
                    confidence += 0.10

                rel_vol = row.get("relative_volume", 1.0)
                if pd.notna(rel_vol) and rel_vol > 1.5:
                    confidence += 0.05

                if row.get("rsi_14", 50) > 65:
                    confidence += 0.05

                sl = row["high"] + atr * 1.5
                tp = row["close"] - atr * 4.0

                support = row.get("nearest_support", np.nan)
                if pd.notna(support) and support < row["close"]:
                    tp = support

                signals.append(Signal(
                    direction="SELL",
                    strategy_name=self.name,
                    confidence=min(confidence, 0.95),
                    timeframe=timeframe,
                    entry_price=row["close"],
                    stop_loss=sl,
                    take_profit=tp,
                    timestamp=df.index[i],
                    reason=f"Wyckoff Upthrust (phase={phase:.0f})",
                    tags=["wyckoff", "upthrust", f"phase_{phase:.0f}"]
                ))

        logger.info(f"  [{timeframe}] Wyckoff Strategy: {len(signals)} signals")
        return signals
