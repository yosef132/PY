"""
Strategy 5: CRT (Candle Range Theory)
Analyzes higher timeframe candle ranges for accumulation-manipulation-distribution.
When a candle's high/low is swept (liquidity grab) and price returns inside the range,
the real move targets the opposite extreme.
Best for: Gold during London/NY sessions on H4 candles analyzed on M15.
"""

import pandas as pd
import numpy as np
from src.strategies.base import BaseStrategy, Signal
from src.utils.logger import setup_logger

logger = setup_logger("strategy_crt")


class CRTStrategy(BaseStrategy):
    name = "crt"

    def detect_signals(self, df: pd.DataFrame, timeframe: str = "") -> list:
        if not self.enabled:
            return []

        signals = []
        n = len(df)

        for i in range(3, n):
            row = df.iloc[i]
            prev1 = df.iloc[i - 1]
            prev2 = df.iloc[i - 2]
            atr = self._get_atr(df, i)

            # --- CRT BULLISH: Previous candle low is swept, then price recovers ---
            # Candle i-2 establishes a range
            # Candle i-1 breaks below the range low (manipulation / liquidity sweep)
            # Candle i closes back inside the range (distribution phase begins upward)
            range_high = prev2["high"]
            range_low = prev2["low"]
            range_size = range_high - range_low

            # Only trade significant ranges (> 0.5 ATR)
            if range_size < atr * 0.5:
                continue

            # Bullish CRT: sweep low then recover
            if (prev1["low"] < range_low                    # Candle broke below range
                and prev1["close"] < range_low              # Closed below (manipulation)
                and row["close"] > range_low                # Current candle recovers inside
                and row["close"] > row["open"]):            # Bullish candle

                confidence = 0.55

                # Larger range = higher probability
                if range_size > atr * 1.5:
                    confidence += 0.10

                # Volume spike on the sweep = stronger
                rel_vol = prev1.get("relative_volume", 1.0)
                if pd.notna(rel_vol) and rel_vol > 1.3:
                    confidence += 0.05

                # Killzone timing
                if row.get("killzone_active", 0) == 1:
                    confidence += 0.10

                # Market trend alignment
                if row.get("market_trend", 0) >= 1:
                    confidence += 0.10

                # RSI confirmation
                if row.get("rsi_14", 50) < 40:
                    confidence += 0.05

                sl = min(prev1["low"], row["low"]) - atr * 0.5
                tp = range_high  # CRT targets opposite extreme of the range
                tp2 = range_high + range_size * 0.5  # Extended target

                signals.append(Signal(
                    direction="BUY",
                    strategy_name=self.name,
                    confidence=min(confidence, 0.95),
                    timeframe=timeframe,
                    entry_price=row["close"],
                    stop_loss=sl,
                    take_profit=tp,
                    take_profit_2=tp2,
                    timestamp=df.index[i],
                    reason=f"CRT bullish: low sweep {range_low:.0f} -> recovery (range={range_size:.0f})",
                    tags=["crt", "bullish", "liquidity_sweep"]
                ))

            # Bearish CRT: sweep high then reverse
            if (prev1["high"] > range_high
                and prev1["close"] > range_high
                and row["close"] < range_high
                and row["close"] < row["open"]):

                confidence = 0.55

                if range_size > atr * 1.5:
                    confidence += 0.10

                rel_vol = prev1.get("relative_volume", 1.0)
                if pd.notna(rel_vol) and rel_vol > 1.3:
                    confidence += 0.05

                if row.get("killzone_active", 0) == 1:
                    confidence += 0.10

                if row.get("market_trend", 0) <= -1:
                    confidence += 0.10

                if row.get("rsi_14", 50) > 60:
                    confidence += 0.05

                sl = max(prev1["high"], row["high"]) + atr * 0.5
                tp = range_low
                tp2 = range_low - range_size * 0.5

                signals.append(Signal(
                    direction="SELL",
                    strategy_name=self.name,
                    confidence=min(confidence, 0.95),
                    timeframe=timeframe,
                    entry_price=row["close"],
                    stop_loss=sl,
                    take_profit=tp,
                    take_profit_2=tp2,
                    timestamp=df.index[i],
                    reason=f"CRT bearish: high sweep {range_high:.0f} -> reversal (range={range_size:.0f})",
                    tags=["crt", "bearish", "liquidity_sweep"]
                ))

        logger.info(f"  [{timeframe}] CRT Strategy: {len(signals)} signals")
        return signals
