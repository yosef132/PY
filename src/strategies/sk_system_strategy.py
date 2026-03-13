"""
Strategy 6: SK System (Stefan Kassing)
Multi-timeframe Fibonacci-based trend continuation strategy.
Uses EMA equilibrium detection, Fibonacci retracements, and MACD alignment.
Best for: High risk-reward trend continuation trades (targets 1:3 to 1:5).
"""

import pandas as pd
import numpy as np
from src.strategies.base import BaseStrategy, Signal
from src.utils.logger import setup_logger

logger = setup_logger("strategy_sk")


class SKSystemStrategy(BaseStrategy):
    name = "sk_system"

    def detect_signals(self, df: pd.DataFrame, timeframe: str = "") -> list:
        if not self.enabled:
            return []

        signals = []
        n = len(df)

        for i in range(55, n):
            row = df.iloc[i]
            atr = self._get_atr(df, i)

            # --- SK STEP 1: Identify trend via EMA alignment ---
            ema_align = row.get("ema_alignment", 0)
            # Need at least +2 for bullish or -2 for bearish (3 EMAs aligned)
            if abs(ema_align) < 2:
                continue

            trend_bullish = ema_align >= 2
            trend_bearish = ema_align <= -2

            # --- SK STEP 2: Price has pulled back to equilibrium (EMA 21 or 50) ---
            ema_21 = row.get("ema_21", np.nan)
            ema_50 = row.get("ema_50", np.nan)

            if pd.isna(ema_21) or pd.isna(ema_50):
                continue

            # Check pullback to EMA zone
            at_equilibrium = False
            if trend_bullish:
                # Price pulled back near EMA 21 or 50 from above
                if row["low"] <= ema_21 * 1.002 and row["close"] > ema_21:
                    at_equilibrium = True
                elif row["low"] <= ema_50 * 1.003 and row["close"] > ema_50:
                    at_equilibrium = True
            elif trend_bearish:
                if row["high"] >= ema_21 * 0.998 and row["close"] < ema_21:
                    at_equilibrium = True
                elif row["high"] >= ema_50 * 0.997 and row["close"] < ema_50:
                    at_equilibrium = True

            if not at_equilibrium:
                continue

            # --- SK STEP 3: Fibonacci retracement zone ---
            # Look back for recent swing to measure fib levels
            lookback = 50
            window = df.iloc[max(0, i - lookback):i + 1]

            if trend_bullish:
                swing_low = window["low"].min()
                swing_high = window["high"].max()
                fib_range = swing_high - swing_low
                fib_618 = swing_high - fib_range * 0.618
                fib_50 = swing_high - fib_range * 0.5

                # Price should be near 50-61.8% retracement
                in_fib_zone = fib_618 <= row["close"] <= fib_50
            else:
                swing_low = window["low"].min()
                swing_high = window["high"].max()
                fib_range = swing_high - swing_low
                fib_618 = swing_low + fib_range * 0.618
                fib_50 = swing_low + fib_range * 0.5

                in_fib_zone = fib_50 <= row["close"] <= fib_618

            # --- SK STEP 4: MACD confirmation ---
            macd_col = [c for c in df.columns if "MACDh_" in c]
            macd_confirming = False
            if macd_col:
                macd_hist = row.get(macd_col[0], 0)
                prev_macd = df.iloc[i - 1].get(macd_col[0], 0)
                if pd.notna(macd_hist) and pd.notna(prev_macd):
                    if trend_bullish and macd_hist > prev_macd:
                        macd_confirming = True
                    elif trend_bearish and macd_hist < prev_macd:
                        macd_confirming = True

            # --- Build signal ---
            if at_equilibrium:
                confidence = 0.50

                if in_fib_zone:
                    confidence += 0.15
                if macd_confirming:
                    confidence += 0.10
                if abs(ema_align) >= 3:  # Perfect EMA stack
                    confidence += 0.10
                if row.get("killzone_active", 0) == 1:
                    confidence += 0.05

                # ADX trend strength
                adx_col = [c for c in df.columns if "ADX_" in c]
                if adx_col:
                    adx_val = row.get(adx_col[0], 0)
                    if pd.notna(adx_val) and adx_val > 25:
                        confidence += 0.05

                if confidence < 0.55:
                    continue

                if trend_bullish:
                    sl = row["low"] - atr * 1.5
                    tp = row["close"] + atr * 5.0  # SK targets 1:5 R:R

                    signals.append(Signal(
                        direction="BUY",
                        strategy_name=self.name,
                        confidence=min(confidence, 0.95),
                        timeframe=timeframe,
                        entry_price=row["close"],
                        stop_loss=sl,
                        take_profit=tp,
                        take_profit_2=row["close"] + atr * 3.0,
                        timestamp=df.index[i],
                        reason=f"SK System bullish: EMA pullback + fib zone (align={ema_align})",
                        tags=["sk_system", "bullish", "fib_retracement"]
                    ))

                elif trend_bearish:
                    sl = row["high"] + atr * 1.5
                    tp = row["close"] - atr * 5.0

                    signals.append(Signal(
                        direction="SELL",
                        strategy_name=self.name,
                        confidence=min(confidence, 0.95),
                        timeframe=timeframe,
                        entry_price=row["close"],
                        stop_loss=sl,
                        take_profit=tp,
                        take_profit_2=row["close"] - atr * 3.0,
                        timestamp=df.index[i],
                        reason=f"SK System bearish: EMA pullback + fib zone (align={ema_align})",
                        tags=["sk_system", "bearish", "fib_retracement"]
                    ))

        logger.info(f"  [{timeframe}] SK System: {len(signals)} signals")
        return signals
