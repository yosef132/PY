"""
Strategy 3: ICT (Inner Circle Trader)
Detects setups based on market structure, order blocks, FVGs, and liquidity sweeps.
Best for: Trend continuation after liquidity grabs during killzones.
"""

import pandas as pd
import numpy as np
from src.strategies.base import BaseStrategy, Signal
from src.utils.logger import setup_logger

logger = setup_logger("strategy_ict")


class ICTStrategy(BaseStrategy):
    name = "ict"

    def detect_signals(self, df: pd.DataFrame, timeframe: str = "") -> list:
        if not self.enabled:
            return []

        signals = []
        n = len(df)
        killzone_only = self.config.get("killzone_only", True)

        for i in range(5, n):
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            atr = self._get_atr(df, i)

            # Skip if not in killzone (when enabled)
            if killzone_only and row.get("killzone_active", 0) == 0:
                continue

            trend = row.get("market_trend", 0)

            # --- ICT BULLISH SETUP ---
            # Conditions: bullish market structure + price returns to bullish OB or FVG
            if trend >= 1:  # Bullish trend (BOS confirmed)
                ob_high = row.get("nearest_ob_bull_high", np.nan)
                ob_low = row.get("nearest_ob_bull_low", np.nan)
                fvg_top = row.get("nearest_fvg_bull_top", np.nan)
                fvg_bot = row.get("nearest_fvg_bull_bot", np.nan)

                entry_zone = False
                entry_type = ""

                # Check if price is in a bullish Order Block zone
                if pd.notna(ob_high) and pd.notna(ob_low):
                    if ob_low <= row["low"] <= ob_high:
                        entry_zone = True
                        entry_type = "order_block"

                # Check if price is in a bullish FVG zone
                if not entry_zone and pd.notna(fvg_top) and pd.notna(fvg_bot):
                    if fvg_bot <= row["low"] <= fvg_top:
                        entry_zone = True
                        entry_type = "fvg"

                if entry_zone and row["close"] > row["open"]:  # Bullish reaction
                    confidence = 0.60

                    # CHOCH adds extra confidence (reversal confirmed)
                    if row.get("choch", 0) == 1:
                        confidence += 0.15

                    # BOS continuation
                    if row.get("bos", 0) == 1:
                        confidence += 0.10

                    # Liquidity sweep below before reversal
                    if row.get("liquidity_below", 0) >= 2:
                        confidence += 0.10
                        entry_type += "+liq_sweep"

                    # Killzone timing
                    if row.get("killzone_active", 0) == 1:
                        confidence += 0.05

                    sl = row["low"] - atr * 1.0
                    tp = row["close"] + atr * 3.5

                    # Use nearest liquidity above as target if available
                    liq_above = row.get("nearest_fvg_bear_bot", np.nan)
                    if pd.notna(liq_above) and liq_above > row["close"]:
                        tp = liq_above

                    signals.append(Signal(
                        direction="BUY",
                        strategy_name=self.name,
                        confidence=min(confidence, 0.95),
                        timeframe=timeframe,
                        entry_price=row["close"],
                        stop_loss=sl,
                        take_profit=tp,
                        timestamp=df.index[i],
                        reason=f"ICT bullish: {entry_type} entry in uptrend",
                        tags=["ict", "bullish", entry_type]
                    ))

            # --- ICT BEARISH SETUP ---
            if trend <= -1:  # Bearish trend
                ob_high = row.get("nearest_ob_bear_high", np.nan)
                ob_low = row.get("nearest_ob_bear_low", np.nan)
                fvg_top = row.get("nearest_fvg_bear_top", np.nan)
                fvg_bot = row.get("nearest_fvg_bear_bot", np.nan)

                entry_zone = False
                entry_type = ""

                if pd.notna(ob_high) and pd.notna(ob_low):
                    if ob_low <= row["high"] <= ob_high:
                        entry_zone = True
                        entry_type = "order_block"

                if not entry_zone and pd.notna(fvg_top) and pd.notna(fvg_bot):
                    if fvg_bot <= row["high"] <= fvg_top:
                        entry_zone = True
                        entry_type = "fvg"

                if entry_zone and row["close"] < row["open"]:  # Bearish reaction
                    confidence = 0.60

                    if row.get("choch", 0) == -1:
                        confidence += 0.15
                    if row.get("bos", 0) == -1:
                        confidence += 0.10
                    if row.get("liquidity_above", 0) >= 2:
                        confidence += 0.10
                        entry_type += "+liq_sweep"
                    if row.get("killzone_active", 0) == 1:
                        confidence += 0.05

                    sl = row["high"] + atr * 1.0
                    tp = row["close"] - atr * 3.5

                    liq_below = row.get("nearest_fvg_bull_top", np.nan)
                    if pd.notna(liq_below) and liq_below < row["close"]:
                        tp = liq_below

                    signals.append(Signal(
                        direction="SELL",
                        strategy_name=self.name,
                        confidence=min(confidence, 0.95),
                        timeframe=timeframe,
                        entry_price=row["close"],
                        stop_loss=sl,
                        take_profit=tp,
                        timestamp=df.index[i],
                        reason=f"ICT bearish: {entry_type} entry in downtrend",
                        tags=["ict", "bearish", entry_type]
                    ))

        logger.info(f"  [{timeframe}] ICT Strategy: {len(signals)} signals")
        return signals
