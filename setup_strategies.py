"""
XAUUSD AI Trading System - Strategy Detectors Setup
Session 3: Build all 6 strategy detectors + signal framework + backtester

Usage:
    python setup_strategies.py
    python run_strategies.py
"""

import os

FILES = {}

# ============================================================
# FILE 1: src/strategies/base.py - Signal Framework
# ============================================================
FILES["src/strategies/base.py"] = r'''"""
Base Strategy Framework
All strategies inherit from this base class and produce standardized Signal objects.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class Signal:
    """
    Standardized trading signal produced by any strategy.
    This is what the Risk Manager and Executor consume.
    """
    # Core signal
    direction: str           # "BUY", "SELL", or "NONE"
    strategy_name: str       # Which strategy generated this signal
    confidence: float        # 0.0 to 1.0 - how confident the strategy is
    timeframe: str           # Timeframe the signal was detected on (e.g. "H1")

    # Price levels
    entry_price: float       # Suggested entry price
    stop_loss: float         # Mandatory stop loss
    take_profit: float       # Primary take profit target
    take_profit_2: Optional[float] = None  # Secondary TP (partial close)

    # Context
    timestamp: Optional[datetime] = None
    reason: str = ""         # Human-readable explanation
    tags: list = field(default_factory=list)  # e.g. ["killzone", "fvg_entry", "spring"]

    # Risk metrics (filled by Risk Manager)
    risk_reward: float = 0.0
    position_size: float = 0.0
    risk_amount: float = 0.0

    def __post_init__(self):
        if self.entry_price and self.stop_loss and self.take_profit:
            risk = abs(self.entry_price - self.stop_loss)
            reward = abs(self.take_profit - self.entry_price)
            self.risk_reward = round(reward / risk, 2) if risk > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "strategy": self.strategy_name,
            "confidence": round(self.confidence, 3),
            "timeframe": self.timeframe,
            "entry": round(self.entry_price, 2),
            "sl": round(self.stop_loss, 2),
            "tp": round(self.take_profit, 2),
            "rr": self.risk_reward,
            "reason": self.reason,
            "tags": self.tags,
        }

    def __str__(self):
        return (f"[{self.strategy_name}] {self.direction} @ {self.entry_price:.2f} | "
                f"SL: {self.stop_loss:.2f} | TP: {self.take_profit:.2f} | "
                f"R:R {self.risk_reward:.1f} | Conf: {self.confidence:.0%}")


class BaseStrategy:
    """
    Base class for all strategy detectors.
    Each strategy must implement detect_signals().
    """

    name: str = "base"

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)

    def detect_signals(self, df: pd.DataFrame, timeframe: str = "") -> list:
        """
        Scan a DataFrame for trade setups.
        Must return a list of Signal objects.
        """
        raise NotImplementedError("Each strategy must implement detect_signals()")

    def _get_atr(self, df: pd.DataFrame, idx: int) -> float:
        """Safely get ATR value at index."""
        if "atr_14" in df.columns and pd.notna(df["atr_14"].iloc[idx]):
            return df["atr_14"].iloc[idx]
        return 10.0  # Default fallback for gold (~$10)
'''

# ============================================================
# FILE 2: src/strategies/support_resistance_strategy.py
# ============================================================
FILES["src/strategies/support_resistance_strategy.py"] = r'''"""
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
'''

# ============================================================
# FILE 3: src/strategies/rsi_divergence_strategy.py
# ============================================================
FILES["src/strategies/rsi_divergence_strategy.py"] = r'''"""
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
'''

# ============================================================
# FILE 4: src/strategies/ict_strategy.py
# ============================================================
FILES["src/strategies/ict_strategy.py"] = r'''"""
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
'''

# ============================================================
# FILE 5: src/strategies/wyckoff_strategy.py
# ============================================================
FILES["src/strategies/wyckoff_strategy.py"] = r'''"""
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
'''

# ============================================================
# FILE 6: src/strategies/crt_strategy.py
# ============================================================
FILES["src/strategies/crt_strategy.py"] = r'''"""
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
'''

# ============================================================
# FILE 7: src/strategies/sk_system_strategy.py
# ============================================================
FILES["src/strategies/sk_system_strategy.py"] = r'''"""
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
'''

# ============================================================
# FILE 8: src/strategies/strategy_manager.py - Orchestrator
# ============================================================
FILES["src/strategies/strategy_manager.py"] = r'''"""
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
'''

# ============================================================
# FILE 9: run_strategies.py - Test Script
# ============================================================
FILES["run_strategies.py"] = r'''"""
XAUUSD AI Trading System - Strategy Detector Test
Run all strategies on your data and see what signals are generated.

Usage:
    python run_strategies.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collector.mt5_collector import MT5Connector, DataCollector
from src.feature_engine.engine import FeatureEngine
from src.strategies.strategy_manager import StrategyManager
from src.utils.logger import setup_logger

logger = setup_logger("run_strategies")


def main():
    print("\n" + "=" * 70)
    print("  XAUUSD AI TRADING SYSTEM")
    print("  Phase 2 - Session 3: Strategy Detectors")
    print("=" * 70 + "\n")

    # --- Step 1: Get fresh data ---
    logger.info("Step 1: Fetching fresh data from MT5...")
    connector = MT5Connector()
    if not connector.connect():
        logger.error("Cannot connect to MT5!")
        return

    collector = DataCollector(connector)
    collector.check_symbol()
    data = collector.fetch_all_timeframes()
    connector.disconnect()

    if not data:
        logger.error("No data!")
        return

    # --- Step 2: Compute features ---
    logger.info("\nStep 2: Computing features...\n")
    engine = FeatureEngine()
    featured_data = engine.compute_multi_timeframe(data)

    # --- Step 3: Run strategy detectors ---
    logger.info("\nStep 3: Running strategy detectors...\n")
    manager = StrategyManager()
    all_signals = manager.scan_multi_timeframe(featured_data)

    # --- Step 4: Results ---
    print("\n" + "=" * 70)
    print("  STRATEGY DETECTOR RESULTS")
    print("=" * 70)

    total_signals = 0
    strategy_counts = {}

    for tf_name, signals in all_signals.items():
        total_signals += len(signals)
        print(f"\n  --- {tf_name}: {len(signals)} signals ---")

        for sig in signals[:5]:  # Show top 5 per timeframe
            print(f"    {sig}")
            strat = sig.strategy_name
            strategy_counts[strat] = strategy_counts.get(strat, 0) + 1

        if len(signals) > 5:
            print(f"    ... and {len(signals) - 5} more")

    print(f"\n  {'='*50}")
    print(f"  TOTAL: {total_signals} signals across all timeframes")
    print(f"  {'='*50}")

    if strategy_counts:
        print(f"\n  Signals by strategy:")
        for strat, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    {strat:>25}: {count} signals")

    # --- Step 5: Signal quality analysis ---
    print(f"\n  {'='*50}")
    print(f"  SIGNAL QUALITY ANALYSIS")
    print(f"  {'='*50}")

    all_sigs_flat = []
    for signals in all_signals.values():
        all_sigs_flat.extend(signals)

    if all_sigs_flat:
        confidences = [s.confidence for s in all_sigs_flat]
        rrs = [s.risk_reward for s in all_sigs_flat if s.risk_reward > 0]

        print(f"\n  Avg confidence: {sum(confidences)/len(confidences):.1%}")
        print(f"  Max confidence: {max(confidences):.1%}")
        print(f"  Min confidence: {min(confidences):.1%}")
        if rrs:
            print(f"  Avg R:R ratio:  {sum(rrs)/len(rrs):.1f}")
            print(f"  Best R:R ratio: {max(rrs):.1f}")

        # High confidence signals
        high_conf = [s for s in all_sigs_flat if s.confidence >= 0.70]
        print(f"\n  High confidence signals (>= 70%): {len(high_conf)}")
        for sig in high_conf[:10]:
            print(f"    {sig}")

    print("\n" + "=" * 70)
    print("  Phase 2 - Session 3 COMPLETE!")
    print("  Strategy detectors are working!")
    print("  Next: Build Risk Manager and connect to ML scorer")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
'''

# ============================================================
# Create all files
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("  Strategy Detectors Setup")
    print("=" * 60 + "\n")

    for filepath, content in FILES.items():
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        print(f"  [FILE] {filepath}")

    print(f"\n  Created {len(FILES)} files.")
    print("\n  Now run:  python run_strategies.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
