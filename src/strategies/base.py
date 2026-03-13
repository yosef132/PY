"""
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
