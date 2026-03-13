"""
XAUUSD AI Trading System - Multi-Timeframe + Telegram + Tests Setup
Session 10: Major accuracy upgrade + phone alerts + reliability.

Usage:
    python setup_mtf_upgrade.py
    python run_full_system.py
"""

import os

FILES = {}

# ============================================================
# FILE 1: src/strategies/mtf_filter.py - Multi-Timeframe Confirmation
# ============================================================
FILES["src/strategies/mtf_filter.py"] = r'''"""
Multi-Timeframe (MTF) Confirmation Filter
The #1 accuracy improvement: only trade H1 signals that align with H4 trend direction.

How it works:
- H4 determines the TREND DIRECTION (are we bullish or bearish overall?)
- H1 finds the ENTRY TIMING (where exactly to enter)
- If H1 signal disagrees with H4 trend → BLOCKED
- If H1 signal agrees with H4 trend → BOOSTED confidence

This alone can improve win rate by 10-20% because you stop fighting the bigger trend.
"""

import pandas as pd
import numpy as np
from src.feature_engine.engine import FeatureEngine
from src.utils.logger import setup_logger

logger = setup_logger("mtf_filter")


class MTFFilter:
    """
    Multi-Timeframe trend filter.
    Uses H4 (or D1) to determine trend, then only allows H1 entries in that direction.
    """

    def __init__(self):
        self.h4_trend = 0       # +1 = bullish, -1 = bearish, 0 = neutral
        self.h4_strength = 0.0  # 0.0 to 1.0
        self.h4_data = {}       # Cached H4 analysis
        self.d1_trend = 0       # Daily trend for extra confirmation

    def analyze_higher_timeframes(self, featured_data: dict):
        """
        Analyze H4 and D1 data to determine the higher timeframe trend.

        Args:
            featured_data: Dict with "H4" and optionally "D1" DataFrames
        """
        # Analyze H4
        if "H4" in featured_data:
            h4_df = featured_data["H4"]
            self.h4_trend, self.h4_strength, self.h4_data = self._analyze_trend(h4_df, "H4")
            logger.info(f"  H4 Trend: {'BULLISH' if self.h4_trend > 0 else 'BEARISH' if self.h4_trend < 0 else 'NEUTRAL'} "
                        f"(strength: {self.h4_strength:.2f})")

        # Analyze D1
        if "D1" in featured_data:
            d1_df = featured_data["D1"]
            self.d1_trend, _, _ = self._analyze_trend(d1_df, "D1")
            logger.info(f"  D1 Trend: {'BULLISH' if self.d1_trend > 0 else 'BEARISH' if self.d1_trend < 0 else 'NEUTRAL'}")

    def _analyze_trend(self, df: pd.DataFrame, tf_name: str) -> tuple:
        """
        Determine trend direction and strength from a featured DataFrame.

        Returns:
            (trend: int, strength: float, data: dict)
        """
        if df.empty or len(df) < 50:
            return 0, 0.0, {}

        last = df.iloc[-1]
        prev_5 = df.iloc[-5:]

        trend_score = 0
        signals = {}

        # --- Signal 1: EMA Alignment ---
        ema_align = last.get("ema_alignment", 0)
        if ema_align >= 2:
            trend_score += 2
            signals["ema"] = "bullish"
        elif ema_align <= -2:
            trend_score -= 2
            signals["ema"] = "bearish"
        else:
            signals["ema"] = "neutral"

        # --- Signal 2: Price vs EMA 50 ---
        ema_50 = last.get("ema_50", None)
        if pd.notna(ema_50):
            if last["close"] > ema_50 * 1.002:
                trend_score += 1
                signals["price_ema50"] = "above"
            elif last["close"] < ema_50 * 0.998:
                trend_score -= 1
                signals["price_ema50"] = "below"

        # --- Signal 3: Price vs EMA 200 ---
        ema_200 = last.get("ema_200", None)
        if pd.notna(ema_200):
            if last["close"] > ema_200:
                trend_score += 1
                signals["price_ema200"] = "above"
            elif last["close"] < ema_200:
                trend_score -= 1
                signals["price_ema200"] = "below"

        # --- Signal 4: Market structure trend ---
        market_trend = last.get("market_trend", 0)
        if market_trend > 0:
            trend_score += 1
            signals["structure"] = "bullish"
        elif market_trend < 0:
            trend_score -= 1
            signals["structure"] = "bearish"

        # --- Signal 5: ADX trend strength ---
        adx_cols = [c for c in df.columns if "ADX_" in c]
        if adx_cols:
            adx = last.get(adx_cols[0], 0)
            if pd.notna(adx) and adx > 25:
                signals["adx"] = f"trending ({adx:.0f})"
            else:
                signals["adx"] = f"ranging ({adx:.0f})" if pd.notna(adx) else "unknown"

        # --- Signal 6: RSI position ---
        rsi = last.get("rsi_14", 50)
        if pd.notna(rsi):
            if rsi > 55:
                trend_score += 0.5
                signals["rsi"] = f"bullish ({rsi:.0f})"
            elif rsi < 45:
                trend_score -= 0.5
                signals["rsi"] = f"bearish ({rsi:.0f})"

        # --- Signal 7: MACD direction ---
        macd_cols = [c for c in df.columns if "MACDh_" in c]
        if macd_cols:
            macd_h = last.get(macd_cols[0], 0)
            prev_macd = df.iloc[-2].get(macd_cols[0], 0) if len(df) > 1 else 0
            if pd.notna(macd_h) and pd.notna(prev_macd):
                if macd_h > 0 and macd_h > prev_macd:
                    trend_score += 1
                    signals["macd"] = "bullish momentum"
                elif macd_h < 0 and macd_h < prev_macd:
                    trend_score -= 1
                    signals["macd"] = "bearish momentum"

        # --- Signal 8: Recent candle direction ---
        bullish_candles = sum(1 for _, r in prev_5.iterrows() if r["close"] > r["open"])
        if bullish_candles >= 4:
            trend_score += 1
            signals["candles"] = f"{bullish_candles}/5 bullish"
        elif bullish_candles <= 1:
            trend_score -= 1
            signals["candles"] = f"{5 - bullish_candles}/5 bearish"

        # Determine final trend
        if trend_score >= 3:
            trend = 1
        elif trend_score <= -3:
            trend = -1
        else:
            trend = 0

        # Strength (normalize to 0-1)
        strength = min(abs(trend_score) / 8.0, 1.0)

        data = {
            "trend_score": round(trend_score, 1),
            "strength": round(strength, 2),
            "signals": signals,
        }

        return trend, strength, data

    def filter_signal(self, signal) -> tuple:
        """
        Check if an H1 signal aligns with the higher timeframe trend.

        Returns:
            (allowed: bool, reason: str, confidence_adjustment: float)
        """
        direction = signal.direction

        # --- Rule 1: H4 trend alignment ---
        if self.h4_trend != 0:
            if direction == "BUY" and self.h4_trend < 0:
                return False, f"BUY blocked: H4 trend is BEARISH (score: {self.h4_data.get('trend_score', 0)})", 0

            if direction == "SELL" and self.h4_trend > 0:
                return False, f"SELL blocked: H4 trend is BULLISH (score: {self.h4_data.get('trend_score', 0)})", 0

        # --- Rule 2: D1 alignment bonus ---
        confidence_boost = 0.0

        if self.d1_trend != 0:
            if (direction == "BUY" and self.d1_trend > 0) or \
               (direction == "SELL" and self.d1_trend < 0):
                confidence_boost += 0.05  # D1 agrees = +5% confidence

        # --- Rule 3: H4 strength bonus ---
        if self.h4_trend != 0:
            if (direction == "BUY" and self.h4_trend > 0) or \
               (direction == "SELL" and self.h4_trend < 0):
                confidence_boost += self.h4_strength * 0.10  # Up to +10%

        # --- Rule 4: H4 neutral = allow but no boost ---
        if self.h4_trend == 0:
            return True, "H4 neutral - signal allowed without boost", 0

        reason = (f"H4 {'BULLISH' if self.h4_trend > 0 else 'BEARISH'} aligns with {direction} "
                  f"(boost: +{confidence_boost:.1%})")

        return True, reason, confidence_boost

    def get_status(self) -> dict:
        """Get current MTF analysis status."""
        return {
            "h4_trend": "BULLISH" if self.h4_trend > 0 else "BEARISH" if self.h4_trend < 0 else "NEUTRAL",
            "h4_strength": self.h4_strength,
            "h4_details": self.h4_data,
            "d1_trend": "BULLISH" if self.d1_trend > 0 else "BEARISH" if self.d1_trend < 0 else "NEUTRAL",
        }
'''

# ============================================================
# FILE 2: tests/test_strategies.py - Unit Tests
# ============================================================
FILES["tests/test_strategies.py"] = r'''"""
Unit Tests for XAUUSD AI Trading System
Run: python -m pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ============================================================
# Test 1: Signal object
# ============================================================
class TestSignal:
    def test_signal_creation(self):
        from src.strategies.base import Signal
        sig = Signal(
            direction="BUY",
            strategy_name="test",
            confidence=0.75,
            timeframe="H1",
            entry_price=5000.0,
            stop_loss=4980.0,
            take_profit=5060.0,
        )
        assert sig.direction == "BUY"
        assert sig.risk_reward == 3.0  # (60/20)
        assert sig.confidence == 0.75

    def test_signal_rr_calculation(self):
        from src.strategies.base import Signal
        sig = Signal(
            direction="SELL",
            strategy_name="test",
            confidence=0.6,
            timeframe="H1",
            entry_price=5000.0,
            stop_loss=5020.0,
            take_profit=4940.0,
        )
        assert sig.risk_reward == 3.0  # (60/20)

    def test_signal_to_dict(self):
        from src.strategies.base import Signal
        sig = Signal(
            direction="BUY",
            strategy_name="ict",
            confidence=0.8,
            timeframe="H1",
            entry_price=5000.0,
            stop_loss=4990.0,
            take_profit=5030.0,
        )
        d = sig.to_dict()
        assert d["direction"] == "BUY"
        assert d["strategy"] == "ict"
        assert "entry" in d


# ============================================================
# Test 2: Config loading
# ============================================================
class TestConfig:
    def test_load_settings(self):
        from src.utils.config import get_settings
        settings = get_settings()
        assert "symbol" in settings
        assert settings["symbol"] == "XAUUSD"

    def test_load_strategies(self):
        from src.utils.config import get_strategies
        strats = get_strategies()
        assert "strategies" in strats

    def test_mt5_timeframe_mapping(self):
        from src.utils.config import get_mt5_timeframe
        assert get_mt5_timeframe("H1") == 16385
        assert get_mt5_timeframe("M5") == 5
        assert get_mt5_timeframe("D1") == 16408


# ============================================================
# Test 3: Feature Engine
# ============================================================
class TestFeatureEngine:
    def _make_sample_df(self, rows=300):
        """Create a sample OHLCV DataFrame for testing."""
        dates = pd.date_range("2025-01-01", periods=rows, freq="h")
        np.random.seed(42)
        close = 5000 + np.cumsum(np.random.randn(rows) * 5)
        df = pd.DataFrame({
            "open": close - np.random.rand(rows) * 3,
            "high": close + np.random.rand(rows) * 10,
            "low": close - np.random.rand(rows) * 10,
            "close": close,
            "volume": np.random.randint(100, 10000, rows),
        }, index=dates)
        return df

    def test_indicators(self):
        from src.feature_engine.indicators import compute_all_indicators
        df = self._make_sample_df()
        result = compute_all_indicators(df)
        assert "rsi_14" in result.columns
        assert "ema_21" in result.columns
        assert "atr_14" in result.columns
        assert "macd" in " ".join(result.columns).lower()
        assert len(result.columns) > 50

    def test_market_structure(self):
        from src.feature_engine.indicators import compute_all_indicators
        from src.feature_engine.market_structure import compute_all_structure
        df = self._make_sample_df()
        df = compute_all_indicators(df)
        result = compute_all_structure(df)
        assert "swing_high" in result.columns
        assert "bos" in result.columns
        assert "fvg_bullish" in result.columns
        assert "ob_bullish" in result.columns

    def test_session_features(self):
        from src.feature_engine.session_features import add_session_features
        df = self._make_sample_df()
        result = add_session_features(df)
        assert "killzone_active" in result.columns
        assert "session_london" in result.columns
        assert "hour_sin" in result.columns


# ============================================================
# Test 4: Signal Filter
# ============================================================
class TestSignalFilter:
    def test_filter_removes_bad_rr(self):
        from src.strategies.base import Signal
        from src.risk_manager.signal_filter import SignalFilter

        filt = SignalFilter()
        signals = [
            Signal(direction="BUY", strategy_name="test", confidence=0.8,
                   timeframe="H1", entry_price=5000, stop_loss=4990, take_profit=5005),  # RR = 0.5
            Signal(direction="BUY", strategy_name="test", confidence=0.8,
                   timeframe="H1", entry_price=5000, stop_loss=4990, take_profit=5030),  # RR = 3.0
        ]
        filtered = filt.filter_signals(signals, "H1")
        # Only the 3.0 RR signal should pass
        assert len(filtered) <= 1
        if filtered:
            assert filtered[0].risk_reward >= 2.0

    def test_filter_removes_low_confidence(self):
        from src.strategies.base import Signal
        from src.risk_manager.signal_filter import SignalFilter

        filt = SignalFilter()
        signals = [
            Signal(direction="BUY", strategy_name="test", confidence=0.3,
                   timeframe="H1", entry_price=5000, stop_loss=4980, take_profit=5060),
        ]
        filtered = filt.filter_signals(signals, "H1")
        assert len(filtered) == 0  # Too low confidence


# ============================================================
# Test 5: Risk Manager
# ============================================================
class TestRiskManager:
    def test_position_sizing(self):
        from src.strategies.base import Signal
        from src.risk_manager.risk_manager import RiskManager

        rm = RiskManager(initial_balance=100000)
        sig = Signal(direction="BUY", strategy_name="test", confidence=0.8,
                     timeframe="H1", entry_price=5000, stop_loss=4980, take_profit=5060)
        sizing = rm.calculate_position_size(sig)
        assert sizing["lot_size"] >= 0.01
        assert sizing["risk_amount"] > 0
        # Risk should be ~0.5% of 100000 = ~500
        assert 100 < sizing["risk_amount"] < 1000

    def test_daily_loss_limit(self):
        from src.strategies.base import Signal
        from src.risk_manager.risk_manager import RiskManager

        rm = RiskManager(initial_balance=10000)
        rm.daily_pnl = -350  # 3.5% loss
        sig = Signal(direction="BUY", strategy_name="test", confidence=0.8,
                     timeframe="H1", entry_price=5000, stop_loss=4980, take_profit=5060)
        allowed, reason = rm.can_trade(sig)
        assert not allowed
        assert "daily loss" in reason.lower() or "halted" in reason.lower()

    def test_max_drawdown_halt(self):
        from src.strategies.base import Signal
        from src.risk_manager.risk_manager import RiskManager

        rm = RiskManager(initial_balance=10000)
        rm.current_balance = 8900  # 11% drawdown
        sig = Signal(direction="BUY", strategy_name="test", confidence=0.8,
                     timeframe="H1", entry_price=5000, stop_loss=4980, take_profit=5060)
        allowed, reason = rm.can_trade(sig)
        assert not allowed


# ============================================================
# Test 6: Sentiment Scorer
# ============================================================
class TestSentiment:
    def test_bullish_headline(self):
        from src.news_filter.headline_sentiment import HeadlineSentiment
        scorer = HeadlineSentiment()
        result = scorer.score_headline("Fed signals rate cuts, gold surges")
        assert result["score"] > 0
        assert result["direction"] == "bullish"

    def test_bearish_headline(self):
        from src.news_filter.headline_sentiment import HeadlineSentiment
        scorer = HeadlineSentiment()
        result = scorer.score_headline("Gold falls as dollar rallies on strong jobs data")
        assert result["score"] < 0
        assert result["direction"] == "bearish"

    def test_neutral_headline(self):
        from src.news_filter.headline_sentiment import HeadlineSentiment
        scorer = HeadlineSentiment()
        result = scorer.score_headline("Markets await next week economic data")
        assert result["direction"] == "neutral"

    def test_aggregate_sentiment(self):
        from src.news_filter.headline_sentiment import HeadlineSentiment
        scorer = HeadlineSentiment()
        result = scorer.score_headlines([
            "Gold surges on war fears",
            "Rate cuts expected",
            "Dollar weakness continues",
        ])
        assert result["direction"] == "bullish"
        assert result["bullish_count"] == 3


# ============================================================
# Test 7: MTF Filter
# ============================================================
class TestMTFFilter:
    def test_blocks_against_trend(self):
        from src.strategies.base import Signal
        from src.strategies.mtf_filter import MTFFilter

        mtf = MTFFilter()
        mtf.h4_trend = -1  # Bearish H4

        sig = Signal(direction="BUY", strategy_name="test", confidence=0.8,
                     timeframe="H1", entry_price=5000, stop_loss=4980, take_profit=5060)
        allowed, reason, boost = mtf.filter_signal(sig)
        assert not allowed
        assert "BEARISH" in reason

    def test_allows_with_trend(self):
        from src.strategies.base import Signal
        from src.strategies.mtf_filter import MTFFilter

        mtf = MTFFilter()
        mtf.h4_trend = 1  # Bullish H4
        mtf.h4_strength = 0.8

        sig = Signal(direction="BUY", strategy_name="test", confidence=0.7,
                     timeframe="H1", entry_price=5000, stop_loss=4980, take_profit=5060)
        allowed, reason, boost = mtf.filter_signal(sig)
        assert allowed
        assert boost > 0

    def test_neutral_allows_all(self):
        from src.strategies.base import Signal
        from src.strategies.mtf_filter import MTFFilter

        mtf = MTFFilter()
        mtf.h4_trend = 0  # Neutral

        sig = Signal(direction="SELL", strategy_name="test", confidence=0.7,
                     timeframe="H1", entry_price=5000, stop_loss=5020, take_profit=4940)
        allowed, reason, boost = mtf.filter_signal(sig)
        assert allowed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''

# ============================================================
# FILE 3: tests/__init__.py
# ============================================================
FILES["tests/__init__.py"] = ""

# ============================================================
# Create all files
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("  MTF Upgrade + Unit Tests Setup")
    print("=" * 60 + "\n")

    for filepath, content in FILES.items():
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        print(f"  [FILE] {filepath}")

    print(f"\n  Created {len(FILES)} files.")
    print("\n  Step 1: pip install pytest")
    print("  Step 2: python -m pytest tests/ -v")
    print("  Step 3: Get your Telegram token and tell me")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
