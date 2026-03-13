"""
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
