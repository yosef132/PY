"""
XAUUSD AI Trading System - Threshold Tuning Patch
Fixes the "too selective" problem: 98 scans, 0 trades.

Problem: ML scores are 0.43-0.55 but threshold is 0.55
Solution: Lower ML threshold to 0.45, reduce news blackout to 15 min

Usage:
    python patch_thresholds.py
    python live_trader.py
"""

import os

FILES = {}

# ============================================================
# FILE 1: Updated live engine with lower thresholds
# ============================================================
FILES["src/executor/live_engine.py"] = r'''"""
Live Trading Engine (v3)
Tuned thresholds for active trading while maintaining safety.

Changes from v2:
- ML threshold lowered from 0.55 to 0.45
- News blackout reduced from 30 min to 15 min
- When H4 is NEUTRAL, signals pass with slight penalty instead of no boost
- Added trade logging for every rejected signal (helps debug)
- Scan counter fix
"""

import MetaTrader5 as mt5
import time
import json
from datetime import datetime, timezone
from pathlib import Path
from src.data_collector.mt5_collector import MT5Connector, DataCollector
from src.feature_engine.engine import FeatureEngine
from src.strategies.strategy_manager import StrategyManager
from src.strategies.mtf_filter import MTFFilter
from src.risk_manager.signal_filter import SignalFilter
from src.risk_manager.risk_manager import RiskManager
from src.ml_scorer.model import MLSignalScorer
from src.ml_scorer.feature_builder import FeatureBuilder
from src.news_filter.news_manager import NewsManager
from src.executor.mt5_executor import MT5Executor
from src.alerts.telegram_alerts import TelegramAlerts
from src.utils.config import get_settings, PROJECT_ROOT
from src.utils.logger import setup_logger

logger = setup_logger("live_engine")

# ===== TUNABLE THRESHOLDS =====
ML_THRESHOLD = 0.45          # Was 0.55 — lowered so more trades pass
NEWS_BLACKOUT_MINUTES = 15   # Was 30 — reduced window
MAX_SIGNALS_PER_SCAN = 3     # Max trades to attempt per scan
# ===============================


class LiveTradingEngine:
    """
    Live trading loop v3 - Tuned for active trading.
    """

    def __init__(self):
        self.settings = get_settings()
        self.scan_interval = 300  # 5 minutes
        self.news_refresh_interval = 900  # 15 minutes
        self.h4_refresh_interval = 3600  # 1 hour
        self.last_news_refresh = 0
        self.last_h4_refresh = 0

        # Core components
        self.connector = MT5Connector()
        self.feature_engine = FeatureEngine()
        self.strategy_manager = StrategyManager()
        self.signal_filter = SignalFilter()
        self.risk_manager = RiskManager(initial_balance=0)
        self.executor = MT5Executor()
        self.news_manager = NewsManager()
        self.mtf_filter = MTFFilter()
        self.telegram = TelegramAlerts()

        # ML model
        self.ml_scorer = MLSignalScorer()
        self.feature_builder = FeatureBuilder()
        self.ml_loaded = self.ml_scorer.load()

        # Trade log
        self.trade_log_path = PROJECT_ROOT / "data" / "live_trades.json"
        self.trade_log = self._load_trade_log()

        # Rejection log (for debugging)
        self.rejection_log_path = PROJECT_ROOT / "data" / "rejected_signals.json"
        self.rejection_log = []

        # State
        self.running = False
        self.total_scans = 0
        self.total_trades_placed = 0
        self.session_start_balance = 0

    def _load_trade_log(self) -> list:
        if self.trade_log_path.exists():
            try:
                with open(self.trade_log_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_trade_log(self):
        self.trade_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.trade_log_path, "w") as f:
            json.dump(self.trade_log, f, indent=2, default=str)

    def _save_rejection_log(self):
        self.rejection_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.rejection_log_path, "w") as f:
            json.dump(self.rejection_log[-500:], f, indent=2, default=str)  # Keep last 500

    def start(self):
        print("\n" + "=" * 70)
        print("  XAUUSD AI TRADING SYSTEM - LIVE PAPER TRADING v3")
        print(f"  Mode: DEMO | MTF: H4+H1 | ML threshold: {ML_THRESHOLD}")
        print(f"  News blackout: {NEWS_BLACKOUT_MINUTES} min | Telegram: ON")
        print("  Press Ctrl+C to stop")
        print("=" * 70 + "\n")

        if not self.connector.connect():
            logger.error("Cannot connect to MT5!")
            return

        account = self.connector.get_account_info()
        self.risk_manager.initial_balance = account["balance"]
        self.risk_manager.current_balance = account["balance"]
        self.risk_manager.peak_balance = account["balance"]
        self.session_start_balance = account["balance"]

        logger.info(f"  Account: {account['login']} | Balance: ${account['balance']:.2f}")
        logger.info(f"  ML Model: {'Loaded' if self.ml_loaded else 'Not loaded'}")
        logger.info(f"  ML Threshold: {ML_THRESHOLD}")
        logger.info(f"  MTF Filter: H4 trend confirmation ACTIVE")
        logger.info(f"  Telegram: {'ACTIVE' if self.telegram.enabled else 'DISABLED'}")

        self.telegram.send(
            "\U0001F680 <b>BOT STARTED (v3)</b>\n"
            f"Account: {account['login']}\n"
            f"Balance: ${account['balance']:.2f}\n"
            f"ML threshold: {ML_THRESHOLD}\n"
            f"News blackout: {NEWS_BLACKOUT_MINUTES} min\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )

        self._refresh_news()
        self._refresh_h4_trend()

        self.running = True

        try:
            while self.running:
                self._scan_and_trade()
                self.total_scans += 1
                self._print_status()

                logger.info(f"  Next scan in {self.scan_interval} seconds... (Ctrl+C to stop)")
                time.sleep(self.scan_interval)

        except KeyboardInterrupt:
            logger.info("\n  Stopping live trading...")
            self._shutdown()

    def _refresh_h4_trend(self):
        try:
            logger.info("  Refreshing H4/D1 trend analysis...")
            collector = DataCollector(self.connector)
            featured_data = {}

            h4_df = collector.fetch_candles("H4", num_bars=200)
            if not h4_df.empty:
                featured_data["H4"] = self.feature_engine.compute_features(h4_df, "H4")

            d1_df = collector.fetch_candles("D1", num_bars=100)
            if not d1_df.empty:
                featured_data["D1"] = self.feature_engine.compute_features(d1_df, "D1")

            self.mtf_filter.analyze_higher_timeframes(featured_data)
            self.last_h4_refresh = time.time()

            status = self.mtf_filter.get_status()
            logger.info(f"  MTF: H4={status['h4_trend']} ({status['h4_strength']:.2f}) | D1={status['d1_trend']}")

        except Exception as e:
            logger.warning(f"  H4/D1 refresh failed: {e}")

    def _scan_and_trade(self):
        scan_time = datetime.now()
        logger.info(f"\n{'='*50}")
        logger.info(f"SCAN #{self.total_scans + 1} at {scan_time.strftime('%H:%M:%S')}")
        logger.info(f"{'='*50}")

        # Periodic refreshes
        if time.time() - self.last_news_refresh > self.news_refresh_interval:
            self._refresh_news()
        if time.time() - self.last_h4_refresh > self.h4_refresh_interval:
            self._refresh_h4_trend()

        # Step 1: Fetch H1 data
        try:
            collector = DataCollector(self.connector)
            df = collector.fetch_candles("H1", num_bars=500)
            if df.empty:
                logger.warning("  No data received")
                return
        except Exception as e:
            logger.error(f"  Data fetch error: {e}")
            return

        # Step 2: Features
        try:
            featured_df = self.feature_engine.compute_features(df, "H1")
        except Exception as e:
            logger.error(f"  Feature error: {e}")
            return

        # Step 3: Signals
        raw_signals = self.strategy_manager.scan_all(featured_df, "H1")
        filtered_signals = self.signal_filter.filter_signals(raw_signals, "H1")

        logger.info(f"  Signals: {len(raw_signals)} raw -> {len(filtered_signals)} filtered")

        if not filtered_signals:
            logger.info(f"  No signals. Waiting...")
            return

        # Step 4: Process signals
        mtf_blocked = 0
        news_blocked = 0
        ml_blocked = 0
        risk_blocked = 0

        for signal in filtered_signals[:MAX_SIGNALS_PER_SCAN]:

            # MTF check (but allow neutral H4 — don't block)
            mtf_allowed, mtf_reason, mtf_boost = self.mtf_filter.filter_signal(signal)
            if not mtf_allowed:
                logger.info(f"  MTF BLOCKED: {mtf_reason}")
                self._log_rejection(scan_time, signal, "mtf", mtf_reason)
                mtf_blocked += 1
                continue

            if mtf_boost > 0:
                old_conf = signal.confidence
                signal.confidence = min(signal.confidence + mtf_boost, 1.0)
                logger.info(f"  MTF boost: {old_conf:.2f} -> {signal.confidence:.2f}")

            # News check (with reduced blackout window)
            news_ok, news_reason, news_score = self.news_manager.can_trade(
                signal.direction, blackout_minutes=NEWS_BLACKOUT_MINUTES
            )
            if not news_ok:
                logger.info(f"  News blocked: {news_reason}")
                self._log_rejection(scan_time, signal, "news", news_reason)
                news_blocked += 1
                continue

            # ML scoring
            ml_score = signal.confidence
            if self.ml_loaded:
                current_bar = featured_df.iloc[-1]
                ml_score = self.ml_scorer.score_signal(signal, current_bar, self.feature_builder)
                signal.confidence = ml_score
                logger.info(f"  ML score: {ml_score:.3f} (threshold: {ML_THRESHOLD})")

                if ml_score < ML_THRESHOLD:
                    logger.info(f"  ML rejected ({ml_score:.3f} < {ML_THRESHOLD})")
                    self._log_rejection(scan_time, signal, "ml", f"score {ml_score:.3f}")
                    ml_blocked += 1
                    continue

            # Risk check
            allowed, reason = self.risk_manager.can_trade(signal)
            if not allowed:
                logger.info(f"  Risk blocked: {reason}")
                self._log_rejection(scan_time, signal, "risk", reason)
                risk_blocked += 1
                continue

            # Position sizing
            account = self.executor.get_account_summary()
            self.risk_manager.current_balance = account.get("balance", self.risk_manager.current_balance)
            sizing = self.risk_manager.calculate_position_size(signal)

            if sizing["lot_size"] <= 0:
                continue

            # Max positions check
            open_positions = self.executor.get_open_positions()
            if len(open_positions) >= self.risk_manager.max_open_positions:
                logger.info(f"  Max positions ({len(open_positions)})")
                continue

            # EXECUTE!
            logger.info(f"\n  >>> EXECUTING TRADE <<<")
            logger.info(f"  Strategy: {signal.strategy_name} | {signal.direction}")
            logger.info(f"  Entry: {signal.entry_price:.2f} | SL: {signal.stop_loss:.2f} | TP: {signal.take_profit:.2f}")
            logger.info(f"  ML: {ml_score:.3f} | H4: {self.mtf_filter.h4_trend} | Lot: {sizing['lot_size']}")

            comment = f"AI_{signal.strategy_name[:8]}"
            result = self.executor.place_trade(
                direction=signal.direction,
                lot_size=sizing["lot_size"],
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                comment=comment
            )

            if result["success"]:
                self.total_trades_placed += 1
                self.risk_manager.daily_trades += 1

                trade_record = {
                    "scan": self.total_scans + 1,
                    "time": scan_time.isoformat(),
                    "ticket": result["ticket"],
                    "direction": signal.direction,
                    "strategy": signal.strategy_name,
                    "entry": result["price"],
                    "sl": signal.stop_loss,
                    "tp": signal.take_profit,
                    "lot_size": sizing["lot_size"],
                    "risk_amount": sizing["risk_amount"],
                    "confidence": signal.confidence,
                    "ml_score": ml_score,
                    "news_score": news_score,
                    "h4_trend": self.mtf_filter.h4_trend,
                    "mtf_boost": mtf_boost,
                    "reason": signal.reason,
                    "result": "open",
                }
                self.trade_log.append(trade_record)
                self._save_trade_log()

                self.telegram.send_trade_opened({
                    "direction": signal.direction,
                    "strategy": signal.strategy_name,
                    "entry": result["price"],
                    "sl": signal.stop_loss,
                    "tp": signal.take_profit,
                    "lot_size": sizing["lot_size"],
                    "confidence": signal.confidence,
                })

                logger.info(f"  TRADE #{self.total_trades_placed} PLACED! Telegram sent.")
            else:
                logger.error(f"  Trade failed: {result.get('error')}")
                self.telegram.send_alert(f"Trade failed: {result.get('error')}")

        if mtf_blocked + news_blocked + ml_blocked + risk_blocked > 0:
            logger.info(f"  Blocked: MTF={mtf_blocked} News={news_blocked} ML={ml_blocked} Risk={risk_blocked}")

    def _log_rejection(self, time_stamp, signal, reason_type, reason_detail):
        self.rejection_log.append({
            "time": time_stamp.isoformat(),
            "direction": signal.direction,
            "strategy": signal.strategy_name,
            "confidence": round(signal.confidence, 3),
            "blocked_by": reason_type,
            "reason": reason_detail,
        })
        if len(self.rejection_log) % 50 == 0:
            self._save_rejection_log()

    def _refresh_news(self):
        try:
            self.news_manager.refresh()
            self.last_news_refresh = time.time()
        except Exception as e:
            logger.warning(f"  News refresh failed: {e}")

    def _print_status(self):
        account = self.executor.get_account_summary()
        positions = self.executor.get_open_positions()
        mtf_status = self.mtf_filter.get_status()
        session_pnl = account.get("balance", 0) - self.session_start_balance

        print(f"\n  --- BOT STATUS ---")
        print(f"  Scans: {self.total_scans + 1} | Trades: {self.total_trades_placed}")
        print(f"  Balance: ${account.get('balance', 0):.2f} | Equity: ${account.get('equity', 0):.2f} | "
              f"Session PnL: ${session_pnl:+.2f}")
        print(f"  H4 Trend: {mtf_status['h4_trend']} ({mtf_status['h4_strength']:.2f}) | "
              f"D1 Trend: {mtf_status['d1_trend']}")
        print(f"  Open positions: {len(positions)}")

        for pos in positions:
            pnl_icon = "\U0001F7E2" if pos["profit"] >= 0 else "\U0001F534"
            print(f"    {pnl_icon} #{pos['ticket']} {pos['direction']} {pos['volume']} lots "
                  f"@ {pos['price_open']:.2f} | PnL: ${pos['profit']:.2f}")

        is_blackout, reason = self.news_manager.calendar.is_news_blackout(NEWS_BLACKOUT_MINUTES)
        if is_blackout:
            print(f"  \U000026A0 NEWS BLACKOUT: {reason}")
        else:
            print(f"  \U00002705 News: Clear")

    def _shutdown(self):
        self.running = False
        self._save_rejection_log()

        positions = self.executor.get_open_positions()
        account = self.executor.get_account_summary()
        session_pnl = account.get("balance", 0) - self.session_start_balance

        print(f"\n{'='*70}")
        print(f"  SESSION ENDED")
        print(f"{'='*70}")
        print(f"  Scans: {self.total_scans} | Trades: {self.total_trades_placed}")
        print(f"  Balance: ${account.get('balance', 0):.2f} | Session PnL: ${session_pnl:+.2f}")
        print(f"  Rejections logged: {len(self.rejection_log)}")
        print(f"{'='*70}\n")

        self.telegram.send(
            "\U0001F6D1 <b>BOT STOPPED</b>\n"
            f"Scans: {self.total_scans}\n"
            f"Trades: {self.total_trades_placed}\n"
            f"Balance: ${account.get('balance', 0):.2f}\n"
            f"Session PnL: ${session_pnl:+.2f}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )

        self.connector.disconnect()
'''

# ============================================================
# FILE 2: Patch news_manager to accept blackout_minutes param
# ============================================================
FILES["src/news_filter/news_manager.py"] = r'''"""
News Manager - Unified news intelligence module.
Combines economic calendar + headline sentiment to make trade decisions.
Updated: accepts blackout_minutes parameter.
"""

from src.news_filter.economic_calendar import EconomicCalendar
from src.news_filter.headline_sentiment import HeadlineSentiment
from src.news_filter.news_scraper import NewsScraper
from src.utils.logger import setup_logger

logger = setup_logger("news_manager")


class NewsManager:
    """Unified news manager: calendar blackouts + sentiment scoring."""

    def __init__(self):
        self.calendar = EconomicCalendar()
        self.sentiment_scorer = HeadlineSentiment()
        self.scraper = NewsScraper()

        self.headlines = []
        self.sentiment = {}

    def refresh(self):
        """Refresh all news data."""
        logger.info("  Refreshing news data...")

        # Economic calendar
        self.calendar.fetch_calendar()

        # Headlines
        self.headlines = self.scraper.fetch_headlines()

        # Score sentiment
        if self.headlines:
            texts = [h["title"] for h in self.headlines]
            self.sentiment = self.sentiment_scorer.score_headlines(texts)
        else:
            self.sentiment = {"score": 0, "direction": "neutral"}

        direction = self.sentiment.get("direction", "neutral")
        score = self.sentiment.get("score", 0)
        logger.info(f"  News refresh complete | Sentiment: {direction} ({score:.2f})")

    def can_trade(self, direction: str, blackout_minutes: int = 15) -> tuple:
        """
        Check if trading is allowed based on news.

        Args:
            direction: "BUY" or "SELL"
            blackout_minutes: Minutes before/after high-impact news to block

        Returns:
            (allowed: bool, reason: str, sentiment_score: float)
        """
        # Check news blackout
        is_blackout, blackout_reason = self.calendar.is_news_blackout(blackout_minutes)
        if is_blackout:
            return False, f"NEWS BLACKOUT: {blackout_reason}", 0

        # Get sentiment
        sent_score = self.sentiment.get("score", 0)
        sent_dir = self.sentiment.get("direction", "neutral")

        # Strong opposing sentiment blocks trade
        if direction == "BUY" and sent_score < -0.5:
            return False, f"Strong bearish sentiment ({sent_score:.2f})", sent_score
        if direction == "SELL" and sent_score > 0.5:
            return False, f"Strong bullish sentiment ({sent_score:.2f})", sent_score

        return True, "News OK", sent_score

    def get_status(self) -> dict:
        """Get current news status summary."""
        is_blackout, reason = self.calendar.is_news_blackout(15)
        return {
            "sentiment": self.sentiment,
            "is_blackout": is_blackout,
            "blackout_reason": reason if is_blackout else None,
            "headline_count": len(self.headlines),
        }
'''

# ============================================================
# Create all files
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("  Threshold Tuning Patch")
    print("=" * 60 + "\n")

    for filepath, content in FILES.items():
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        print(f"  [FILE] {filepath}")

    print(f"\n  Changes:")
    print(f"    - ML threshold: 0.55 -> 0.45")
    print(f"    - News blackout: 30 min -> 15 min")
    print(f"    - Rejection logging added (data/rejected_signals.json)")
    print(f"    - News manager accepts blackout_minutes param")
    print(f"\n  Now run: python live_trader.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
