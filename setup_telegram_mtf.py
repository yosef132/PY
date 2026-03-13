"""
XAUUSD AI Trading System - Telegram + MTF Integration
Patches the live engine to use multi-timeframe filtering and Telegram alerts.

Usage:
    python setup_telegram_mtf.py
    python patch_config.py
    python test_telegram.py
    python live_trader.py
"""

import os

FILES = {}

# ============================================================
# FILE 1: Patch config with Telegram credentials
# ============================================================
FILES["patch_config.py"] = r'''"""
Patches settings.yaml to add Telegram config.
Run once then delete.
"""
import yaml
from pathlib import Path

config_path = Path("config/settings.yaml")
settings = yaml.safe_load(config_path.read_text())

# Add alerts section
settings["alerts"] = {
    "telegram_enabled": True,
    "telegram_token": "8760045272:AAH2VKn6vNRFlr0pb_KVGOIl8k_v_wz0FGg",
    "telegram_chat_id": "6553286553286774",
}

config_path.write_text(yaml.dump(settings, default_flow_style=False, sort_keys=False))
print("  settings.yaml updated with Telegram config!")
'''

# ============================================================
# FILE 2: Updated live engine with MTF + Telegram
# ============================================================
FILES["src/executor/live_engine.py"] = r'''"""
Live Trading Engine (v2)
Now with Multi-Timeframe confirmation and Telegram alerts.

Changes from v1:
- Fetches H4 data alongside H1 for trend confirmation
- Blocks H1 signals that fight the H4 trend
- Sends Telegram alerts for trades, summaries, and errors
- Confidence boosted when H4 agrees with signal
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


class LiveTradingEngine:
    """
    Live trading loop v2.
    Now with H4 trend confirmation and Telegram notifications.
    """

    def __init__(self):
        self.settings = get_settings()
        self.scan_interval = 300  # 5 minutes
        self.news_refresh_interval = 900  # 15 minutes
        self.h4_refresh_interval = 3600  # Refresh H4 trend every 1 hour
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

        # NEW: Multi-timeframe filter
        self.mtf_filter = MTFFilter()

        # NEW: Telegram alerts
        self.telegram = TelegramAlerts()

        # ML model
        self.ml_scorer = MLSignalScorer()
        self.feature_builder = FeatureBuilder()
        self.ml_loaded = self.ml_scorer.load()

        # Trade log
        self.trade_log_path = PROJECT_ROOT / "data" / "live_trades.json"
        self.trade_log = self._load_trade_log()

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

    def start(self):
        print("\n" + "=" * 70)
        print("  XAUUSD AI TRADING SYSTEM - LIVE PAPER TRADING v2")
        print("  Mode: DEMO | MTF: H4+H1 | Telegram: ON")
        print("  Press Ctrl+C to stop")
        print("=" * 70 + "\n")

        # Connect to MT5
        if not self.connector.connect():
            logger.error("Cannot connect to MT5!")
            return

        # Update risk manager with actual account balance
        account = self.connector.get_account_info()
        self.risk_manager.initial_balance = account["balance"]
        self.risk_manager.current_balance = account["balance"]
        self.risk_manager.peak_balance = account["balance"]
        self.session_start_balance = account["balance"]

        logger.info(f"  Account: {account['login']} | Balance: ${account['balance']:.2f}")
        logger.info(f"  ML Model: {'Loaded' if self.ml_loaded else 'Not loaded'}")
        logger.info(f"  MTF Filter: H4 trend confirmation ACTIVE")
        logger.info(f"  Telegram: {'ACTIVE' if self.telegram.enabled else 'DISABLED'}")
        logger.info(f"  Scan interval: {self.scan_interval} seconds")

        # Send startup notification
        self.telegram.send(
            "\U0001F680 <b>BOT STARTED</b>\n"
            f"Account: {account['login']}\n"
            f"Balance: ${account['balance']:.2f}\n"
            f"Mode: DEMO | MTF: H4+H1\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )

        # Initial refreshes
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
        """Fetch H4 and D1 data, compute features, analyze trend."""
        try:
            logger.info("  Refreshing H4/D1 trend analysis...")
            collector = DataCollector(self.connector)

            featured_data = {}

            # Fetch H4
            h4_df = collector.fetch_candles("H4", num_bars=200)
            if not h4_df.empty:
                featured_data["H4"] = self.feature_engine.compute_features(h4_df, "H4")

            # Fetch D1
            d1_df = collector.fetch_candles("D1", num_bars=100)
            if not d1_df.empty:
                featured_data["D1"] = self.feature_engine.compute_features(d1_df, "D1")

            self.mtf_filter.analyze_higher_timeframes(featured_data)
            self.last_h4_refresh = time.time()

            # Log MTF status
            status = self.mtf_filter.get_status()
            logger.info(f"  MTF Status: H4={status['h4_trend']} (strength: {status['h4_strength']:.2f}) | D1={status['d1_trend']}")

        except Exception as e:
            logger.warning(f"  H4/D1 refresh failed: {e}")

    def _scan_and_trade(self):
        """Single scan iteration with MTF + Telegram."""
        scan_time = datetime.now()
        logger.info(f"\n{'='*50}")
        logger.info(f"SCAN #{self.total_scans + 1} at {scan_time.strftime('%H:%M:%S')}")
        logger.info(f"{'='*50}")

        # Refresh news periodically
        if time.time() - self.last_news_refresh > self.news_refresh_interval:
            self._refresh_news()

        # Refresh H4 trend periodically (every hour)
        if time.time() - self.last_h4_refresh > self.h4_refresh_interval:
            self._refresh_h4_trend()

        # Step 1: Fetch latest H1 data
        try:
            collector = DataCollector(self.connector)
            df = collector.fetch_candles("H1", num_bars=500)
            if df.empty:
                logger.warning("  No data received, skipping scan")
                return
        except Exception as e:
            logger.error(f"  Data fetch error: {e}")
            return

        # Step 2: Compute features
        try:
            featured_df = self.feature_engine.compute_features(df, "H1")
        except Exception as e:
            logger.error(f"  Feature computation error: {e}")
            return

        # Step 3: Generate signals
        raw_signals = self.strategy_manager.scan_all(featured_df, "H1")
        filtered_signals = self.signal_filter.filter_signals(raw_signals, "H1")

        logger.info(f"  Signals: {len(raw_signals)} raw -> {len(filtered_signals)} filtered")

        if not filtered_signals:
            logger.info(f"  No quality signals found. Waiting...")
            return

        # Step 4: Process each signal
        mtf_blocked = 0
        news_blocked = 0
        ml_blocked = 0

        for signal in filtered_signals[:3]:  # Max 3 per scan

            # NEW: Multi-timeframe check
            mtf_allowed, mtf_reason, mtf_boost = self.mtf_filter.filter_signal(signal)
            if not mtf_allowed:
                logger.info(f"  MTF BLOCKED: {mtf_reason}")
                mtf_blocked += 1
                continue

            # Apply MTF confidence boost
            if mtf_boost > 0:
                old_conf = signal.confidence
                signal.confidence = min(signal.confidence + mtf_boost, 1.0)
                logger.info(f"  MTF boost: {old_conf:.2f} -> {signal.confidence:.2f} (+{mtf_boost:.2f})")

            # News check
            news_ok, news_reason, news_score = self.news_manager.can_trade(signal.direction)
            if not news_ok:
                logger.info(f"  News blocked: {news_reason}")
                news_blocked += 1
                continue

            # ML scoring
            if self.ml_loaded:
                current_bar = featured_df.iloc[-1]
                ml_score = self.ml_scorer.score_signal(signal, current_bar, self.feature_builder)
                signal.confidence = ml_score
                logger.info(f"  ML score: {ml_score:.2f} (threshold: 0.55)")

                if ml_score < 0.55:
                    logger.info(f"  Signal rejected by ML (score too low)")
                    ml_blocked += 1
                    continue

            # Risk manager check
            allowed, reason = self.risk_manager.can_trade(signal)
            if not allowed:
                logger.info(f"  Risk blocked: {reason}")
                continue

            # Position sizing
            account = self.executor.get_account_summary()
            self.risk_manager.current_balance = account.get("balance", self.risk_manager.current_balance)
            sizing = self.risk_manager.calculate_position_size(signal)

            if sizing["lot_size"] <= 0:
                continue

            # Check current open positions
            open_positions = self.executor.get_open_positions()
            if len(open_positions) >= self.risk_manager.max_open_positions:
                logger.info(f"  Max positions reached ({len(open_positions)})")
                continue

            # EXECUTE THE TRADE!
            logger.info(f"\n  >>> EXECUTING TRADE <<<")
            logger.info(f"  {signal}")

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

                # Log the trade
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
                    "ml_score": signal.confidence,
                    "news_score": news_score,
                    "h4_trend": self.mtf_filter.h4_trend,
                    "mtf_boost": mtf_boost,
                    "reason": signal.reason,
                    "result": "open",
                }
                self.trade_log.append(trade_record)
                self._save_trade_log()

                # Update risk manager
                self.risk_manager.daily_trades += 1

                # NEW: Telegram notification
                self.telegram.send_trade_opened({
                    "direction": signal.direction,
                    "strategy": signal.strategy_name,
                    "entry": result["price"],
                    "sl": signal.stop_loss,
                    "tp": signal.take_profit,
                    "lot_size": sizing["lot_size"],
                    "confidence": signal.confidence,
                })

                logger.info(f"  Trade #{self.total_trades_placed} placed + Telegram sent!")
            else:
                logger.error(f"  Trade failed: {result.get('error', 'Unknown')}")
                self.telegram.send_alert(f"Trade execution failed: {result.get('error', 'Unknown')}")

        # Log filter summary
        if mtf_blocked + news_blocked + ml_blocked > 0:
            logger.info(f"  Filter summary: MTF blocked {mtf_blocked}, "
                        f"News blocked {news_blocked}, ML blocked {ml_blocked}")

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

        print(f"\n  --- BOT STATUS ---")
        print(f"  Scans: {self.total_scans + 1} | Trades placed: {self.total_trades_placed}")
        print(f"  Balance: ${account.get('balance', 0):.2f} | "
              f"Equity: ${account.get('equity', 0):.2f} | "
              f"Profit: ${account.get('profit', 0):.2f}")
        print(f"  Open positions: {len(positions)}")
        print(f"  H4 Trend: {mtf_status['h4_trend']} (strength: {mtf_status['h4_strength']:.2f}) | "
              f"D1: {mtf_status['d1_trend']}")

        for pos in positions:
            pnl_icon = "\U0001F7E2" if pos["profit"] >= 0 else "\U0001F534"
            print(f"    {pnl_icon} #{pos['ticket']} {pos['direction']} {pos['volume']} lots "
                  f"@ {pos['price_open']:.2f} | PnL: ${pos['profit']:.2f}")

        # News status
        is_blackout, reason = self.news_manager.calendar.is_news_blackout(30)
        if is_blackout:
            print(f"  \U000026A0\U0000FE0F  NEWS BLACKOUT: {reason}")
        else:
            print(f"  \U00002705 News: Clear")

    def _shutdown(self):
        self.running = False

        positions = self.executor.get_open_positions()
        account = self.executor.get_account_summary()
        session_pnl = account.get("balance", 0) - self.session_start_balance

        print(f"\n{'='*70}")
        print(f"  LIVE TRADING SESSION ENDED")
        print(f"{'='*70}")
        print(f"  Total scans: {self.total_scans}")
        print(f"  Trades placed: {self.total_trades_placed}")
        print(f"  Open positions: {len(positions)}")
        print(f"  Balance: ${account.get('balance', 0):.2f}")
        print(f"  Equity: ${account.get('equity', 0):.2f}")
        print(f"  Session PnL: ${session_pnl:+.2f}")
        print(f"  Trade log: {self.trade_log_path}")
        print(f"{'='*70}\n")

        # NEW: Telegram shutdown summary
        self.telegram.send(
            "\U0001F6D1 <b>BOT STOPPED</b>\n"
            f"Scans: {self.total_scans}\n"
            f"Trades: {self.total_trades_placed}\n"
            f"Open: {len(positions)}\n"
            f"Balance: ${account.get('balance', 0):.2f}\n"
            f"Session PnL: ${session_pnl:+.2f}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )

        self.connector.disconnect()
'''

# ============================================================
# FILE 3: Test Telegram script
# ============================================================
FILES["test_telegram.py"] = r'''"""
Quick test: sends a test message to your Telegram.
Run: python test_telegram.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.alerts.telegram_alerts import TelegramAlerts

def main():
    print("\n  Testing Telegram alerts...")
    t = TelegramAlerts()

    if not t.enabled:
        print("  Telegram is DISABLED in settings.yaml!")
        print("  Make sure alerts.telegram_enabled is True")
        return

    success = t.send(
        "\U00002705 <b>XAUUSD AI TRADER - TEST</b>\n\n"
        "Your trading bot is connected!\n"
        "You will receive alerts for:\n"
        "- Trade opened\n"
        "- Trade closed (with PnL)\n"
        "- Daily summaries\n"
        "- System errors\n\n"
        "\U0001F916 Bot is ready!"
    )

    if success:
        print("  SUCCESS! Check your Telegram!")
    else:
        print("  FAILED! Check your token and chat_id in settings.yaml")

if __name__ == "__main__":
    main()
'''

# ============================================================
# Create all files
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("  Telegram + MTF Integration Setup")
    print("=" * 60 + "\n")

    for filepath, content in FILES.items():
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        print(f"  [FILE] {filepath}")

    print(f"\n  Created {len(FILES)} files.")
    print("\n  Run these in order:")
    print("  1. python setup_telegram_mtf.py")
    print("  2. python patch_config.py")
    print("  3. python test_telegram.py")
    print("  4. python live_trader.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
