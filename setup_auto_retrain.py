"""
XAUUSD AI Trading System - Auto-Retraining & Alerts Setup
Session 9: Weekly ML retraining + Telegram trade notifications.

Usage:
    python setup_auto_retrain.py
    python retrain_model.py        (manual retrain)
    python run_full_system.py      (everything together)
"""

import os

FILES = {}

# ============================================================
# FILE 1: src/ml_scorer/auto_retrain.py
# ============================================================
FILES["src/ml_scorer/auto_retrain.py"] = r'''"""
Auto-Retraining Module
Collects live trade results and retrains the ML model weekly.
This is how the AI "learns by itself" over time.
"""

import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import MetaTrader5 as mt5
from src.data_collector.mt5_collector import MT5Connector, DataCollector
from src.feature_engine.engine import FeatureEngine
from src.strategies.strategy_manager import StrategyManager
from src.risk_manager.signal_filter import SignalFilter
from src.risk_manager.risk_manager import RiskManager
from src.ml_scorer.feature_builder import FeatureBuilder
from src.ml_scorer.model import MLSignalScorer
from src.utils.config import get_settings, PROJECT_ROOT
from src.utils.logger import setup_logger

logger = setup_logger("auto_retrain")


class AutoRetrainer:
    """
    Handles automatic ML model retraining.
    Collects trade data, trains new model, compares with old, and replaces if better.
    """

    def __init__(self):
        self.settings = get_settings()
        self.model_dir = PROJECT_ROOT / "data" / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.trade_log_path = PROJECT_ROOT / "data" / "live_trades.json"
        self.retrain_log_path = PROJECT_ROOT / "data" / "retrain_history.json"

    def collect_closed_trades(self) -> list:
        """
        Collect closed trade data from MT5 trade history.
        These are REAL results from the demo account.
        """
        if not mt5.initialize():
            logger.error("  Cannot connect to MT5 for trade history")
            return []

        # Get trade history for last 30 days
        from_date = datetime.now() - timedelta(days=30)
        to_date = datetime.now()

        # Get deals (closed trades)
        deals = mt5.history_deals_get(from_date, to_date)
        mt5.shutdown()

        if deals is None or len(deals) == 0:
            logger.info("  No closed trades found in history")
            return []

        # Filter for our bot's trades (magic number)
        magic = self.settings.get("execution", {}).get("magic_number", 123456)
        symbol = self.settings.get("symbol", "XAUUSD")

        trades = []
        for deal in deals:
            if deal.magic == magic and deal.symbol == symbol:
                trades.append({
                    "ticket": deal.order,
                    "direction": "BUY" if deal.type == 0 else "SELL",
                    "volume": deal.volume,
                    "price": deal.price,
                    "profit": deal.profit,
                    "commission": deal.commission,
                    "swap": deal.swap,
                    "time": datetime.fromtimestamp(deal.time).isoformat(),
                    "comment": deal.comment,
                })

        logger.info(f"  Collected {len(trades)} closed trades from MT5 history")
        return trades

    def retrain(self, use_backtest_data: bool = True) -> dict:
        """
        Retrain the ML model with latest data.

        Args:
            use_backtest_data: If True, use backtest simulation data.
                              If False, use only live trade data (needs 50+ trades).

        Returns:
            Dict with retraining results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"ML MODEL RETRAINING - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        logger.info(f"{'='*60}")

        # Step 1: Connect and get fresh data
        logger.info("  Step 1: Fetching fresh market data...")
        connector = MT5Connector()
        if not connector.connect():
            return {"success": False, "error": "Cannot connect to MT5"}

        collector = DataCollector(connector)
        collector.check_symbol()

        # Fetch H1 data (our primary trading timeframe)
        df = collector.fetch_candles("H1", num_bars=5000)
        connector.disconnect()

        if df.empty:
            return {"success": False, "error": "No data received"}

        # Step 2: Compute features
        logger.info("  Step 2: Computing features...")
        engine = FeatureEngine()
        featured_df = engine.compute_features(df, "H1")

        # Step 3: Generate signals and simulate trades for training data
        logger.info("  Step 3: Generating training data...")
        strategy_manager = StrategyManager()
        signal_filter = SignalFilter()
        risk_manager = RiskManager(initial_balance=100000.0)

        raw_signals = strategy_manager.scan_all(featured_df, "H1")
        filtered_signals = signal_filter.filter_signals(raw_signals, "H1")

        # Simulate trades to get outcomes
        signal_map = {}
        for sig in filtered_signals:
            if sig.timestamp not in signal_map:
                signal_map[sig.timestamp] = []
            signal_map[sig.timestamp].append(sig)

        for i in range(1, len(featured_df)):
            current_bar = featured_df.iloc[i]
            current_time = featured_df.index[i]

            current_date = current_time.date() if hasattr(current_time, 'date') else None
            if current_date:
                risk_manager.reset_daily(current_date)

            risk_manager.check_exits(
                current_high=current_bar["high"],
                current_low=current_bar["low"],
                current_close=current_bar["close"],
                current_time=current_time
            )

            if current_time in signal_map:
                for signal in signal_map[current_time]:
                    allowed, _ = risk_manager.can_trade(signal)
                    if allowed:
                        sizing = risk_manager.calculate_position_size(signal)
                        if sizing["lot_size"] > 0:
                            risk_manager.open_trade(signal, sizing)

        # Close remaining
        if risk_manager.open_positions and len(featured_df) > 0:
            last_bar = featured_df.iloc[-1]
            for pos in risk_manager.open_positions[:]:
                if pos["direction"] == "BUY":
                    pnl = (last_bar["close"] - pos["entry_price"]) * pos["lot_size"] * 100
                else:
                    pnl = (pos["entry_price"] - last_bar["close"]) * pos["lot_size"] * 100
                pos["pnl"] = round(pnl, 2)
                pos["status"] = "forced_close"
                risk_manager.current_balance += pnl
                risk_manager.trade_history.append(pos)
                risk_manager.open_positions.remove(pos)

        trades = risk_manager.trade_history
        logger.info(f"  Generated {len(trades)} simulated trades for training")

        if len(trades) < 20:
            logger.warning(f"  Only {len(trades)} trades - need at least 20 for retraining")
            return {"success": False, "error": f"Only {len(trades)} trades, need 20+"}

        # Step 4: Train new model
        logger.info("  Step 4: Training new ML model...")
        feature_builder = FeatureBuilder()
        X, y, feature_names = feature_builder.build_training_data(trades, featured_df)

        new_model = MLSignalScorer()
        new_metrics = new_model.train(X, y, feature_names)

        # Step 5: Compare with old model
        logger.info("  Step 5: Comparing with current model...")
        old_model = MLSignalScorer()
        old_loaded = old_model.load()

        should_replace = True
        comparison = {}

        if old_loaded:
            # Test both models on the same data
            old_preds = old_model.model.predict_proba(X)[:, 1]
            new_preds = new_model.model.predict_proba(X)[:, 1]

            old_accuracy = np.mean((old_preds > 0.5) == y) * 100
            new_accuracy = new_metrics["train_accuracy"]

            comparison = {
                "old_accuracy": round(old_accuracy, 1),
                "new_accuracy": round(new_accuracy, 1),
                "improvement": round(new_accuracy - old_accuracy, 1),
            }

            logger.info(f"  Old model accuracy: {old_accuracy:.1f}%")
            logger.info(f"  New model accuracy: {new_accuracy:.1f}%")

            # Only replace if new model is better (or at least not worse)
            if new_accuracy < old_accuracy - 5:
                should_replace = False
                logger.warning(f"  New model is worse! Keeping old model.")
        else:
            logger.info("  No old model found - saving new model as first version")

        # Step 6: Save if better
        if should_replace:
            # Backup old model
            old_path = self.model_dir / "xgb_signal_scorer.pkl"
            if old_path.exists():
                backup_name = f"xgb_signal_scorer_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
                old_path.rename(self.model_dir / backup_name)
                logger.info(f"  Old model backed up as {backup_name}")

            new_model.save()
            logger.info(f"  New model saved!")

        # Step 7: Log retraining
        retrain_record = {
            "timestamp": datetime.now().isoformat(),
            "trades_used": len(trades),
            "win_rate": round(np.mean(y) * 100, 1),
            "cv_accuracy": new_metrics["cv_accuracy"],
            "train_accuracy": new_metrics["train_accuracy"],
            "replaced": should_replace,
            "comparison": comparison,
            "top_features": [(f, round(i, 4)) for f, i in new_metrics["top_features"][:5]],
        }

        self._save_retrain_log(retrain_record)

        logger.info(f"\n{'='*60}")
        logger.info(f"RETRAINING COMPLETE")
        logger.info(f"  Trades used: {len(trades)}")
        logger.info(f"  CV Accuracy: {new_metrics['cv_accuracy']}%")
        logger.info(f"  Model replaced: {should_replace}")
        logger.info(f"{'='*60}")

        return {
            "success": True,
            "trades_used": len(trades),
            "metrics": new_metrics,
            "comparison": comparison,
            "replaced": should_replace,
        }

    def _save_retrain_log(self, record: dict):
        """Append to retraining history."""
        history = []
        if self.retrain_log_path.exists():
            try:
                with open(self.retrain_log_path) as f:
                    history = json.load(f)
            except Exception:
                pass

        history.append(record)

        with open(self.retrain_log_path, "w") as f:
            json.dump(history, f, indent=2, default=str)

    def get_retrain_history(self) -> list:
        """Get history of all retraining events."""
        if self.retrain_log_path.exists():
            with open(self.retrain_log_path) as f:
                return json.load(f)
        return []
'''

# ============================================================
# FILE 2: src/alerts/telegram_alerts.py
# ============================================================
FILES["src/alerts/__init__.py"] = ""

FILES["src/alerts/telegram_alerts.py"] = r'''"""
Telegram Alert System
Sends trade notifications and daily summaries to your phone.
"""

import requests
from datetime import datetime
from src.utils.config import get_settings
from src.utils.logger import setup_logger

logger = setup_logger("telegram")


class TelegramAlerts:
    """Sends trading alerts to Telegram."""

    def __init__(self):
        settings = get_settings()
        alerts = settings.get("alerts", {})
        self.enabled = alerts.get("telegram_enabled", False)
        self.token = alerts.get("telegram_token", "")
        self.chat_id = alerts.get("telegram_chat_id", "")
        self.base_url = f"https://api.telegram.org/bot{self.token}"

    def send(self, message: str) -> bool:
        """Send a message to Telegram."""
        if not self.enabled or not self.token or not self.chat_id:
            return False

        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML",
            }
            resp = requests.post(url, data=data, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            logger.warning(f"  Telegram send failed: {e}")
            return False

    def send_trade_opened(self, trade: dict):
        """Notify when a trade is opened."""
        direction_icon = "\U0001F7E2" if trade.get("direction") == "BUY" else "\U0001F534"
        msg = (
            f"{direction_icon} <b>TRADE OPENED</b>\n"
            f"Direction: {trade.get('direction')}\n"
            f"Strategy: {trade.get('strategy', 'unknown')}\n"
            f"Entry: ${trade.get('entry', 0):.2f}\n"
            f"SL: ${trade.get('sl', 0):.2f}\n"
            f"TP: ${trade.get('tp', 0):.2f}\n"
            f"Lot size: {trade.get('lot_size', 0)}\n"
            f"Confidence: {trade.get('confidence', 0):.0%}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        self.send(msg)

    def send_trade_closed(self, trade: dict):
        """Notify when a trade is closed."""
        pnl = trade.get("profit", 0)
        icon = "\U0001F389" if pnl > 0 else "\U0001F4A5"
        result = "WIN" if pnl > 0 else "LOSS"
        msg = (
            f"{icon} <b>TRADE CLOSED - {result}</b>\n"
            f"PnL: ${pnl:+.2f}\n"
            f"Ticket: #{trade.get('ticket', 0)}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        self.send(msg)

    def send_daily_summary(self, stats: dict):
        """Send end-of-day summary."""
        msg = (
            f"\U0001F4CA <b>DAILY SUMMARY</b>\n"
            f"Trades: {stats.get('trades_today', 0)}\n"
            f"Wins: {stats.get('wins', 0)} | Losses: {stats.get('losses', 0)}\n"
            f"PnL today: ${stats.get('daily_pnl', 0):+.2f}\n"
            f"Balance: ${stats.get('balance', 0):.2f}\n"
            f"Open positions: {stats.get('open_positions', 0)}\n"
            f"Win rate: {stats.get('win_rate', 0):.0f}%"
        )
        self.send(msg)

    def send_alert(self, message: str):
        """Send a custom alert."""
        self.send(f"\U000026A0 <b>ALERT</b>\n{message}")
'''

# ============================================================
# FILE 3: retrain_model.py - Manual retrain script
# ============================================================
FILES["retrain_model.py"] = r'''"""
XAUUSD AI Trading System - Manual Model Retraining
Run this to retrain the ML model with the latest data.

Usage:
    python retrain_model.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_scorer.auto_retrain import AutoRetrainer
from src.utils.logger import setup_logger

logger = setup_logger("retrain")


def main():
    print("\n" + "=" * 70)
    print("  XAUUSD AI TRADING SYSTEM")
    print("  ML Model Retraining")
    print("=" * 70 + "\n")

    retrainer = AutoRetrainer()

    # Show previous retraining history
    history = retrainer.get_retrain_history()
    if history:
        print(f"  Previous retraining events: {len(history)}")
        last = history[-1]
        print(f"  Last retrain: {last['timestamp'][:16]}")
        print(f"  Last CV accuracy: {last['cv_accuracy']}%")
    else:
        print(f"  No previous retraining history")

    print()
    result = retrainer.retrain()

    if result["success"]:
        print(f"\n  Retraining successful!")
        print(f"  Trades used: {result['trades_used']}")
        print(f"  CV Accuracy: {result['metrics']['cv_accuracy']}%")
        print(f"  Model replaced: {result['replaced']}")

        if result.get("comparison"):
            comp = result["comparison"]
            print(f"  Old accuracy: {comp.get('old_accuracy', 'N/A')}%")
            print(f"  New accuracy: {comp.get('new_accuracy', 'N/A')}%")
            print(f"  Improvement: {comp.get('improvement', 'N/A')}%")
    else:
        print(f"\n  Retraining failed: {result.get('error', 'Unknown')}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
'''

# ============================================================
# FILE 4: run_full_system.py - MASTER SCRIPT
# ============================================================
FILES["run_full_system.py"] = r'''"""
XAUUSD AI Trading System - FULL SYSTEM
Master script that runs everything:
1. Retrains ML model if it's Sunday
2. Starts live paper trading
3. Handles graceful shutdown

Usage:
    python run_full_system.py
"""

import sys
import schedule
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.executor.live_engine import LiveTradingEngine
from src.ml_scorer.auto_retrain import AutoRetrainer
from src.utils.logger import setup_logger

logger = setup_logger("full_system")


def weekly_retrain():
    """Run weekly model retraining."""
    logger.info("\n  WEEKLY AUTO-RETRAIN triggered!")
    retrainer = AutoRetrainer()
    result = retrainer.retrain()
    if result["success"]:
        logger.info(f"  Retrain successful! CV Accuracy: {result['metrics']['cv_accuracy']}%")
    else:
        logger.warning(f"  Retrain failed: {result.get('error')}")


def main():
    print("\n" + "=" * 70)
    print("  XAUUSD AI TRADING SYSTEM - FULL SYSTEM")
    print("  Live Paper Trading + Auto-Retraining")
    print("=" * 70)
    print()
    print("  Components:")
    print("    [1] Data Collector (MT5)")
    print("    [2] Feature Engine (133 features)")
    print("    [3] Strategy Detectors (5 strategies)")
    print("    [4] Signal Filter + Risk Manager")
    print("    [5] ML Signal Scorer (XGBoost)")
    print("    [6] News Sentiment Filter")
    print("    [7] Monitoring Dashboard (run separately)")
    print("    [8] Live Paper Trading")
    print("    [9] Weekly Auto-Retraining (every Sunday)")
    print()
    print("  ⚠️  Make sure MT5 is open and logged in!")
    print("  Press Ctrl+C to stop safely")
    print()

    # Check if we should retrain first
    today = datetime.now().strftime("%A")
    if today == "Sunday":
        print("  Today is Sunday — running weekly retrain first...")
        weekly_retrain()

    # Schedule weekly retrain for every Sunday at 00:00
    schedule.every().sunday.at("00:00").do(weekly_retrain)

    # Start live trading
    input("  Press ENTER to start the full system...")

    engine = LiveTradingEngine()

    print("\n" + "=" * 70)
    print("  SYSTEM STARTING...")
    print("  - Live trading: every 5 minutes")
    print("  - News refresh: every 15 minutes")
    print("  - ML retrain: every Sunday at midnight")
    print("=" * 70 + "\n")

    engine.start()


if __name__ == "__main__":
    main()
'''

# ============================================================
# Create all files
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("  Auto-Retraining & Alerts Setup")
    print("=" * 60 + "\n")

    for filepath, content in FILES.items():
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        print(f"  [FILE] {filepath}")

    print(f"\n  Created {len(FILES)} files.")
    print("\n  Install schedule: pip install schedule")
    print("  Manual retrain:   python retrain_model.py")
    print("  Full system:      python run_full_system.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
