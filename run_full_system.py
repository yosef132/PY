"""
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
