"""
XAUUSD AI Trading System - Live Paper Trader
Runs the bot continuously on your MT5 demo account.

The bot will:
1. Scan the market every 5 minutes
2. Compute 133 features on H1 data
3. Run all 5 strategy detectors
4. Filter signals (min 2:1 R:R, min 60% confidence)
5. Apply ML scoring (blocks low-probability trades)
6. Check news (blocks during high-impact events)
7. Place trades on your MT5 demo account
8. Log everything

Usage:
    python live_trader.py

    Press Ctrl+C to stop the bot safely.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.executor.live_engine import LiveTradingEngine
from src.utils.logger import setup_logger

logger = setup_logger("live_trader")


def main():
    print("\n" + "=" * 70)
    print("  XAUUSD AI TRADING SYSTEM")
    print("  Session 8: Live Paper Trading on MT5 Demo")
    print("=" * 70)
    print()
    print("  !! IMPORTANT REMINDERS:")
    print("  - This trades on your DEMO account only")
    print("  - Make sure MT5 is running and logged in")
    print("  - The bot scans every 5 minutes")
    print("  - Press Ctrl+C at any time to stop safely")
    print("  - All trades are logged in data/live_trades.json")
    print()

    input("  Press ENTER to start the bot...")

    engine = LiveTradingEngine()
    engine.start()


if __name__ == "__main__":
    main()
