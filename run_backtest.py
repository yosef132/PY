"""
XAUUSD AI Trading System - Backtest Runner
Run the full pipeline: Data -> Features -> Strategies -> Filter -> Risk Manager -> Results

Usage:
    python run_backtest.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collector.mt5_collector import MT5Connector, DataCollector
from src.feature_engine.engine import FeatureEngine
from src.risk_manager.backtester import Backtester
from src.utils.logger import setup_logger

logger = setup_logger("run_backtest")


def main():
    print("\n" + "=" * 70)
    print("  XAUUSD AI TRADING SYSTEM")
    print("  Phase 2 - Session 4: Risk Manager & Backtester")
    print("=" * 70 + "\n")

    # --- Step 1: Get data ---
    logger.info("Step 1: Fetching data from MT5...")
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

    # --- Step 3: Run backtest ---
    logger.info("\nStep 3: Running backtest...\n")
    backtester = Backtester(initial_balance=3000.0)

    # Backtest on H1 (best balance of signal quality and trade frequency)
    stats = backtester.run(featured_data, target_timeframes=["H1"])

    # --- Step 4: Print results ---
    backtester.print_results(stats)

    # --- Step 5: Also test H4 ---
    print("\n  Running H4 backtest for comparison...\n")
    backtester_h4 = Backtester(initial_balance=3000.0)
    stats_h4 = backtester_h4.run(featured_data, target_timeframes=["H4"])
    backtester_h4.print_results(stats_h4)

    print("\n" + "=" * 70)
    print("  Phase 2 - Session 4 COMPLETE!")
    print("  Risk Manager and Backtester are working!")
    print("  Next: Build ML Signal Scorer to improve signal quality")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
