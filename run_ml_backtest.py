"""
XAUUSD AI Trading System - ML Enhanced Backtest
Trains XGBoost on trade history, then re-runs backtest with ML scoring.

Usage:
    python run_ml_backtest.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collector.mt5_collector import MT5Connector, DataCollector
from src.feature_engine.engine import FeatureEngine
from src.ml_scorer.enhanced_backtester import EnhancedBacktester
from src.utils.logger import setup_logger

logger = setup_logger("run_ml_backtest")


def main():
    print("\n" + "=" * 70)
    print("  XAUUSD AI TRADING SYSTEM")
    print("  Phase 3 - Session 5: ML Signal Scorer")
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

    # --- Step 3: Run ML Pipeline ---
    logger.info("\nStep 3: Running ML-enhanced backtest pipeline...\n")
    backtester = EnhancedBacktester(initial_balance=3000.0)

    # Run on H1 (our best timeframe from previous backtest)
    results = backtester.run_full_pipeline(featured_data, timeframe="H1")

    # --- Step 4: Print comparison ---
    backtester.print_comparison(results)

    print("\n" + "=" * 70)
    print("  Phase 3 - Session 5 COMPLETE!")
    print("  ML model is trained and saved in: data/models/")
    print("  The model will improve as more trade data is collected.")
    print("  Next: Connect to MT5 demo for live paper trading")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
