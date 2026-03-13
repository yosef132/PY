"""
XAUUSD AI Trading System - Main Entry Point
Phase 1: Data Collection & Verification

Usage:
    python main.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collector.mt5_collector import MT5Connector, DataCollector
from src.utils.logger import setup_logger

logger = setup_logger("main")


def main():
    print("\n" + "=" * 70)
    print("  XAUUSD AI TRADING SYSTEM")
    print("  Phase 1: Data Collection")
    print("=" * 70 + "\n")

    # --- Step 1: Connect to MT5 ---
    logger.info("Step 1: Connecting to MetaTrader 5...")
    connector = MT5Connector()

    if not connector.connect():
        logger.error("Cannot connect to MT5. Make sure the terminal is running!")
        return

    account = connector.get_account_info()
    print(f"\n  Account: {account['login']} ({account['trade_mode']})")
    print(f"  Server:  {account['server']}")
    print(f"  Balance: ${account['balance']:.2f}")
    print(f"  Leverage: 1:{account['leverage']}\n")

    # --- Step 2: Check symbol ---
    logger.info("Step 2: Checking XAUUSD symbol...")
    collector = DataCollector(connector)

    if not collector.check_symbol():
        logger.error("XAUUSD not available. Check your broker or symbol name.")
        connector.disconnect()
        return

    # --- Step 3: Fetch historical data ---
    logger.info("Step 3: Fetching historical data for all timeframes...")
    data = collector.fetch_all_timeframes()

    if not data:
        logger.error("No data collected!")
        connector.disconnect()
        return

    # --- Step 4: Save to disk ---
    logger.info("Step 4: Saving data to Parquet files...")
    collector.save_data(data)

    # --- Step 5: Summary ---
    collector.print_data_summary(data)

    # --- Step 6: Data quality checks ---
    logger.info("Step 5: Running data quality checks...")
    for tf_name, df in data.items():
        nulls = df.isnull().sum().sum()
        if nulls > 0:
            logger.warning(f"{tf_name}: Found {nulls} missing values!")
        else:
            logger.info(f"{tf_name}: No missing values - OK")

        zero_vol = (df["volume"] == 0).sum()
        if zero_vol > 0:
            logger.warning(f"{tf_name}: Found {zero_vol} zero-volume candles")

        dupes = df.index.duplicated().sum()
        if dupes > 0:
            logger.warning(f"{tf_name}: Found {dupes} duplicate timestamps!")
        else:
            logger.info(f"{tf_name}: No duplicates - OK")

    print("\n" + "=" * 70)
    print("  Phase 1 - Step 1 COMPLETE!")
    print("  Data is saved in: data/historical/")
    print("  Next: Run feature_engine to compute indicators")
    print("=" * 70 + "\n")

    connector.disconnect()


if __name__ == "__main__":
    main()
