"""
XAUUSD AI Trading System - Feature Engine Test
Run this to compute all features on your saved data.

Usage:
    python run_features.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collector.mt5_collector import MT5Connector, DataCollector
from src.feature_engine.engine import FeatureEngine
from src.utils.logger import setup_logger

logger = setup_logger("run_features")


def main():
    print("\n" + "=" * 70)
    print("  XAUUSD AI TRADING SYSTEM")
    print("  Phase 1 - Session 2: Feature Engine")
    print("=" * 70 + "\n")

    # --- Step 1: Load data ---
    logger.info("Step 1: Connecting to MT5 and fetching fresh data...")
    connector = MT5Connector()

    if not connector.connect():
        logger.error("Cannot connect to MT5!")
        return

    collector = DataCollector(connector)
    collector.check_symbol()
    data = collector.fetch_all_timeframes()
    connector.disconnect()

    if not data:
        logger.error("No data to process!")
        return

    # --- Step 2: Compute features ---
    logger.info("\nStep 2: Computing features for all timeframes...\n")
    engine = FeatureEngine()
    featured_data = engine.compute_multi_timeframe(data)

    # --- Step 3: Summary ---
    print("\n" + "=" * 70)
    print("  FEATURE ENGINE RESULTS")
    print("=" * 70)

    for tf_name, df in featured_data.items():
        summary = engine.get_feature_summary(df)
        total_features = sum(summary.values())

        print(f"\n  {tf_name}: {len(df)} candles x {len(df.columns)} columns ({total_features} features)")
        for cat, count in summary.items():
            if count > 0:
                print(f"    {cat:>12}: {count} features")

    # --- Step 4: Save featured data ---
    logger.info("\nStep 3: Saving featured data...")
    data_dir = PROJECT_ROOT / "data" / "historical"
    data_dir.mkdir(parents=True, exist_ok=True)

    for tf_name, df in featured_data.items():
        filepath = data_dir / f"XAUUSD_{tf_name}_features.parquet"
        df.to_parquet(filepath, engine="pyarrow")
        size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"  Saved {tf_name}: {len(df.columns)} columns -> {filepath.name} ({size_mb:.2f} MB)")

    # --- Step 5: Show sample data ---
    print("\n" + "=" * 70)
    print("  SAMPLE: Last 3 H1 candles with key features")
    print("=" * 70)

    h1 = featured_data.get("H1")
    if h1 is not None:
        key_cols = ["close", "ema_21", "rsi_14", "atr_14", "market_trend",
                    "wyckoff_phase", "sr_strength", "killzone_active"]
        available = [c for c in key_cols if c in h1.columns]
        print(h1[available].tail(3).to_string())

    print("\n" + "=" * 70)
    print("  Phase 1 - Session 2 COMPLETE!")
    print("  Featured data saved in: data/historical/")
    print("  Next: Build strategy detectors")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
