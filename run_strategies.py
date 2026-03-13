"""
XAUUSD AI Trading System - Strategy Detector Test
Run all strategies on your data and see what signals are generated.

Usage:
    python run_strategies.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collector.mt5_collector import MT5Connector, DataCollector
from src.feature_engine.engine import FeatureEngine
from src.strategies.strategy_manager import StrategyManager
from src.utils.logger import setup_logger

logger = setup_logger("run_strategies")


def main():
    print("\n" + "=" * 70)
    print("  XAUUSD AI TRADING SYSTEM")
    print("  Phase 2 - Session 3: Strategy Detectors")
    print("=" * 70 + "\n")

    # --- Step 1: Get fresh data ---
    logger.info("Step 1: Fetching fresh data from MT5...")
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

    # --- Step 3: Run strategy detectors ---
    logger.info("\nStep 3: Running strategy detectors...\n")
    manager = StrategyManager()
    all_signals = manager.scan_multi_timeframe(featured_data)

    # --- Step 4: Results ---
    print("\n" + "=" * 70)
    print("  STRATEGY DETECTOR RESULTS")
    print("=" * 70)

    total_signals = 0
    strategy_counts = {}

    for tf_name, signals in all_signals.items():
        total_signals += len(signals)
        print(f"\n  --- {tf_name}: {len(signals)} signals ---")

        for sig in signals[:5]:  # Show top 5 per timeframe
            print(f"    {sig}")
            strat = sig.strategy_name
            strategy_counts[strat] = strategy_counts.get(strat, 0) + 1

        if len(signals) > 5:
            print(f"    ... and {len(signals) - 5} more")

    print(f"\n  {'='*50}")
    print(f"  TOTAL: {total_signals} signals across all timeframes")
    print(f"  {'='*50}")

    if strategy_counts:
        print(f"\n  Signals by strategy:")
        for strat, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    {strat:>25}: {count} signals")

    # --- Step 5: Signal quality analysis ---
    print(f"\n  {'='*50}")
    print(f"  SIGNAL QUALITY ANALYSIS")
    print(f"  {'='*50}")

    all_sigs_flat = []
    for signals in all_signals.values():
        all_sigs_flat.extend(signals)

    if all_sigs_flat:
        confidences = [s.confidence for s in all_sigs_flat]
        rrs = [s.risk_reward for s in all_sigs_flat if s.risk_reward > 0]

        print(f"\n  Avg confidence: {sum(confidences)/len(confidences):.1%}")
        print(f"  Max confidence: {max(confidences):.1%}")
        print(f"  Min confidence: {min(confidences):.1%}")
        if rrs:
            print(f"  Avg R:R ratio:  {sum(rrs)/len(rrs):.1f}")
            print(f"  Best R:R ratio: {max(rrs):.1f}")

        # High confidence signals
        high_conf = [s for s in all_sigs_flat if s.confidence >= 0.70]
        print(f"\n  High confidence signals (>= 70%): {len(high_conf)}")
        for sig in high_conf[:10]:
            print(f"    {sig}")

    print("\n" + "=" * 70)
    print("  Phase 2 - Session 3 COMPLETE!")
    print("  Strategy detectors are working!")
    print("  Next: Build Risk Manager and connect to ML scorer")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
