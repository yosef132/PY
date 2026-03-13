"""
Save backtest results to JSON for the dashboard to read.
Run this after run_ml_backtest.py to populate the dashboard.

Usage:
    python save_backtest_results.py
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collector.mt5_collector import MT5Connector, DataCollector
from src.feature_engine.engine import FeatureEngine
from src.ml_scorer.enhanced_backtester import EnhancedBacktester
from src.utils.logger import setup_logger

logger = setup_logger("save_results")


def main():
    print("\n  Generating backtest results for dashboard...\n")

    # Get data
    connector = MT5Connector()
    if not connector.connect():
        print("  Cannot connect to MT5!")
        return

    collector = DataCollector(connector)
    collector.check_symbol()
    data = collector.fetch_all_timeframes()

    # Save raw candle data
    collector.save_data(data)
    connector.disconnect()

    # Compute features
    engine = FeatureEngine()
    featured_data = engine.compute_multi_timeframe(data)

    # Run ML backtest
    backtester = EnhancedBacktester(initial_balance=3000.0)
    results = backtester.run_full_pipeline(featured_data, timeframe="H1")

    # Save results as JSON for dashboard
    output_path = PROJECT_ROOT / "data" / "backtest_results.json"

    ml_stats = results.get("ml_enhanced", results.get("baseline", {}))

    # Make serializable
    save_data = {
        "stats": {k: v for k, v in ml_stats.items() if k != "balance_curve"},
        "balance_curve": ml_stats.get("balance_curve", []),
        "baseline": {k: v for k, v in results.get("baseline", {}).items() if k != "balance_curve"},
        "ml_metrics": results.get("ml_metrics", {}),
        "timestamp": str(Path(__file__).stat().st_mtime),
    }

    # Convert numpy types
    def convert(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return obj

    clean = json.loads(json.dumps(save_data, default=convert))

    with open(output_path, "w") as f:
        json.dump(clean, f, indent=2)

    print(f"\n  Results saved to: {output_path}")
    print(f"  Now run: streamlit run dashboard.py\n")


if __name__ == "__main__":
    main()
