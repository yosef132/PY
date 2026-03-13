"""
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
