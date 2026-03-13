"""
XAUUSD AI Trading System - News Filter Test
Check current economic calendar and headline sentiment.

Usage:
    python run_news_check.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.news_filter.news_manager import NewsManager
from src.news_filter.headline_sentiment import HeadlineSentiment
from src.utils.logger import setup_logger

logger = setup_logger("run_news")


def main():
    print("\n" + "=" * 70)
    print("  XAUUSD AI TRADING SYSTEM")
    print("  Phase 3 - Session 6: News Sentiment Filter")
    print("=" * 70 + "\n")

    # --- Step 1: Initialize and refresh news ---
    logger.info("Step 1: Fetching news data...\n")
    manager = NewsManager()
    manager.refresh()

    # --- Step 2: Display status ---
    logger.info("\nStep 2: News status report\n")
    manager.print_status()

    # --- Step 3: Test trade checks ---
    print(f"\n  {'='*60}")
    print(f"  TRADE PERMISSION CHECKS")
    print(f"  {'='*60}")

    for direction in ["BUY", "SELL"]:
        allowed, reason, score = manager.can_trade(direction)
        status = "ALLOWED" if allowed else "BLOCKED"
        print(f"\n  {direction} trade: {status}")
        print(f"    Reason: {reason}")
        print(f"    Sentiment score: {score:.2f}")

    # --- Step 4: Test sentiment on sample headlines ---
    print(f"\n  {'='*60}")
    print(f"  SENTIMENT ENGINE TEST")
    print(f"  {'='*60}")

    test_headlines = [
        "Fed signals rate cuts ahead as inflation cools",
        "Gold surges to record high amid Middle East tensions",
        "Strong US jobs report beats expectations, dollar rallies",
        "ECB holds rates steady, signals caution on growth",
        "Russia-Ukraine conflict escalation drives safe haven demand",
        "CPI comes in higher than expected, markets sell off",
        "Gold falls as treasury yields rise sharply",
        "Central banks increase gold reserves to record levels",
    ]

    sentiment = HeadlineSentiment()
    print()
    for headline in test_headlines:
        result = sentiment.score_headline(headline)
        icon = "🟢" if result["score"] > 0.2 else ("🔴" if result["score"] < -0.2 else "⚪")
        print(f"  {icon} [{result['score']:+.2f}] {result['direction']:>8} | {headline[:65]}")

    aggregate = sentiment.score_headlines(test_headlines)
    print(f"\n  Aggregate: {aggregate['direction'].upper()} "
          f"(score: {aggregate['avg_score']:.2f}, "
          f"bullish: {aggregate['bullish_count']}, "
          f"bearish: {aggregate['bearish_count']}, "
          f"neutral: {aggregate['neutral_count']})")

    print("\n" + "=" * 70)
    print("  Phase 3 - Session 6 COMPLETE!")
    print("  News filter is active!")
    print("  The bot will now block trades during high-impact events")
    print("  and check headline sentiment before every trade.")
    print("  Next: Build monitoring dashboard")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
