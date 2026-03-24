"""
News Manager - Unified news intelligence module.
Combines economic calendar + headline sentiment to make trade decisions.
Updated: accepts blackout_minutes parameter.
"""

from src.news_filter.economic_calendar import EconomicCalendar
from src.news_filter.headline_sentiment import HeadlineSentiment
from src.news_filter.news_scraper import GoldNewsScraper as NewsScraper
from src.utils.logger import setup_logger

logger = setup_logger("news_manager")


class NewsManager:
    """Unified news manager: calendar blackouts + sentiment scoring."""

    def __init__(self):
        self.calendar = EconomicCalendar()
        self.sentiment_scorer = HeadlineSentiment()
        self.scraper = NewsScraper()

        self.headlines = []
        self.sentiment = {}

    def refresh(self):
        """Refresh all news data."""
        logger.info("  Refreshing news data...")

        # Economic calendar
        self.calendar.fetch_today_events()

        # Headlines (scraper returns list of strings)
        self.headlines = self.scraper.fetch_all_headlines()

        # Score sentiment
        if self.headlines:
            self.sentiment = self.sentiment_scorer.score_headlines(self.headlines)
        else:
            self.sentiment = {"avg_score": 0, "direction": "neutral", "confidence": 0}

        direction = self.sentiment.get("direction", "neutral")
        score = self.sentiment.get("score", 0)
        logger.info(f"  News refresh complete | Sentiment: {direction} ({score:.2f})")

    def can_trade(self, direction: str, blackout_minutes: int = 15) -> tuple:
        """
        Check if trading is allowed based on news.

        Args:
            direction: "BUY" or "SELL"
            blackout_minutes: Minutes before/after high-impact news to block

        Returns:
            (allowed: bool, reason: str, sentiment_score: float)
        """
        # Check news blackout
        is_blackout, blackout_reason = self.calendar.is_news_blackout(blackout_minutes)
        if is_blackout:
            return False, f"NEWS BLACKOUT: {blackout_reason}", 0

        # Get sentiment
        sent_score = self.sentiment.get("avg_score", self.sentiment.get("score", 0))
        sent_dir = self.sentiment.get("direction", "neutral")

        # Strong opposing sentiment blocks trade
        if direction == "BUY" and sent_score < -0.5:
            return False, f"Strong bearish sentiment ({sent_score:.2f})", sent_score
        if direction == "SELL" and sent_score > 0.5:
            return False, f"Strong bullish sentiment ({sent_score:.2f})", sent_score

        return True, "News OK", sent_score

    def get_status(self) -> dict:
        """Get current news status summary."""
        is_blackout, reason = self.calendar.is_news_blackout(15)
        return {
            "sentiment": self.sentiment,
            "is_blackout": is_blackout,
            "blackout_reason": reason if is_blackout else None,
            "headline_count": len(self.headlines),
        }
