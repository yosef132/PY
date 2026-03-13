"""
XAUUSD AI Trading System - News Sentiment Filter Setup
Session 6: News economic calendar + gold headline sentiment + trade blocking.

Usage:
    python setup_news_filter.py
    python run_news_check.py
"""

import os

FILES = {}

# ============================================================
# FILE 1: src/news_filter/__init__.py
# ============================================================
FILES["src/news_filter/__init__.py"] = ""

# ============================================================
# FILE 2: src/news_filter/economic_calendar.py
# ============================================================
FILES["src/news_filter/economic_calendar.py"] = r'''"""
Economic Calendar Module
Fetches upcoming high-impact economic events that affect gold (XAUUSD).
Uses free APIs + built-in known events database as fallback.
Gold is primarily affected by: USD events (Fed, CPI, NFP, GDP, PPI).
"""

import requests
import json
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from src.utils.logger import setup_logger

logger = setup_logger("eco_calendar")


# ============================================================
# Gold-critical events — these move XAUUSD the most
# ============================================================
GOLD_CRITICAL_EVENTS = {
    # Event keyword -> impact level (1=low, 2=medium, 3=high)
    "fed interest rate": 3,
    "federal funds rate": 3,
    "fomc": 3,
    "fomc minutes": 3,
    "fomc press conference": 3,
    "fed chair powell": 3,
    "non-farm payrolls": 3,
    "nonfarm payrolls": 3,
    "nfp": 3,
    "cpi": 3,
    "consumer price index": 3,
    "core cpi": 3,
    "ppi": 2,
    "producer price index": 2,
    "gdp": 2,
    "gross domestic product": 2,
    "unemployment rate": 2,
    "unemployment claims": 2,
    "initial jobless claims": 2,
    "retail sales": 2,
    "ism manufacturing": 2,
    "ism services": 2,
    "pce price index": 3,
    "core pce": 3,
    "ecb interest rate": 2,
    "ecb press conference": 2,
    "boe interest rate": 2,
    "aud interest rate": 1,
    "trade balance": 1,
    "adp employment": 2,
    "average hourly earnings": 2,
    "durable goods": 1,
    "michigan consumer sentiment": 1,
    "new home sales": 1,
    "existing home sales": 1,
    "crude oil inventories": 1,
}

# Currencies that affect gold price
GOLD_CURRENCIES = ["USD", "EUR", "GBP", "CHF", "JPY"]


class EconomicEvent:
    """Represents a single economic calendar event."""

    def __init__(self, name: str, currency: str, impact: str,
                 event_time: datetime, actual: str = "", forecast: str = "",
                 previous: str = "", source: str = ""):
        self.name = name
        self.currency = currency
        self.impact = impact  # "High", "Medium", "Low"
        self.event_time = event_time
        self.actual = actual
        self.forecast = forecast
        self.previous = previous
        self.source = source

        # Calculate gold impact score (0-3)
        self.gold_impact = self._calc_gold_impact()

    def _calc_gold_impact(self) -> int:
        """Calculate how much this event impacts gold specifically."""
        name_lower = self.name.lower()

        # Check against known gold-critical events
        for keyword, impact in GOLD_CRITICAL_EVENTS.items():
            if keyword in name_lower:
                return impact

        # Default based on general impact + currency
        if self.currency == "USD":
            if self.impact == "High":
                return 3
            elif self.impact == "Medium":
                return 2
            return 1
        elif self.currency in GOLD_CURRENCIES:
            if self.impact == "High":
                return 2
            return 1

        return 0

    def __str__(self):
        impact_stars = "★" * self.gold_impact + "☆" * (3 - self.gold_impact)
        return (f"[{impact_stars}] {self.event_time.strftime('%H:%M')} "
                f"{self.currency} - {self.name} ({self.impact})")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "currency": self.currency,
            "impact": self.impact,
            "gold_impact": self.gold_impact,
            "time": self.event_time.isoformat(),
            "actual": self.actual,
            "forecast": self.forecast,
            "previous": self.previous,
        }


class EconomicCalendar:
    """Fetches and manages economic calendar events."""

    def __init__(self):
        self.events: List[EconomicEvent] = []
        self.last_fetch = None

    def fetch_today_events(self) -> List[EconomicEvent]:
        """Fetch today's economic events from multiple sources."""
        events = []

        # Try free APIs in order of reliability
        events = self._fetch_from_forex_factory_api()

        if not events:
            events = self._fetch_from_mql5_api()

        if not events:
            logger.warning("  Could not fetch calendar from APIs, using built-in schedule")
            events = self._get_known_recurring_events()

        # Filter for gold-relevant events only
        gold_events = [e for e in events if e.gold_impact >= 1]

        self.events = sorted(gold_events, key=lambda e: e.event_time)
        self.last_fetch = datetime.now(timezone.utc)

        logger.info(f"  Calendar: {len(gold_events)} gold-relevant events today "
                     f"(out of {len(events)} total)")

        return self.events

    def _fetch_from_forex_factory_api(self) -> List[EconomicEvent]:
        """Fetch from free Forex Factory calendar (JBlanked API or direct)."""
        events = []
        try:
            # Forex Factory provides weekly data in XML/JSON
            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
            resp = requests.get(url, timeout=10, headers={
                "User-Agent": "XAUUSD-AI-Trader/1.0"
            })

            if resp.status_code == 200:
                data = resp.json()
                today = datetime.now(timezone.utc).date()

                for item in data:
                    try:
                        # Parse date
                        date_str = item.get("date", "")
                        if not date_str:
                            continue

                        event_time = datetime.fromisoformat(
                            date_str.replace("Z", "+00:00")
                        )

                        # Only today's events
                        if event_time.date() != today:
                            continue

                        impact_map = {
                            "High": "High",
                            "Medium": "Medium",
                            "Low": "Low",
                            "Holiday": "Low",
                        }

                        event = EconomicEvent(
                            name=item.get("title", "Unknown"),
                            currency=item.get("country", "USD"),
                            impact=impact_map.get(item.get("impact", ""), "Low"),
                            event_time=event_time,
                            actual=str(item.get("actual", "")),
                            forecast=str(item.get("forecast", "")),
                            previous=str(item.get("previous", "")),
                            source="forex_factory"
                        )
                        events.append(event)
                    except Exception:
                        continue

                logger.info(f"  Fetched {len(events)} events from Forex Factory")

        except Exception as e:
            logger.warning(f"  Forex Factory API failed: {e}")

        return events

    def _fetch_from_mql5_api(self) -> List[EconomicEvent]:
        """Fallback: try MQL5-based calendar API."""
        events = []
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            url = f"https://www.jblanked.com/news/api/mql5/calendar/today/"
            resp = requests.get(url, timeout=10, headers={
                "User-Agent": "XAUUSD-AI-Trader/1.0",
                "Content-Type": "application/json"
            })

            if resp.status_code == 200:
                data = resp.json()
                for item in data:
                    try:
                        event = EconomicEvent(
                            name=item.get("Name", item.get("Event", "Unknown")),
                            currency=item.get("Currency", "USD"),
                            impact=item.get("Impact", item.get("Importance", "Low")),
                            event_time=datetime.now(timezone.utc),
                            source="mql5"
                        )
                        events.append(event)
                    except Exception:
                        continue

                logger.info(f"  Fetched {len(events)} events from MQL5 calendar")

        except Exception as e:
            logger.warning(f"  MQL5 API failed: {e}")

        return events

    def _get_known_recurring_events(self) -> List[EconomicEvent]:
        """
        Fallback: generate events based on known schedule patterns.
        Major US economic releases follow predictable schedules.
        """
        events = []
        now = datetime.now(timezone.utc)
        today = now.date()
        weekday = today.weekday()  # 0=Monday

        # NFP: First Friday of each month
        if weekday == 4 and today.day <= 7:
            events.append(EconomicEvent(
                name="Non-Farm Payrolls",
                currency="USD",
                impact="High",
                event_time=now.replace(hour=13, minute=30),
                source="known_schedule"
            ))
            events.append(EconomicEvent(
                name="Unemployment Rate",
                currency="USD",
                impact="High",
                event_time=now.replace(hour=13, minute=30),
                source="known_schedule"
            ))

        # Weekly: Thursdays have Initial Jobless Claims
        if weekday == 3:
            events.append(EconomicEvent(
                name="Initial Jobless Claims",
                currency="USD",
                impact="Medium",
                event_time=now.replace(hour=13, minute=30),
                source="known_schedule"
            ))

        return events

    def get_upcoming_high_impact(self, within_minutes: int = 60) -> List[EconomicEvent]:
        """Get high-impact events happening within the next N minutes."""
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(minutes=within_minutes)

        upcoming = []
        for event in self.events:
            if event.gold_impact >= 2 and now <= event.event_time <= cutoff:
                upcoming.append(event)

        return upcoming

    def get_recent_high_impact(self, within_minutes: int = 30) -> List[EconomicEvent]:
        """Get high-impact events that happened in the last N minutes."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=within_minutes)

        recent = []
        for event in self.events:
            if event.gold_impact >= 2 and cutoff <= event.event_time <= now:
                recent.append(event)

        return recent

    def is_news_blackout(self, blackout_minutes: int = 30) -> tuple:
        """
        Check if we're in a news blackout period.
        Returns (is_blackout: bool, reason: str)
        """
        upcoming = self.get_upcoming_high_impact(blackout_minutes)
        recent = self.get_recent_high_impact(blackout_minutes)

        if upcoming:
            event = upcoming[0]
            mins_until = (event.event_time - datetime.now(timezone.utc)).total_seconds() / 60
            return True, f"High-impact event in {mins_until:.0f} min: {event.name}"

        if recent:
            event = recent[0]
            mins_since = (datetime.now(timezone.utc) - event.event_time).total_seconds() / 60
            return True, f"High-impact event {mins_since:.0f} min ago: {event.name}"

        return False, "No news blackout"
'''

# ============================================================
# FILE 3: src/news_filter/headline_sentiment.py
# ============================================================
FILES["src/news_filter/headline_sentiment.py"] = r'''"""
Gold Headline Sentiment Analyzer
Scores news headlines for gold market sentiment using keyword analysis.
Later can be upgraded to FinBERT or LLM API for more accuracy.
"""

import re
from datetime import datetime
from typing import List
from src.utils.logger import setup_logger

logger = setup_logger("sentiment")


# ============================================================
# Gold-specific sentiment dictionaries
# ============================================================

# Words/phrases that are BULLISH for gold (gold goes UP)
GOLD_BULLISH = {
    # Geopolitical risk = gold up (safe haven)
    "war": 0.8, "conflict": 0.7, "tension": 0.6, "crisis": 0.7,
    "sanctions": 0.5, "missile": 0.7, "invasion": 0.8, "attack": 0.6,
    "nuclear": 0.8, "escalation": 0.7, "geopolitical": 0.5,
    "middle east": 0.6, "iran": 0.4, "russia": 0.4, "north korea": 0.5,

    # Inflation = gold up (inflation hedge)
    "inflation rises": 0.7, "inflation higher": 0.7, "cpi higher": 0.6,
    "inflation surge": 0.8, "prices rise": 0.5, "cost of living": 0.5,
    "inflation persistent": 0.6, "sticky inflation": 0.6,

    # Dovish Fed = gold up (lower rates = weaker dollar)
    "rate cut": 0.8, "rate cuts": 0.8, "dovish": 0.7, "pause": 0.5,
    "fed pivot": 0.7, "easing": 0.6, "accommodation": 0.5,
    "lower rates": 0.7, "rate reduction": 0.7,

    # Dollar weakness = gold up (inverse relationship)
    "dollar falls": 0.7, "dollar weakness": 0.7, "dollar decline": 0.7,
    "usd falls": 0.6, "dollar down": 0.6, "dollar drops": 0.7,
    "dxy falls": 0.5, "dollar index falls": 0.6,

    # Economic fear = gold up (safe haven)
    "recession": 0.7, "slowdown": 0.5, "downturn": 0.6,
    "unemployment rises": 0.5, "job losses": 0.6,
    "bank failure": 0.7, "banking crisis": 0.8, "debt crisis": 0.7,
    "default": 0.6, "market crash": 0.7, "sell-off": 0.5,
    "stock market falls": 0.5, "fear": 0.4, "panic": 0.6,

    # Central bank gold buying = gold up
    "central bank gold": 0.7, "gold reserves": 0.6, "gold buying": 0.7,
    "gold demand": 0.6, "gold imports": 0.5,

    # Direct gold bullish
    "gold rally": 0.8, "gold surges": 0.8, "gold hits high": 0.7,
    "gold record": 0.8, "gold breakout": 0.7, "gold bulls": 0.6,
    "gold safe haven": 0.7, "gold demand rises": 0.7,
}

# Words/phrases that are BEARISH for gold (gold goes DOWN)
GOLD_BEARISH = {
    # Hawkish Fed = gold down (higher rates = stronger dollar)
    "rate hike": -0.8, "rate hikes": -0.8, "hawkish": -0.7,
    "tightening": -0.6, "higher rates": -0.7, "rate increase": -0.7,
    "fed raises": -0.7, "more hikes": -0.7,

    # Strong dollar = gold down
    "dollar rally": -0.7, "dollar strength": -0.7, "dollar rises": -0.7,
    "usd rises": -0.6, "dollar up": -0.6, "dollar surges": -0.7,
    "dxy rises": -0.5, "dollar index rises": -0.6,

    # Strong economy = gold down (risk-on = no need for safe haven)
    "strong jobs": -0.5, "nfp beats": -0.6, "employment strong": -0.5,
    "gdp growth": -0.5, "economy strong": -0.5, "growth accelerates": -0.5,
    "risk-on": -0.4, "stock market rally": -0.4, "equities rise": -0.4,

    # Inflation cooling = gold down (less need for hedge)
    "inflation falls": -0.6, "inflation cools": -0.6, "cpi lower": -0.6,
    "inflation eases": -0.6, "disinflation": -0.5, "prices fall": -0.4,

    # Bond yields up = gold down (opportunity cost)
    "yields rise": -0.6, "treasury yields": -0.4, "bond yields up": -0.6,
    "10-year rises": -0.5, "real yields": -0.5,

    # Direct gold bearish
    "gold falls": -0.7, "gold drops": -0.7, "gold decline": -0.6,
    "gold sells off": -0.7, "gold bears": -0.6, "gold pressure": -0.5,
    "gold slumps": -0.7, "gold sell-off": -0.7,
}


class HeadlineSentiment:
    """Scores news headlines for gold market sentiment."""

    def score_headline(self, headline: str) -> dict:
        """
        Score a single headline for gold sentiment.

        Returns:
            {
                "headline": str,
                "score": float (-1.0 to +1.0),
                "direction": "bullish" / "bearish" / "neutral",
                "matched_keywords": list,
                "confidence": float (0.0 to 1.0)
            }
        """
        headline_lower = headline.lower().strip()
        score = 0.0
        matched = []

        # Check bullish keywords
        for keyword, weight in GOLD_BULLISH.items():
            if keyword in headline_lower:
                score += weight
                matched.append((keyword, weight))

        # Check bearish keywords
        for keyword, weight in GOLD_BEARISH.items():
            if keyword in headline_lower:
                score += weight  # weight is already negative
                matched.append((keyword, weight))

        # Normalize to -1.0 to +1.0 range
        if abs(score) > 1.0:
            score = max(-1.0, min(1.0, score))

        # Determine direction
        if score > 0.2:
            direction = "bullish"
        elif score < -0.2:
            direction = "bearish"
        else:
            direction = "neutral"

        # Confidence based on number of matched keywords
        confidence = min(len(matched) * 0.25, 1.0)

        return {
            "headline": headline,
            "score": round(score, 3),
            "direction": direction,
            "matched_keywords": matched,
            "confidence": round(confidence, 2),
        }

    def score_headlines(self, headlines: List[str]) -> dict:
        """
        Score multiple headlines and return aggregate sentiment.

        Returns:
            {
                "avg_score": float,
                "direction": str,
                "confidence": float,
                "bullish_count": int,
                "bearish_count": int,
                "neutral_count": int,
                "details": list of individual scores,
            }
        """
        if not headlines:
            return {
                "avg_score": 0.0,
                "direction": "neutral",
                "confidence": 0.0,
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0,
                "details": [],
            }

        details = [self.score_headline(h) for h in headlines]

        scores = [d["score"] for d in details]
        avg_score = sum(scores) / len(scores)

        bullish = sum(1 for d in details if d["direction"] == "bullish")
        bearish = sum(1 for d in details if d["direction"] == "bearish")
        neutral = sum(1 for d in details if d["direction"] == "neutral")

        if avg_score > 0.15:
            direction = "bullish"
        elif avg_score < -0.15:
            direction = "bearish"
        else:
            direction = "neutral"

        confidence = max(d["confidence"] for d in details) if details else 0.0

        return {
            "avg_score": round(avg_score, 3),
            "direction": direction,
            "confidence": round(confidence, 2),
            "bullish_count": bullish,
            "bearish_count": bearish,
            "neutral_count": neutral,
            "details": details,
        }

    def should_block_trade(self, sentiment_result: dict, signal_direction: str) -> tuple:
        """
        Check if sentiment strongly opposes the trade direction.

        Returns:
            (should_block: bool, reason: str)
        """
        score = sentiment_result["avg_score"]
        confidence = sentiment_result["confidence"]

        # Only block if sentiment is strong AND confident
        if confidence < 0.5:
            return False, "Sentiment confidence too low to override"

        if signal_direction == "BUY" and score < -0.5:
            return True, f"Strong bearish sentiment ({score:.2f}) opposes BUY signal"

        if signal_direction == "SELL" and score > 0.5:
            return True, f"Strong bullish sentiment ({score:.2f}) opposes SELL signal"

        return False, "Sentiment aligned or neutral"
'''

# ============================================================
# FILE 4: src/news_filter/news_scraper.py
# ============================================================
FILES["src/news_filter/news_scraper.py"] = r'''"""
Gold News Headline Scraper
Fetches latest gold-related headlines for sentiment analysis.
Uses multiple free sources for reliability.
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from typing import List
from src.utils.logger import setup_logger

logger = setup_logger("news_scraper")


class GoldNewsScraper:
    """Fetches gold-related news headlines from free sources."""

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    def fetch_all_headlines(self) -> List[str]:
        """Fetch gold headlines from all available sources."""
        headlines = []

        # Try multiple sources for reliability
        headlines.extend(self._fetch_google_news_gold())

        if len(headlines) < 3:
            headlines.extend(self._fetch_kitco_headlines())

        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for h in headlines:
            h_lower = h.lower().strip()
            if h_lower not in seen and len(h) > 15:
                seen.add(h_lower)
                unique.append(h)

        logger.info(f"  Fetched {len(unique)} unique gold headlines")
        return unique[:20]  # Max 20 headlines

    def _fetch_google_news_gold(self) -> List[str]:
        """Fetch gold news from Google News RSS."""
        headlines = []
        try:
            url = "https://news.google.com/rss/search?q=gold+price+XAUUSD&hl=en-US&gl=US&ceid=US:en"
            resp = requests.get(url, headers=self.HEADERS, timeout=10)

            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "html.parser")
                items = soup.find_all("item") or soup.find_all("title")

                for item in items[:15]:
                    title = item.find("title")
                    if title:
                        text = title.get_text(strip=True)
                        if text and "google news" not in text.lower():
                            headlines.append(text)

                logger.info(f"  Google News: {len(headlines)} headlines")

        except Exception as e:
            logger.warning(f"  Google News failed: {e}")

        return headlines

    def _fetch_kitco_headlines(self) -> List[str]:
        """Fetch from Kitco (major gold news source)."""
        headlines = []
        try:
            url = "https://www.kitco.com/news/gold/"
            resp = requests.get(url, headers=self.HEADERS, timeout=10)

            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "html.parser")

                # Look for headline elements
                for tag in soup.find_all(["h2", "h3", "h4"], limit=15):
                    text = tag.get_text(strip=True)
                    if text and len(text) > 15:
                        headlines.append(text)

                logger.info(f"  Kitco: {len(headlines)} headlines")

        except Exception as e:
            logger.warning(f"  Kitco failed: {e}")

        return headlines
'''

# ============================================================
# FILE 5: src/news_filter/news_manager.py
# ============================================================
FILES["src/news_filter/news_manager.py"] = r'''"""
News Manager
Combines economic calendar + headline sentiment into a unified news filter.
This is what the trading bot queries before every trade.
"""

from datetime import datetime, timezone
from src.news_filter.economic_calendar import EconomicCalendar
from src.news_filter.headline_sentiment import HeadlineSentiment
from src.news_filter.news_scraper import GoldNewsScraper
from src.utils.config import get_settings
from src.utils.logger import setup_logger

logger = setup_logger("news_manager")


class NewsManager:
    """
    Unified news intelligence layer.
    Queries before every trade to check:
    1. Is there a news blackout? (high-impact event nearby)
    2. What is the current headline sentiment for gold?
    3. Does sentiment oppose the trade direction?
    """

    def __init__(self):
        settings = get_settings()
        self.blackout_minutes = settings.get("risk", {}).get("news_blackout_minutes", 30)

        self.calendar = EconomicCalendar()
        self.sentiment = HeadlineSentiment()
        self.scraper = GoldNewsScraper()

        # Cache
        self.last_calendar_fetch = None
        self.last_sentiment = None
        self.last_headlines_fetch = None
        self.cached_headlines = []

    def refresh(self):
        """Refresh all news data. Call this periodically (every 15 min)."""
        logger.info("  Refreshing news data...")

        # Fetch economic calendar
        self.calendar.fetch_today_events()
        self.last_calendar_fetch = datetime.now(timezone.utc)

        # Fetch headlines and score sentiment
        self.cached_headlines = self.scraper.fetch_all_headlines()
        if self.cached_headlines:
            self.last_sentiment = self.sentiment.score_headlines(self.cached_headlines)
        else:
            self.last_sentiment = {
                "avg_score": 0.0, "direction": "neutral",
                "confidence": 0.0, "bullish_count": 0,
                "bearish_count": 0, "neutral_count": 0, "details": []
            }

        self.last_headlines_fetch = datetime.now(timezone.utc)
        logger.info(f"  News refresh complete | Sentiment: {self.last_sentiment['direction']} "
                     f"({self.last_sentiment['avg_score']:.2f})")

    def can_trade(self, signal_direction: str = "BUY") -> tuple:
        """
        Master check: should we allow trading right now?

        Returns:
            (allowed: bool, reason: str, sentiment_score: float)
        """
        # Check 1: News blackout
        is_blackout, blackout_reason = self.calendar.is_news_blackout(self.blackout_minutes)
        if is_blackout:
            return False, f"NEWS BLACKOUT: {blackout_reason}", 0.0

        # Check 2: Sentiment opposition
        if self.last_sentiment and self.last_sentiment["confidence"] >= 0.5:
            blocked, block_reason = self.sentiment.should_block_trade(
                self.last_sentiment, signal_direction
            )
            if blocked:
                return False, f"SENTIMENT BLOCK: {block_reason}", self.last_sentiment["avg_score"]

        # All clear
        sentiment_score = self.last_sentiment["avg_score"] if self.last_sentiment else 0.0
        return True, "News: OK", sentiment_score

    def get_status(self) -> dict:
        """Get current news status for dashboard."""
        is_blackout, blackout_reason = self.calendar.is_news_blackout(self.blackout_minutes)

        upcoming_events = self.calendar.get_upcoming_high_impact(120)  # Next 2 hours

        return {
            "is_blackout": is_blackout,
            "blackout_reason": blackout_reason,
            "sentiment": self.last_sentiment or {"direction": "unknown", "avg_score": 0},
            "upcoming_events": [e.to_dict() for e in upcoming_events],
            "total_events_today": len(self.calendar.events),
            "headlines_count": len(self.cached_headlines),
            "last_refresh": self.last_calendar_fetch.isoformat() if self.last_calendar_fetch else None,
        }

    def print_status(self):
        """Print current news status to console."""
        status = self.get_status()

        print(f"\n  {'='*60}")
        print(f"  NEWS INTELLIGENCE STATUS")
        print(f"  {'='*60}")

        # Blackout
        if status["is_blackout"]:
            print(f"\n  ⚠  BLACKOUT ACTIVE: {status['blackout_reason']}")
        else:
            print(f"\n  ✓  No news blackout — trading allowed")

        # Sentiment
        sent = status["sentiment"]
        if sent["direction"] == "bullish":
            arrow = "↑"
        elif sent["direction"] == "bearish":
            arrow = "↓"
        else:
            arrow = "→"

        print(f"\n  Headline Sentiment: {arrow} {sent['direction'].upper()} "
              f"(score: {sent.get('avg_score', 0):.2f})")

        # Today's events
        print(f"\n  Events today: {status['total_events_today']} gold-relevant")
        if self.calendar.events:
            for event in self.calendar.events[:10]:
                print(f"    {event}")

        # Upcoming high-impact
        if status["upcoming_events"]:
            print(f"\n  ⚡ Upcoming high-impact (next 2 hours):")
            for e in status["upcoming_events"]:
                print(f"    [{e['gold_impact']}★] {e['time'][:16]} {e['currency']} - {e['name']}")
        else:
            print(f"\n  No high-impact events in next 2 hours")

        # Headlines
        if self.cached_headlines:
            print(f"\n  Latest headlines ({len(self.cached_headlines)}):")
            for h in self.cached_headlines[:5]:
                result = self.sentiment.score_headline(h)
                icon = "🟢" if result["score"] > 0.2 else ("🔴" if result["score"] < -0.2 else "⚪")
                print(f"    {icon} {h[:80]}{'...' if len(h) > 80 else ''}")

        print(f"  {'='*60}\n")
'''

# ============================================================
# FILE 6: run_news_check.py
# ============================================================
FILES["run_news_check.py"] = r'''"""
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
'''

# ============================================================
# Create all files
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("  News Sentiment Filter Setup")
    print("=" * 60 + "\n")

    for filepath, content in FILES.items():
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        print(f"  [FILE] {filepath}")

    print(f"\n  Created {len(FILES)} files.")
    print("\n  Now run:  python run_news_check.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
