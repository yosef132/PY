"""
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
        impact_stars = "*" * self.gold_impact + "-" * (3 - self.gold_impact)
        return (f"[{impact_stars}] {self.event_time.strftime('%Y-%m-%d %H:%M')} "
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
        """Fetch economic events for the next 48 hours (covers weekend -> Monday)."""
        # Return cached data if fetched within the last 30 minutes
        if self.last_fetch is not None:
            age_minutes = (datetime.now(timezone.utc) - self.last_fetch).total_seconds() / 60
            if age_minutes < 30 and self.events:
                logger.info(f"  Calendar: using cached data ({age_minutes:.0f} min old, {len(self.events)} events)")
                return self.events

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

        logger.info(f"  Calendar: {len(gold_events)} gold-relevant events in next 48h "
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
                now = datetime.now(timezone.utc)
                window_end = now + timedelta(hours=48)

                for item in data:
                    try:
                        # Parse date
                        date_str = item.get("date", "")
                        if not date_str:
                            continue

                        event_time = datetime.fromisoformat(
                            date_str.replace("Z", "+00:00")
                        )

                        # Events in next 48 hours (covers weekend -> Monday session)
                        if not (now <= event_time <= window_end):
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
