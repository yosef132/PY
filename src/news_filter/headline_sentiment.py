"""
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
