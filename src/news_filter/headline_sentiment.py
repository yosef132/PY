"""
Gold Headline Sentiment Analyzer
Primary: FinBERT (ProsusAI/finbert) — understands complex financial language.
Fallback: keyword matching — used when torch/transformers not installed.

FinBERT gives positive/negative/neutral probabilities for any financial text.
We then map those to gold-specific sentiment using known market dynamics:
  - Rate cuts / dovish Fed   -> bullish gold (lower dollar, lower opportunity cost)
  - Rate hikes / hawkish Fed -> bearish gold
  - Crisis / war / recession -> bullish gold (safe haven)
  - Strong economy / risk-on -> bearish gold
  - Direct gold news         -> follow FinBERT directly
"""

import re
from typing import List
from src.utils.logger import setup_logger

logger = setup_logger("sentiment")


# ============================================================
# FinBERT — theme-aware gold mapping
# ============================================================

# ── Theme detection for FinBERT gold mapping ────────────────────────────────
#
# Gold sentiment ≠ financial market sentiment.  We split themes into 4 buckets:
#
# 1. FEAR themes  — bad news (crisis, recession, war…)
#    FinBERT output: NEGATIVE  →  flip sign  (neg - pos) → bullish gold
#    Because: bad economic news → safe-haven demand → gold up
#
# 2. RELIEF themes — good policy news for gold (rate cuts, dovish Fed…)
#    FinBERT output: POSITIVE  →  keep sign  (pos - neg) → bullish gold
#    Because: rate cuts = weaker dollar = lower opportunity cost = gold up
#
# 3. STRENGTH themes — good economy or dollar strength
#    FinBERT output: POSITIVE  →  flip sign  (neg - pos) → bearish gold
#    Because: risk-on, strong dollar = gold not needed
#
# 4. HAWKISH themes — rate hikes, tightening
#    FinBERT output: POSITIVE  →  flip sign  (neg - pos) → bearish gold
#    Because: higher rates = stronger dollar = gold down
#
# Direct gold headlines always follow FinBERT directly.

# Priority 1 — gold is literally dropping right now
_GOLD_BEARISH_DIRECT = {
    "gold falls", "gold drops", "gold decline", "gold sells off",
    "gold slumps", "gold sell-off", "gold bears", "gold pressure",
    "dump gold", "sold gold", "selling gold", "exit gold",
}

# Priority 2 — direct gold reference (gold itself mentioned)
_GOLD_DIRECT = {"gold", "xauusd", "xau/usd", "xau usd", "bullion", "gold price"}

# Priority 3 — bearish gold drivers (these override fear signals)
# FinBERT positive on these → risk-on → bearish gold  (use neg - pos to flip)
_HAWKISH_THEMES = {
    "rate hike", "rate hikes", "hawkish", "tightening", "higher rates",
    "rate increase", "fed raises", "more hikes", "aggressive hikes",
}
_STRENGTH_THEMES = {
    "strong jobs", "nfp beats", "employment strong", "gdp growth",
    "economy strong", "growth accelerates", "risk-on", "equities rise",
    "stock market rally", "risk appetite",
    "inflation falls", "inflation cools", "cpi lower", "disinflation",
    "inflation eases", "prices fall",
    "dollar rally", "dollar strength", "dollar rises", "dollar surges",
    "usd rises", "dollar up", "dxy rises", "dollar index rises",
    "yields rise", "yields surge", "yields jump", "yields climb",
    "bond yields up", "10-year rises", "real yields up",
}

# Priority 4 — fear / crisis (safe-haven buying → gold up)
# FinBERT negative on these → crisis worsening → bullish gold  (use neg - pos)
_FEAR_THEMES = {
    "war", "conflict", "tension", "invasion", "attack", "nuclear", "missile",
    "sanctions", "geopolitical", "escalation", "middle east", "iran", "russia",
    "recession", "slowdown", "downturn", "contraction", "depression",
    "bank failure", "banking crisis", "debt crisis", "default",
    "market crash", "sell-off", "stock market falls", "fear", "panic",
    "safe haven",                      # "flee to safe havens" = fear context
    "inflation rises", "inflation surge", "inflation higher", "cpi higher",
    "prices rise", "sticky inflation", "inflation persistent",
    "dollar falls", "dollar weakness", "dollar decline", "dollar drops",
    "usd falls", "dollar down", "dxy falls",
    "unemployment rises", "job losses",
}

# Priority 5 — dovish / relief for gold
# FinBERT positive on these → rate cuts / gold demand up → bullish gold  (use pos - neg)
_RELIEF_THEMES = {
    "rate cut", "rate cuts", "dovish", "fed pivot", "fed pause", "easing",
    "lower rates", "rate reduction", "accommodation",
    "gold rally", "gold surges", "gold demand", "gold buying", "gold record",
    "central bank gold", "gold reserves", "gold breakout", "gold bulls",
}


class FinBERTScorer:
    """
    Lazy-loading FinBERT sentiment scorer.

    The pipeline is loaded once (class-level cache) on first call.
    If torch or transformers are not installed, is_available() returns False
    and all score() calls return None — HeadlineSentiment then falls back
    to keyword matching automatically.

    Install: pip install torch transformers
    """

    MODEL_NAME = "ProsusAI/finbert"
    _pipeline = None       # shared across all instances
    _available = None      # None=unchecked, True/False after first check

    @classmethod
    def is_available(cls) -> bool:
        """Check (once) whether torch + transformers are installed."""
        if cls._available is None:
            try:
                import torch          # noqa: F401
                import transformers   # noqa: F401
                cls._available = True
            except ImportError:
                cls._available = False
                logger.info(
                    "  FinBERT not available (torch/transformers not installed). "
                    "Using keyword fallback. To enable: pip install torch transformers"
                )
        return cls._available

    @classmethod
    def get_pipeline(cls):
        """Load and cache the FinBERT pipeline (first call takes ~30s)."""
        if cls._pipeline is None and cls.is_available():
            try:
                from transformers import pipeline as hf_pipeline
                logger.info(f"  Loading FinBERT ({cls.MODEL_NAME}) — first load may take ~30s...")
                cls._pipeline = hf_pipeline(
                    "text-classification",
                    model=cls.MODEL_NAME,
                    top_k=None,        # return all label scores (replaces deprecated return_all_scores)
                    truncation=True,
                    max_length=512,
                )
                logger.info("  FinBERT model loaded OK")
            except Exception as e:
                logger.warning(f"  FinBERT load failed: {e}. Falling back to keywords.")
                cls._available = False
        return cls._pipeline

    def score_raw(self, text: str) -> dict | None:
        """
        Run FinBERT on text.
        Returns {positive, negative, neutral} probabilities or None on failure.

        Handles both transformers output formats:
          - top_k=None  → [[{label, score}, ...]]  (transformers 4.x / 5.x)
          - legacy       → [{label, score}]          (older versions)
        """
        pipe = self.get_pipeline()
        if pipe is None:
            return None
        try:
            raw = pipe(text[:512])
            # Unwrap: [[{...}]] → [{...}]  or  [{...}] → [{...}]
            if raw and isinstance(raw[0], list):
                results = raw[0]
            else:
                results = raw
            return {r["label"].lower(): r["score"] for r in results}
        except Exception as e:
            logger.warning(f"  FinBERT inference error: {e}")
            return None

    def map_to_gold_score(self, headline: str, probs: dict) -> float:
        """
        Convert FinBERT probabilities to a gold-specific sentiment score (-1 to +1).

        Uses four theme buckets (see module-level comments):
          FEAR     → neg - pos  (bad news = safe-haven buying = gold up)
          RELIEF   → pos - neg  (good policy news for gold = gold up)
          STRENGTH → neg - pos  (good economy = risk-on = gold down; flip so positive = bearish)
          HAWKISH  → neg - pos  (rate hikes = dollar up = gold down; flip so positive = bearish)
        Direct gold headlines always follow FinBERT directly.
        """
        h = headline.lower()
        pos = probs.get("positive", 0.33)
        neg = probs.get("negative", 0.33)

        # ── Priority 1: gold literally dropping ──────────────────────────────
        if any(kw in h for kw in _GOLD_BEARISH_DIRECT):
            # neg FinBERT (gold falling is bad) → pos - neg < 0 → bearish gold ✓
            return round(max(-1.0, min(1.0, pos - neg)), 3)

        # ── Priority 2: direct gold mention + FinBERT knows it's good news ──
        is_gold_direct = any(kw in h for kw in _GOLD_DIRECT)
        if is_gold_direct and pos > neg:
            return round(max(-1.0, min(1.0, pos - neg)), 3)

        # ── Priority 3: hawkish / strong economy → bearish gold ─────────────
        # FinBERT positive on these (good news) → flip → bearish gold
        if any(kw in h for kw in _HAWKISH_THEMES) or any(kw in h for kw in _STRENGTH_THEMES):
            return round(max(-1.0, min(1.0, neg - pos)), 3)

        # ── Priority 4: fear / crisis → bullish gold (safe-haven) ───────────
        # FinBERT negative (bad news) → neg - pos > 0 → bullish gold ✓
        if any(kw in h for kw in _FEAR_THEMES):
            return round(max(-1.0, min(1.0, neg - pos)), 3)

        # ── Priority 5: dovish / rate-cut relief → bullish gold ─────────────
        # FinBERT positive (rate cuts = good for markets) → pos - neg > 0 → bullish gold ✓
        if any(kw in h for kw in _RELIEF_THEMES):
            return round(max(-1.0, min(1.0, pos - neg)), 3)

        # ── Priority 6: no clear theme — gold safe-haven default ─────────────
        # Negative FinBERT = market fear = slight gold support
        return round(max(-1.0, min(1.0, neg * 0.2 - pos * 0.2)), 3)


# ============================================================
# Keyword fallback (gold-specific dictionaries)
# ============================================================

GOLD_BULLISH = {
    "war": 0.8, "conflict": 0.7, "tension": 0.6, "crisis": 0.7,
    "sanctions": 0.5, "missile": 0.7, "invasion": 0.8, "attack": 0.6,
    "nuclear": 0.8, "escalation": 0.7, "geopolitical": 0.5,
    "middle east": 0.6, "iran": 0.4, "russia": 0.4, "north korea": 0.5,
    "inflation rises": 0.7, "inflation higher": 0.7, "cpi higher": 0.6,
    "inflation surge": 0.8, "prices rise": 0.5, "cost of living": 0.5,
    "inflation persistent": 0.6, "sticky inflation": 0.6,
    "rate cut": 0.8, "rate cuts": 0.8, "dovish": 0.7, "pause": 0.5,
    "fed pivot": 0.7, "easing": 0.6, "accommodation": 0.5,
    "lower rates": 0.7, "rate reduction": 0.7,
    "dollar falls": 0.7, "dollar weakness": 0.7, "dollar decline": 0.7,
    "usd falls": 0.6, "dollar down": 0.6, "dollar drops": 0.7,
    "dxy falls": 0.5, "dollar index falls": 0.6,
    "recession": 0.7, "slowdown": 0.5, "downturn": 0.6,
    "unemployment rises": 0.5, "job losses": 0.6,
    "bank failure": 0.7, "banking crisis": 0.8, "debt crisis": 0.7,
    "default": 0.6, "market crash": 0.7, "sell-off": 0.5,
    "stock market falls": 0.5, "fear": 0.4, "panic": 0.6,
    "central bank gold": 0.7, "gold reserves": 0.6, "gold buying": 0.7,
    "gold demand": 0.6, "gold imports": 0.5,
    "gold rally": 0.8, "gold surges": 0.8, "gold hits high": 0.7,
    "gold record": 0.8, "gold breakout": 0.7, "gold bulls": 0.6,
    "gold safe haven": 0.7, "gold demand rises": 0.7,
}

GOLD_BEARISH = {
    "rate hike": -0.8, "rate hikes": -0.8, "hawkish": -0.7,
    "tightening": -0.6, "higher rates": -0.7, "rate increase": -0.7,
    "fed raises": -0.7, "more hikes": -0.7,
    "dollar rally": -0.7, "dollar strength": -0.7, "dollar rises": -0.7,
    "usd rises": -0.6, "dollar up": -0.6, "dollar surges": -0.7,
    "dxy rises": -0.5, "dollar index rises": -0.6,
    "strong jobs": -0.5, "nfp beats": -0.6, "employment strong": -0.5,
    "gdp growth": -0.5, "economy strong": -0.5, "growth accelerates": -0.5,
    "risk-on": -0.4, "stock market rally": -0.4, "equities rise": -0.4,
    "inflation falls": -0.6, "inflation cools": -0.6, "cpi lower": -0.6,
    "inflation eases": -0.6, "disinflation": -0.5, "prices fall": -0.4,
    "yields rise": -0.6, "treasury yields": -0.4, "bond yields up": -0.6,
    "10-year rises": -0.5, "real yields": -0.5,
    "gold falls": -0.7, "gold drops": -0.7, "gold decline": -0.6,
    "gold sells off": -0.7, "gold bears": -0.6, "gold pressure": -0.5,
    "gold slumps": -0.7, "gold sell-off": -0.7,
}


# ============================================================
# Main class — same interface as before
# ============================================================

class HeadlineSentiment:
    """
    Gold headline sentiment scorer.

    Uses FinBERT when available (torch + transformers installed),
    falls back to keyword matching otherwise. The public interface
    (score_headline / score_headlines / should_block_trade) is
    identical in both modes so NewsManager requires no changes.
    """

    def __init__(self):
        self._finbert = FinBERTScorer()
        self._use_finbert = FinBERTScorer.is_available()
        mode = "FinBERT" if self._use_finbert else "keyword fallback"
        logger.info(f"  HeadlineSentiment: using {mode}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_headline(self, headline: str) -> dict:
        """
        Score a single headline for gold sentiment.

        Returns:
            {
                "headline": str,
                "score": float (-1.0 to +1.0),
                "direction": "bullish" / "bearish" / "neutral",
                "method": "finbert" or "keyword",
                "confidence": float (0.0 to 1.0),
                "matched_keywords": list,   # non-empty only in keyword mode
            }
        """
        if self._use_finbert:
            return self._score_with_finbert(headline)
        return self._score_with_keywords(headline)

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
                "method": str,
                "details": list,
            }
        """
        if not headlines:
            return {
                "avg_score": 0.0, "direction": "neutral", "confidence": 0.0,
                "bullish_count": 0, "bearish_count": 0, "neutral_count": 0,
                "method": "none", "details": [],
            }

        details = [self.score_headline(h) for h in headlines]
        scores = [d["score"] for d in details]
        avg_score = sum(scores) / len(scores)

        bullish = sum(1 for d in details if d["direction"] == "bullish")
        bearish = sum(1 for d in details if d["direction"] == "bearish")
        neutral = sum(1 for d in details if d["direction"] == "neutral")

        direction = "neutral"
        if avg_score > 0.15:
            direction = "bullish"
        elif avg_score < -0.15:
            direction = "bearish"

        # Confidence: higher when headlines agree with each other
        if len(scores) > 1:
            agreement = max(bullish, bearish) / len(scores)   # 0.5–1.0
            avg_individual_conf = sum(d["confidence"] for d in details) / len(details)
            confidence = round(agreement * avg_individual_conf, 2)
        else:
            confidence = details[0]["confidence"] if details else 0.0

        method = details[0].get("method", "keyword") if details else "keyword"

        return {
            "avg_score": round(avg_score, 3),
            "direction": direction,
            "confidence": confidence,
            "bullish_count": bullish,
            "bearish_count": bearish,
            "neutral_count": neutral,
            "method": method,
            "details": details,
        }

    def should_block_trade(self, sentiment_result: dict, signal_direction: str) -> tuple:
        """
        Check if sentiment strongly opposes the trade direction.

        Returns:
            (should_block: bool, reason: str)
        """
        score = sentiment_result.get("avg_score", 0.0)
        confidence = sentiment_result.get("confidence", 0.0)

        # Only block when sentiment is both strong AND confident
        threshold = 0.4 if self._use_finbert else 0.5   # FinBERT is more precise

        if confidence < threshold:
            return False, "Sentiment confidence too low to override"

        if signal_direction == "BUY" and score < -0.5:
            return True, f"Strong bearish sentiment ({score:.2f}) opposes BUY signal"

        if signal_direction == "SELL" and score > 0.5:
            return True, f"Strong bullish sentiment ({score:.2f}) opposes SELL signal"

        return False, "Sentiment aligned or neutral"

    # ------------------------------------------------------------------
    # Internal scorers
    # ------------------------------------------------------------------

    def _score_with_finbert(self, headline: str) -> dict:
        """Score using FinBERT + gold theme mapping."""
        probs = self._finbert.score_raw(headline)

        if probs is None:
            # FinBERT failed mid-session — fall back to keywords
            return self._score_with_keywords(headline)

        score = self._finbert.map_to_gold_score(headline, probs)

        # Confidence = how far from neutral FinBERT's max probability is
        max_prob = max(probs.values())
        confidence = round(max(0.0, (max_prob - 0.33) / 0.67), 2)

        if score > 0.15:
            direction = "bullish"
        elif score < -0.15:
            direction = "bearish"
        else:
            direction = "neutral"

        return {
            "headline": headline,
            "score": score,
            "direction": direction,
            "method": "finbert",
            "confidence": confidence,
            "matched_keywords": [],   # not applicable in FinBERT mode
            "finbert_probs": probs,
        }

    def _score_with_keywords(self, headline: str) -> dict:
        """Original keyword-based scoring (fallback)."""
        headline_lower = headline.lower().strip()
        score = 0.0
        matched = []

        for keyword, weight in GOLD_BULLISH.items():
            if keyword in headline_lower:
                score += weight
                matched.append((keyword, weight))

        for keyword, weight in GOLD_BEARISH.items():
            if keyword in headline_lower:
                score += weight
                matched.append((keyword, weight))

        score = max(-1.0, min(1.0, score))

        if score > 0.2:
            direction = "bullish"
        elif score < -0.2:
            direction = "bearish"
        else:
            direction = "neutral"

        confidence = min(len(matched) * 0.25, 1.0)

        return {
            "headline": headline,
            "score": round(score, 3),
            "direction": direction,
            "method": "keyword",
            "confidence": round(confidence, 2),
            "matched_keywords": matched,
        }
