"""
Multi-Timeframe (MTF) Confirmation Filter
The #1 accuracy improvement: only trade H1 signals that align with H4 trend direction.

How it works:
- H4 determines the TREND DIRECTION (are we bullish or bearish overall?)
- H1 finds the ENTRY TIMING (where exactly to enter)
- If H1 signal disagrees with H4 trend → BLOCKED
- If H1 signal agrees with H4 trend → BOOSTED confidence

This alone can improve win rate by 10-20% because you stop fighting the bigger trend.
"""

import pandas as pd
import numpy as np
from src.feature_engine.engine import FeatureEngine
from src.utils.logger import setup_logger

logger = setup_logger("mtf_filter")


class MTFFilter:
    """
    Multi-Timeframe trend filter.
    Uses H4 (or D1) to determine trend, then only allows H1 entries in that direction.
    """

    def __init__(self):
        self.w1_trend = 0       # +1 = bullish, -1 = bearish, 0 = neutral (weekly macro)
        self.h4_trend = 0       # +1 = bullish, -1 = bearish, 0 = neutral
        self.h4_strength = 0.0  # 0.0 to 1.0
        self.h4_data = {}       # Cached H4 analysis
        self.d1_trend = 0       # Daily trend for extra confirmation

    def analyze_higher_timeframes(self, featured_data: dict):
        """
        Analyze W1, H4 and D1 data to determine the higher timeframe trend.
        W1 = macro context, H4 = medium-term bias, D1 = daily confirmation.
        """
        # Analyze W1 (macro trend — overrides H4 hard block when disagreement = pullback)
        if "W1" in featured_data:
            self.w1_trend, _, _ = self._analyze_trend(featured_data["W1"], "W1")
            logger.info(f"  W1 Trend: {'BULLISH' if self.w1_trend > 0 else 'BEARISH' if self.w1_trend < 0 else 'NEUTRAL'}")

        # Analyze H4
        if "H4" in featured_data:
            h4_df = featured_data["H4"]
            self.h4_trend, self.h4_strength, self.h4_data = self._analyze_trend(h4_df, "H4")
            logger.info(f"  H4 Trend: {'BULLISH' if self.h4_trend > 0 else 'BEARISH' if self.h4_trend < 0 else 'NEUTRAL'} "
                        f"(strength: {self.h4_strength:.2f})")

        # Analyze D1
        if "D1" in featured_data:
            d1_df = featured_data["D1"]
            self.d1_trend, _, _ = self._analyze_trend(d1_df, "D1")
            logger.info(f"  D1 Trend: {'BULLISH' if self.d1_trend > 0 else 'BEARISH' if self.d1_trend < 0 else 'NEUTRAL'}")

    def _analyze_trend(self, df: pd.DataFrame, tf_name: str) -> tuple:
        """
        Determine trend direction and strength from a featured DataFrame.

        Returns:
            (trend: int, strength: float, data: dict)
        """
        if df.empty or len(df) < 50:
            return 0, 0.0, {}

        last = df.iloc[-1]
        prev_5 = df.iloc[-5:]

        trend_score = 0
        signals = {}

        # --- Signal 1: EMA Alignment ---
        ema_align = last.get("ema_alignment", 0)
        if ema_align >= 2:
            trend_score += 2
            signals["ema"] = "bullish"
        elif ema_align <= -2:
            trend_score -= 2
            signals["ema"] = "bearish"
        else:
            signals["ema"] = "neutral"

        # --- Signal 2: Price vs EMA 50 ---
        ema_50 = last.get("ema_50", None)
        if pd.notna(ema_50):
            if last["close"] > ema_50 * 1.002:
                trend_score += 1
                signals["price_ema50"] = "above"
            elif last["close"] < ema_50 * 0.998:
                trend_score -= 1
                signals["price_ema50"] = "below"

        # --- Signal 3: Price vs EMA 200 ---
        ema_200 = last.get("ema_200", None)
        if pd.notna(ema_200):
            if last["close"] > ema_200:
                trend_score += 1
                signals["price_ema200"] = "above"
            elif last["close"] < ema_200:
                trend_score -= 1
                signals["price_ema200"] = "below"

        # --- Signal 4: Market structure trend ---
        market_trend = last.get("market_trend", 0)
        if market_trend > 0:
            trend_score += 1
            signals["structure"] = "bullish"
        elif market_trend < 0:
            trend_score -= 1
            signals["structure"] = "bearish"

        # --- Signal 5: ADX trend strength ---
        adx_cols = [c for c in df.columns if "ADX_" in c]
        if adx_cols:
            adx = last.get(adx_cols[0], 0)
            if pd.notna(adx) and adx > 25:
                signals["adx"] = f"trending ({adx:.0f})"
            else:
                signals["adx"] = f"ranging ({adx:.0f})" if pd.notna(adx) else "unknown"

        # --- Signal 6: RSI position ---
        rsi = last.get("rsi_14", 50)
        if pd.notna(rsi):
            if rsi > 55:
                trend_score += 0.5
                signals["rsi"] = f"bullish ({rsi:.0f})"
            elif rsi < 45:
                trend_score -= 0.5
                signals["rsi"] = f"bearish ({rsi:.0f})"

        # --- Signal 7: MACD direction ---
        macd_cols = [c for c in df.columns if "MACDh_" in c]
        if macd_cols:
            macd_h = last.get(macd_cols[0], 0)
            prev_macd = df.iloc[-2].get(macd_cols[0], 0) if len(df) > 1 else 0
            if pd.notna(macd_h) and pd.notna(prev_macd):
                if macd_h > 0 and macd_h > prev_macd:
                    trend_score += 1
                    signals["macd"] = "bullish momentum"
                elif macd_h < 0 and macd_h < prev_macd:
                    trend_score -= 1
                    signals["macd"] = "bearish momentum"

        # --- Signal 8: Recent candle direction ---
        bullish_candles = sum(1 for _, r in prev_5.iterrows() if r["close"] > r["open"])
        if bullish_candles >= 4:
            trend_score += 1
            signals["candles"] = f"{bullish_candles}/5 bullish"
        elif bullish_candles <= 1:
            trend_score -= 1
            signals["candles"] = f"{5 - bullish_candles}/5 bearish"

        # Determine final trend
        if trend_score >= 3:
            trend = 1
        elif trend_score <= -3:
            trend = -1
        else:
            trend = 0

        # Strength (normalize to 0-1)
        strength = min(abs(trend_score) / 8.0, 1.0)

        data = {
            "trend_score": round(trend_score, 1),
            "strength": round(strength, 2),
            "signals": signals,
        }

        return trend, strength, data

    def filter_signal(self, signal) -> tuple:
        """
        3-tier filter: W1 (macro) → H4 (medium) → entry allowed/blocked.

        W1 available:
          W1 + H4 both agree with signal  → strong allow, max boost (+12%)
          W1 agrees, H4 neutral            → allow, medium boost (+7%)
          W1 agrees, H4 against (pullback) → allow as pullback entry (+5%)
          W1 against, H4 agrees            → allow, no boost (risky counter-macro)
          W1 against, H4 neutral           → allow, no boost
          W1 + H4 both against             → block

        W1 neutral → fall back to original H4+D1 logic.
        """
        direction = signal.direction
        is_buy = direction == "BUY"

        # ── W1 macro context available ──────────────────────────────────
        if self.w1_trend != 0:
            w1_with = (is_buy and self.w1_trend > 0) or (not is_buy and self.w1_trend < 0)
            h4_with = (is_buy and self.h4_trend > 0) or (not is_buy and self.h4_trend < 0)
            h4_neutral = self.h4_trend == 0

            if w1_with and h4_with:
                # All timeframes aligned — strongest signal
                boost = 0.12 + self.h4_strength * 0.05
                if self.d1_trend != 0 and ((is_buy and self.d1_trend > 0) or (not is_buy and self.d1_trend < 0)):
                    boost += 0.03  # D1 also agrees
                return True, f"W1+H4 aligned with {direction} (+{boost:.0%})", boost

            elif w1_with and h4_neutral:
                return True, f"W1 aligned with {direction}, H4 neutral (+7%)", 0.07

            elif w1_with and not h4_with:
                # W1 agrees but H4 disagrees = pullback in macro trend (high-probability setup)
                w1_label = "BULLISH" if self.w1_trend > 0 else "BEARISH"
                return True, f"Pullback entry: W1={w1_label}, H4 short-term against (+5%)", 0.05

            elif not w1_with and (h4_neutral or h4_with):
                # Counter-macro trade — allow but no boost
                w1_label = "BULLISH" if self.w1_trend > 0 else "BEARISH"
                return True, f"Counter W1 macro ({w1_label}), H4 local align — no boost", 0.0

            else:
                # Both W1 and H4 against the signal — block
                w1_label = "BULLISH" if self.w1_trend > 0 else "BEARISH"
                h4_label = "BULLISH" if self.h4_trend > 0 else "BEARISH"
                return False, f"{direction} blocked: W1={w1_label}, H4={h4_label} both against", 0

        # ── W1 neutral: fall back to original H4 + D1 logic ────────────
        if self.h4_trend != 0:
            if direction == "BUY" and self.h4_trend < 0:
                return False, f"BUY blocked: H4 BEARISH (score: {self.h4_data.get('trend_score', 0)})", 0
            if direction == "SELL" and self.h4_trend > 0:
                return False, f"SELL blocked: H4 BULLISH (score: {self.h4_data.get('trend_score', 0)})", 0

        confidence_boost = 0.0
        if self.d1_trend != 0:
            if (is_buy and self.d1_trend > 0) or (not is_buy and self.d1_trend < 0):
                confidence_boost += 0.05

        if self.h4_trend != 0:
            if (is_buy and self.h4_trend > 0) or (not is_buy and self.h4_trend < 0):
                confidence_boost += self.h4_strength * 0.10

        if self.h4_trend == 0:
            return True, "H4 neutral — allowed without boost", 0

        reason = f"H4 {'BULLISH' if self.h4_trend > 0 else 'BEARISH'} aligns with {direction} (+{confidence_boost:.1%})"
        return True, reason, confidence_boost

    def get_status(self) -> dict:
        """Get current MTF analysis status."""
        return {
            "w1_trend": "BULLISH" if self.w1_trend > 0 else "BEARISH" if self.w1_trend < 0 else "NEUTRAL",
            "h4_trend": "BULLISH" if self.h4_trend > 0 else "BEARISH" if self.h4_trend < 0 else "NEUTRAL",
            "h4_strength": self.h4_strength,
            "h4_details": self.h4_data,
            "d1_trend": "BULLISH" if self.d1_trend > 0 else "BEARISH" if self.d1_trend < 0 else "NEUTRAL",
        }
