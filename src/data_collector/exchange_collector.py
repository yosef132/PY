"""
Exchange Data Collector
Fetches XAUUSDT perpetual futures data from Binance as a cross-reference signal.
NOT used for execution — only for confirming/opposing MT5 signals.

Data collected:
  - Funding rate: positive = longs paying shorts (bearish pressure)
  - Open interest change: rising OI + rising price = strong trend
  - Futures price vs spot price: premium/discount shows market bias
"""

import requests
from datetime import datetime, timezone
from src.utils.logger import setup_logger

logger = setup_logger("exchange_collector")

BINANCE_BASE = "https://fapi.binance.com"
SYMBOL = "XAUUSDT"


class ExchangeCollector:
    """Fetches gold futures data from Binance for cross-reference signals."""

    def __init__(self):
        self.last_data = {}

    def fetch_all(self) -> dict:
        """Fetch all cross-reference data. Returns dict with sentiment bias."""
        data = {}

        funding = self._fetch_funding_rate()
        if funding:
            data.update(funding)

        oi = self._fetch_open_interest()
        if oi:
            data.update(oi)

        price = self._fetch_futures_price()
        if price:
            data.update(price)

        if data:
            data["bias"] = self._calculate_bias(data)
            data["timestamp"] = datetime.now(timezone.utc).isoformat()
            self.last_data = data
            logger.info(
                f"  Binance XAUUSDT: Funding={data.get('funding_rate_pct', 0):+.4f}% "
                f"| OI=${data.get('open_interest_usd', 0)/1e6:.1f}M "
                f"| Bias={data.get('bias', 'neutral')}"
            )

        return data

    def _fetch_funding_rate(self) -> dict:
        """
        Fetch current funding rate for XAUUSDT perpetual.
        Positive funding = longs pay shorts = slight bearish pressure.
        Negative funding = shorts pay longs = slight bullish pressure.
        """
        try:
            resp = requests.get(
                f"{BINANCE_BASE}/fapi/v1/premiumIndex",
                params={"symbol": SYMBOL},
                timeout=8
            )
            if resp.status_code == 200:
                d = resp.json()
                rate = float(d.get("lastFundingRate", 0))
                mark_price = float(d.get("markPrice", 0))
                index_price = float(d.get("indexPrice", 0))

                premium_pct = ((mark_price - index_price) / index_price * 100
                               if index_price > 0 else 0)

                return {
                    "funding_rate": rate,
                    "funding_rate_pct": round(rate * 100, 4),
                    "mark_price": mark_price,
                    "index_price": index_price,
                    "futures_premium_pct": round(premium_pct, 4),
                }
        except Exception as e:
            logger.warning(f"  Binance funding rate failed: {e}")
        return {}

    def _fetch_open_interest(self) -> dict:
        """
        Fetch open interest for XAUUSDT.
        Rising OI + rising price = strong bullish trend confirmation.
        Rising OI + falling price = strong bearish trend confirmation.
        """
        try:
            resp = requests.get(
                f"{BINANCE_BASE}/fapi/v1/openInterest",
                params={"symbol": SYMBOL},
                timeout=8
            )
            if resp.status_code == 200:
                d = resp.json()
                oi = float(d.get("openInterest", 0))

                # Also get in USD value
                price_resp = requests.get(
                    f"{BINANCE_BASE}/fapi/v1/ticker/price",
                    params={"symbol": SYMBOL},
                    timeout=5
                )
                price = float(price_resp.json().get("price", 0)) if price_resp.status_code == 200 else 0
                oi_usd = oi * price

                return {
                    "open_interest": oi,
                    "open_interest_usd": oi_usd,
                    "futures_price": price,
                }
        except Exception as e:
            logger.warning(f"  Binance open interest failed: {e}")
        return {}

    def _fetch_futures_price(self) -> dict:
        """Fetch 24h price statistics."""
        try:
            resp = requests.get(
                f"{BINANCE_BASE}/fapi/v1/ticker/24hr",
                params={"symbol": SYMBOL},
                timeout=8
            )
            if resp.status_code == 200:
                d = resp.json()
                return {
                    "price_change_pct_24h": float(d.get("priceChangePercent", 0)),
                    "volume_24h": float(d.get("volume", 0)),
                    "high_24h": float(d.get("highPrice", 0)),
                    "low_24h": float(d.get("lowPrice", 0)),
                }
        except Exception as e:
            logger.warning(f"  Binance 24h ticker failed: {e}")
        return {}

    def _calculate_bias(self, data: dict) -> str:
        """
        Calculate overall market bias from exchange data.
        Returns: "bullish", "bearish", or "neutral"
        """
        score = 0

        # Funding rate signal
        # Negative funding = bullish (longs being paid, market favors shorts short-term)
        funding = data.get("funding_rate", 0)
        if funding < -0.0001:
            score += 1   # Bullish — market oversold on futures
        elif funding > 0.0003:
            score -= 1   # Bearish — longs overextended

        # Futures premium signal
        premium = data.get("futures_premium_pct", 0)
        if premium > 0.05:
            score -= 1   # Futures trading at big premium = longs extended (bearish)
        elif premium < -0.05:
            score += 1   # Futures at discount = potential squeeze (bullish)

        # 24h price change
        change_pct = data.get("price_change_pct_24h", 0)
        if change_pct > 1.5:
            score += 1
        elif change_pct < -1.5:
            score -= 1

        if score >= 1:
            return "bullish"
        elif score <= -1:
            return "bearish"
        return "neutral"

    def get_signal_boost(self, trade_direction: str) -> float:
        """
        Returns confidence boost/penalty based on exchange data alignment.
        +0.05 if exchange data confirms direction, -0.05 if it opposes.
        """
        if not self.last_data:
            return 0.0

        bias = self.last_data.get("bias", "neutral")

        if bias == "neutral":
            return 0.0
        if trade_direction == "BUY" and bias == "bullish":
            return 0.05
        if trade_direction == "SELL" and bias == "bearish":
            return 0.05
        if trade_direction == "BUY" and bias == "bearish":
            return -0.05
        if trade_direction == "SELL" and bias == "bullish":
            return -0.05

        return 0.0
