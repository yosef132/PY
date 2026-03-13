"""
Telegram Alert System
Sends trade notifications and daily summaries to your phone.
"""

import requests
from datetime import datetime
from src.utils.config import get_settings
from src.utils.logger import setup_logger

logger = setup_logger("telegram")


class TelegramAlerts:
    """Sends trading alerts to Telegram."""

    def __init__(self):
        settings = get_settings()
        alerts = settings.get("alerts", {})
        self.enabled = alerts.get("telegram_enabled", False)
        self.token = alerts.get("telegram_token", "")
        self.chat_id = alerts.get("telegram_chat_id", "")
        self.base_url = f"https://api.telegram.org/bot{self.token}"

    def send(self, message: str) -> bool:
        """Send a message to Telegram."""
        if not self.enabled or not self.token or not self.chat_id:
            return False

        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML",
            }
            resp = requests.post(url, data=data, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            logger.warning(f"  Telegram send failed: {e}")
            return False

    def send_trade_opened(self, trade: dict):
        """Notify when a trade is opened."""
        direction_icon = "\U0001F7E2" if trade.get("direction") == "BUY" else "\U0001F534"
        msg = (
            f"{direction_icon} <b>TRADE OPENED</b>\n"
            f"Direction: {trade.get('direction')}\n"
            f"Strategy: {trade.get('strategy', 'unknown')}\n"
            f"Entry: ${trade.get('entry', 0):.2f}\n"
            f"SL: ${trade.get('sl', 0):.2f}\n"
            f"TP: ${trade.get('tp', 0):.2f}\n"
            f"Lot size: {trade.get('lot_size', 0)}\n"
            f"Confidence: {trade.get('confidence', 0):.0%}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        self.send(msg)

    def send_trade_closed(self, trade: dict):
        """Notify when a trade is closed."""
        pnl = trade.get("profit", 0)
        icon = "\U0001F389" if pnl > 0 else "\U0001F4A5"
        result = "WIN" if pnl > 0 else "LOSS"
        msg = (
            f"{icon} <b>TRADE CLOSED - {result}</b>\n"
            f"PnL: ${pnl:+.2f}\n"
            f"Ticket: #{trade.get('ticket', 0)}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        self.send(msg)

    def send_daily_summary(self, stats: dict):
        """Send end-of-day summary."""
        msg = (
            f"\U0001F4CA <b>DAILY SUMMARY</b>\n"
            f"Trades: {stats.get('trades_today', 0)}\n"
            f"Wins: {stats.get('wins', 0)} | Losses: {stats.get('losses', 0)}\n"
            f"PnL today: ${stats.get('daily_pnl', 0):+.2f}\n"
            f"Balance: ${stats.get('balance', 0):.2f}\n"
            f"Open positions: {stats.get('open_positions', 0)}\n"
            f"Win rate: {stats.get('win_rate', 0):.0f}%"
        )
        self.send(msg)

    def send_alert(self, message: str):
        """Send a custom alert."""
        self.send(f"\U000026A0 <b>ALERT</b>\n{message}")
