"""
Quick test: sends a test message to your Telegram.
Run: python test_telegram.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.alerts.telegram_alerts import TelegramAlerts

def main():
    print("\n  Testing Telegram alerts...")
    t = TelegramAlerts()

    if not t.enabled:
        print("  Telegram is DISABLED in settings.yaml!")
        print("  Make sure alerts.telegram_enabled is True")
        return

    success = t.send(
        "\U00002705 <b>XAUUSD AI TRADER - TEST</b>\n\n"
        "Your trading bot is connected!\n"
        "You will receive alerts for:\n"
        "- Trade opened\n"
        "- Trade closed (with PnL)\n"
        "- Daily summaries\n"
        "- System errors\n\n"
        "\U0001F916 Bot is ready!"
    )

    if success:
        print("  SUCCESS! Check your Telegram!")
    else:
        print("  FAILED! Check your token and chat_id in settings.yaml")

if __name__ == "__main__":
    main()
