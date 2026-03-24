"""
Configuration loader for the XAUUSD AI Trading System.
Credentials are loaded from .env (never hardcoded in YAML).
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

# Load .env file once at import time
load_dotenv(PROJECT_ROOT / ".env")


def load_yaml(filename: str) -> dict:
    filepath = CONFIG_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_settings() -> dict:
    settings = load_yaml("settings.yaml")

    # Inject credentials from environment variables (override YAML values)
    settings.setdefault("mt5", {})
    if os.getenv("MT5_LOGIN"):
        try:
            settings["mt5"]["login"] = int(os.getenv("MT5_LOGIN"))
        except ValueError:
            raise ValueError(f"MT5_LOGIN in .env must be a number, got: '{os.getenv('MT5_LOGIN')}'")
    if os.getenv("MT5_PASSWORD"):
        settings["mt5"]["password"] = os.getenv("MT5_PASSWORD")
    if os.getenv("MT5_SERVER"):
        settings["mt5"]["server"] = os.getenv("MT5_SERVER")

    settings.setdefault("alerts", {})
    if os.getenv("TELEGRAM_TOKEN"):
        settings["alerts"]["telegram_token"] = os.getenv("TELEGRAM_TOKEN")
    if os.getenv("TELEGRAM_CHAT_ID"):
        settings["alerts"]["telegram_chat_id"] = os.getenv("TELEGRAM_CHAT_ID")

    return settings


def get_strategies() -> dict:
    return load_yaml("strategies.yaml")


MT5_TIMEFRAMES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 16385,
    "H4": 16388,
    "D1": 16408,
    "W1": 32769,
}


def get_mt5_timeframe(name: str) -> int:
    if name not in MT5_TIMEFRAMES:
        raise ValueError(f"Unknown timeframe: {name}. Valid: {list(MT5_TIMEFRAMES.keys())}")
    return MT5_TIMEFRAMES[name]


TF_ORDER = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]
