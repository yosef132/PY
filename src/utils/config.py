"""
Configuration loader for the XAUUSD AI Trading System.
"""

import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def load_yaml(filename: str) -> dict:
    filepath = CONFIG_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_settings() -> dict:
    return load_yaml("settings.yaml")


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
