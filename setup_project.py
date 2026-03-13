"""
XAUUSD AI Trading System - Project Setup Script
Run this once to create the entire project structure.

Usage:
    python setup_project.py
"""

import os

# ============================================================
# Define all project files
# ============================================================

FILES = {}

# --- config/settings.yaml ---
FILES["config/settings.yaml"] = r"""# ============================================================
# XAUUSD AI Trading System - Main Configuration
# ============================================================

# --- MetaTrader 5 Connection ---
mt5:
  login: null
  password: null
  server: null
  terminal_path: null

# --- Trading Symbol ---
symbol: "XAUUSD"

# --- Timeframes to collect ---
timeframes:
  - M5
  - M15
  - H1
  - H4
  - D1

# --- Data Collection ---
data:
  history_bars: 5000
  storage_format: "parquet"
  data_dir: "data/historical"

# --- Risk Management ---
risk:
  risk_per_trade_pct: 0.5
  max_daily_loss_pct: 3.0
  max_open_positions: 2
  min_risk_reward: 2.0
  max_drawdown_pct: 10.0
  sl_atr_multiplier: 1.5
  tp_atr_multiplier: 3.0
  trailing_stop_atr: 0.5
  news_blackout_minutes: 30

# --- Session Filters ---
sessions:
  london_start: "07:00"
  london_end: "12:00"
  newyork_start: "12:00"
  newyork_end: "17:00"
  trade_overlap_only: false

# --- ML Model ---
ml:
  model_type: "xgboost"
  confidence_threshold: 0.65
  retrain_day: "Sunday"
  lookback_months: 6
  min_trades_to_train: 100

# --- Execution ---
execution:
  mode: "demo"
  lot_size: 0.01
  slippage_points: 30
  magic_number: 123456

# --- Logging ---
logging:
  level: "INFO"
  log_dir: "logs"
  trade_log_db: "data/trades.db"

# --- Alerts (Phase 3+) ---
alerts:
  telegram_enabled: false
  telegram_token: null
  telegram_chat_id: null
  email_enabled: false
"""

# --- config/strategies.yaml ---
FILES["config/strategies.yaml"] = r"""# ============================================================
# Strategy Configuration
# ============================================================

strategies:
  support_resistance:
    enabled: true
    lookback: 50
    min_touches: 2
    zone_atr_mult: 0.3

  rsi_divergence:
    enabled: true
    rsi_period: 14
    min_divergence: 5
    signal_tf: "H1"
    entry_tf: "M15"

  ict:
    enabled: true
    ob_lookback: 20
    fvg_min_points: 50
    killzone_only: true

  wyckoff:
    enabled: true
    range_lookback: 100
    volume_confirm: true
    min_range_candles: 30

  crt:
    enabled: true
    session_candle_tf: "H4"

  sk_system:
    enabled: false
"""

# --- src/__init__.py ---
FILES["src/__init__.py"] = ""

# --- src/utils/__init__.py ---
FILES["src/utils/__init__.py"] = ""

# --- src/utils/config.py ---
FILES["src/utils/config.py"] = r'''"""
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
'''

# --- src/utils/logger.py ---
FILES["src/utils/logger.py"] = r'''"""
Logging setup for the XAUUSD AI Trading System.
"""

import logging
from datetime import datetime
from pathlib import Path

from src.utils.config import PROJECT_ROOT


def setup_logger(name: str = "xauusd_ai", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    file_handler = logging.FileHandler(
        log_dir / f"{today}.log", encoding="utf-8"
    )
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger
'''

# --- src/data_collector/__init__.py ---
FILES["src/data_collector/__init__.py"] = ""

# --- src/data_collector/mt5_collector.py ---
FILES["src/data_collector/mt5_collector.py"] = r'''"""
Module 1: Data Collector
Connects to MetaTrader 5 and collects XAUUSD price data across multiple timeframes.
Stores data in Parquet format for fast reading during backtesting and ML training.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from src.utils.config import get_settings, get_mt5_timeframe, PROJECT_ROOT
from src.utils.logger import setup_logger

logger = setup_logger("data_collector")


class MT5Connector:
    """Handles connection to MetaTrader 5 terminal."""

    def __init__(self):
        self.settings = get_settings()
        self.connected = False

    def connect(self) -> bool:
        mt5_config = self.settings.get("mt5", {})

        kwargs = {}
        if mt5_config.get("login"):
            kwargs["login"] = mt5_config["login"]
        if mt5_config.get("password"):
            kwargs["password"] = mt5_config["password"]
        if mt5_config.get("server"):
            kwargs["server"] = mt5_config["server"]
        if mt5_config.get("terminal_path"):
            kwargs["path"] = mt5_config["terminal_path"]

        if not mt5.initialize(**kwargs):
            error = mt5.last_error()
            logger.error(f"MT5 initialization failed: {error}")
            return False

        account = mt5.account_info()
        version = mt5.version()
        logger.info(f"Connected to MT5 - Build: {version[1]} ({version[2]})")
        logger.info(f"Account: {account.login} | Server: {account.server}")
        logger.info(f"Balance: ${account.balance:.2f} | Equity: ${account.equity:.2f}")
        logger.info(f"Account type: {'Demo' if account.trade_mode == 0 else 'Real'}")

        self.connected = True
        return True

    def disconnect(self):
        mt5.shutdown()
        self.connected = False
        logger.info("Disconnected from MT5")

    def get_account_info(self) -> dict:
        info = mt5.account_info()
        if info is None:
            return {}
        return {
            "login": info.login,
            "server": info.server,
            "balance": info.balance,
            "equity": info.equity,
            "margin_free": info.margin_free,
            "profit": info.profit,
            "trade_mode": "Demo" if info.trade_mode == 0 else "Real",
            "leverage": info.leverage,
            "currency": info.currency,
        }


class DataCollector:
    """
    Collects XAUUSD price data from MT5 across multiple timeframes.
    Stores historical data in Parquet files for fast access.
    """

    def __init__(self, connector: MT5Connector):
        self.connector = connector
        self.settings = get_settings()
        self.symbol = self.settings["symbol"]
        self.data_dir = PROJECT_ROOT / self.settings["data"]["data_dir"]
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def check_symbol(self) -> bool:
        info = mt5.symbol_info(self.symbol)
        if info is None:
            logger.error(f"Symbol {self.symbol} not found! Check your broker.")
            alternatives = ["XAUUSD", "GOLD", "XAUUSDm", "XAUUSD.a", "XAUUSD.raw"]
            for alt in alternatives:
                if mt5.symbol_info(alt) is not None:
                    logger.info(f"Found alternative symbol: {alt}")
            return False

        if not info.visible:
            if not mt5.symbol_select(self.symbol, True):
                logger.error(f"Cannot select {self.symbol} in Market Watch")
                return False
            logger.info(f"Added {self.symbol} to Market Watch")

        logger.info(f"Symbol {self.symbol} OK - Spread: {info.spread} points | "
                     f"Bid: {info.bid} | Ask: {info.ask}")
        return True

    def fetch_candles(self, timeframe: str, num_bars: int = 5000,
                      start_date: Optional[datetime] = None) -> pd.DataFrame:
        mt5_tf = get_mt5_timeframe(timeframe)

        if start_date:
            rates = mt5.copy_rates_from(self.symbol, mt5_tf, start_date, num_bars)
        else:
            rates = mt5.copy_rates_from_pos(self.symbol, mt5_tf, 0, num_bars)

        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            logger.error(f"Failed to fetch {timeframe} data: {error}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")

        df = df.rename(columns={
            "tick_volume": "volume",
            "real_volume": "real_volume"
        })

        columns_to_keep = ["time", "open", "high", "low", "close", "volume", "spread"]
        available_columns = [c for c in columns_to_keep if c in df.columns]
        df = df[available_columns]

        df = df.set_index("time")
        df.index.name = "time"

        logger.info(f"Fetched {len(df)} {timeframe} candles | "
                     f"From: {df.index[0]} | To: {df.index[-1]}")

        return df

    def fetch_all_timeframes(self) -> dict:
        timeframes = self.settings.get("timeframes", ["M5", "M15", "H1", "H4", "D1"])
        num_bars = self.settings["data"].get("history_bars", 5000)

        all_data = {}
        for tf in timeframes:
            logger.info(f"Fetching {tf} data...")
            df = self.fetch_candles(tf, num_bars)
            if not df.empty:
                all_data[tf] = df
            else:
                logger.warning(f"No data received for {tf}")

        return all_data

    def save_data(self, data: dict, format: str = "parquet"):
        for tf_name, df in data.items():
            if format == "parquet":
                filepath = self.data_dir / f"{self.symbol}_{tf_name}.parquet"
                df.to_parquet(filepath, engine="pyarrow")
            else:
                filepath = self.data_dir / f"{self.symbol}_{tf_name}.csv"
                df.to_csv(filepath)

            size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"Saved {tf_name}: {len(df)} candles -> {filepath.name} ({size_mb:.2f} MB)")

    def load_data(self, timeframe: str, format: str = "parquet") -> pd.DataFrame:
        if format == "parquet":
            filepath = self.data_dir / f"{self.symbol}_{timeframe}.parquet"
            if filepath.exists():
                df = pd.read_parquet(filepath)
                logger.info(f"Loaded {timeframe}: {len(df)} candles from {filepath.name}")
                return df
        else:
            filepath = self.data_dir / f"{self.symbol}_{timeframe}.csv"
            if filepath.exists():
                df = pd.read_csv(filepath, index_col="time", parse_dates=True)
                logger.info(f"Loaded {timeframe}: {len(df)} candles from {filepath.name}")
                return df

        logger.warning(f"No saved data found for {timeframe}")
        return pd.DataFrame()

    def get_latest_price(self) -> dict:
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return {}
        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": round(tick.ask - tick.bid, 2),
            "time": datetime.fromtimestamp(tick.time),
            "volume": tick.volume,
        }

    def print_data_summary(self, data: dict):
        print("\n" + "=" * 70)
        print(f"  DATA COLLECTION SUMMARY - {self.symbol}")
        print("=" * 70)

        for tf_name, df in data.items():
            duration = df.index[-1] - df.index[0]
            print(f"\n  {tf_name:>4} | {len(df):>6} candles | "
                  f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')} | "
                  f"{duration.days} days")
            print(f"       | Open: {df['open'].iloc[-1]:.2f} | High: {df['high'].max():.2f} | "
                  f"Low: {df['low'].min():.2f} | Close: {df['close'].iloc[-1]:.2f}")

        price = self.get_latest_price()
        if price:
            print(f"\n  LIVE  | Bid: {price['bid']:.2f} | Ask: {price['ask']:.2f} | "
                  f"Spread: {price['spread']:.2f}")

        print("=" * 70 + "\n")
'''

# --- src/feature_engine/__init__.py ---
FILES["src/feature_engine/__init__.py"] = ""

# --- src/strategies/__init__.py ---
FILES["src/strategies/__init__.py"] = ""

# --- src/ml_scorer/__init__.py ---
FILES["src/ml_scorer/__init__.py"] = ""

# --- src/risk_manager/__init__.py ---
FILES["src/risk_manager/__init__.py"] = ""

# --- src/executor/__init__.py ---
FILES["src/executor/__init__.py"] = ""

# --- src/dashboard/__init__.py ---
FILES["src/dashboard/__init__.py"] = ""

# --- main.py ---
FILES["main.py"] = r'''"""
XAUUSD AI Trading System - Main Entry Point
Phase 1: Data Collection & Verification

Usage:
    python main.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collector.mt5_collector import MT5Connector, DataCollector
from src.utils.logger import setup_logger

logger = setup_logger("main")


def main():
    print("\n" + "=" * 70)
    print("  XAUUSD AI TRADING SYSTEM")
    print("  Phase 1: Data Collection")
    print("=" * 70 + "\n")

    # --- Step 1: Connect to MT5 ---
    logger.info("Step 1: Connecting to MetaTrader 5...")
    connector = MT5Connector()

    if not connector.connect():
        logger.error("Cannot connect to MT5. Make sure the terminal is running!")
        return

    account = connector.get_account_info()
    print(f"\n  Account: {account['login']} ({account['trade_mode']})")
    print(f"  Server:  {account['server']}")
    print(f"  Balance: ${account['balance']:.2f}")
    print(f"  Leverage: 1:{account['leverage']}\n")

    # --- Step 2: Check symbol ---
    logger.info("Step 2: Checking XAUUSD symbol...")
    collector = DataCollector(connector)

    if not collector.check_symbol():
        logger.error("XAUUSD not available. Check your broker or symbol name.")
        connector.disconnect()
        return

    # --- Step 3: Fetch historical data ---
    logger.info("Step 3: Fetching historical data for all timeframes...")
    data = collector.fetch_all_timeframes()

    if not data:
        logger.error("No data collected!")
        connector.disconnect()
        return

    # --- Step 4: Save to disk ---
    logger.info("Step 4: Saving data to Parquet files...")
    collector.save_data(data)

    # --- Step 5: Summary ---
    collector.print_data_summary(data)

    # --- Step 6: Data quality checks ---
    logger.info("Step 5: Running data quality checks...")
    for tf_name, df in data.items():
        nulls = df.isnull().sum().sum()
        if nulls > 0:
            logger.warning(f"{tf_name}: Found {nulls} missing values!")
        else:
            logger.info(f"{tf_name}: No missing values - OK")

        zero_vol = (df["volume"] == 0).sum()
        if zero_vol > 0:
            logger.warning(f"{tf_name}: Found {zero_vol} zero-volume candles")

        dupes = df.index.duplicated().sum()
        if dupes > 0:
            logger.warning(f"{tf_name}: Found {dupes} duplicate timestamps!")
        else:
            logger.info(f"{tf_name}: No duplicates - OK")

    print("\n" + "=" * 70)
    print("  Phase 1 - Step 1 COMPLETE!")
    print("  Data is saved in: data/historical/")
    print("  Next: Run feature_engine to compute indicators")
    print("=" * 70 + "\n")

    connector.disconnect()


if __name__ == "__main__":
    main()
'''

# --- requirements.txt ---
FILES["requirements.txt"] = """MetaTrader5>=5.0.5600
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=14.0.0
pyyaml>=6.0
python-dotenv>=1.0.0
pandas-ta>=0.3.14b
scikit-learn>=1.3.0
xgboost>=2.0.0
flask>=3.0.0
requests>=2.31.0
beautifulsoup4>=4.12.0
"""

# --- .gitignore ---
FILES[".gitignore"] = """__pycache__/
*.py[cod]
.venv/
data/historical/*.parquet
data/historical/*.csv
data/models/*.pkl
data/*.db
logs/*.log
.idea/
.vscode/
"""

# --- README.md ---
FILES["README.md"] = """# XAUUSD AI Trading System

AI-powered automated trading bot for Gold (XAUUSD) on MetaTrader 5.

## Setup
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python main.py
```
"""

# ============================================================
# Create all files and directories
# ============================================================

DIRS = [
    "data/historical",
    "data/models",
    "logs",
    "tests",
    "notebooks",
    "mt5_ea",
]


def main():
    print("\n" + "=" * 60)
    print("  XAUUSD AI TRADER - Project Setup")
    print("=" * 60 + "\n")

    # Create directories
    for d in DIRS:
        os.makedirs(d, exist_ok=True)
        print(f"  [DIR]  {d}/")

    # Create files
    for filepath, content in FILES.items():
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        print(f"  [FILE] {filepath}")

    print(f"\n  Created {len(DIRS)} directories and {len(FILES)} files.")
    print("\n  Now run:  python main.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
