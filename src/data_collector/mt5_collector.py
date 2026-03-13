"""
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
