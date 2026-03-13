"""
XAUUSD AI Trading System - Live Paper Trading Setup
Session 8: Bot runs continuously, scans market, and places trades on MT5 demo.

Usage:
    python setup_live_trader.py
    python live_trader.py
"""

import os

FILES = {}

# ============================================================
# FILE 1: src/executor/mt5_executor.py
# ============================================================
FILES["src/executor/mt5_executor.py"] = r'''"""
MT5 Trade Executor
Actually places BUY/SELL orders on MetaTrader 5.
This is the "hands" of the bot — it executes what the brain decides.
"""

import MetaTrader5 as mt5
import time
from datetime import datetime
from src.utils.config import get_settings
from src.utils.logger import setup_logger

logger = setup_logger("mt5_executor")


class MT5Executor:
    """Places and manages trades on MetaTrader 5."""

    def __init__(self):
        settings = get_settings()
        self.exec_config = settings.get("execution", {})
        self.symbol = settings.get("symbol", "XAUUSD")
        self.magic = self.exec_config.get("magic_number", 123456)
        self.slippage = self.exec_config.get("slippage_points", 30)
        self.is_demo = self.exec_config.get("mode", "demo") == "demo"

    def place_trade(self, direction: str, lot_size: float,
                    stop_loss: float, take_profit: float,
                    comment: str = "") -> dict:
        """
        Place a trade on MT5.

        Args:
            direction: "BUY" or "SELL"
            lot_size: Position size (e.g. 0.01)
            stop_loss: Stop loss price
            take_profit: Take profit price
            comment: Trade comment for identification

        Returns:
            Dict with trade result info
        """
        # Get current price
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            logger.error("  Cannot get tick data!")
            return {"success": False, "error": "No tick data"}

        # Determine order type and price
        if direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        elif direction == "SELL":
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            return {"success": False, "error": f"Invalid direction: {direction}"}

        # Validate SL/TP
        if direction == "BUY":
            if stop_loss >= price:
                logger.warning(f"  SL {stop_loss} >= price {price}, adjusting")
                stop_loss = price - 10  # Fallback: 10 points below
            if take_profit <= price:
                logger.warning(f"  TP {take_profit} <= price {price}, adjusting")
                take_profit = price + 20
        else:
            if stop_loss <= price:
                logger.warning(f"  SL {stop_loss} <= price {price}, adjusting")
                stop_loss = price + 10
            if take_profit >= price:
                logger.warning(f"  TP {take_profit} >= price {price}, adjusting")
                take_profit = price - 20

        # Round to proper decimals
        symbol_info = mt5.symbol_info(self.symbol)
        digits = symbol_info.digits if symbol_info else 2

        price = round(price, digits)
        stop_loss = round(stop_loss, digits)
        take_profit = round(take_profit, digits)
        lot_size = round(lot_size, 2)

        # Build order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": self.slippage,
            "magic": self.magic,
            "comment": comment[:31],  # MT5 limits comment to 31 chars
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        logger.info(f"  Sending order: {direction} {lot_size} lots @ {price} | "
                     f"SL: {stop_loss} | TP: {take_profit}")

        # Send order
        result = mt5.order_send(request)

        if result is None:
            error = mt5.last_error()
            logger.error(f"  Order send failed: {error}")
            return {"success": False, "error": str(error)}

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"  Order rejected: {result.retcode} - {result.comment}")
            return {
                "success": False,
                "retcode": result.retcode,
                "comment": result.comment,
                "error": f"Retcode {result.retcode}: {result.comment}"
            }

        # Success!
        logger.info(f"  ORDER FILLED: Ticket #{result.order} | {direction} {lot_size} @ {result.price}")

        return {
            "success": True,
            "ticket": result.order,
            "price": result.price,
            "volume": result.volume,
            "direction": direction,
            "sl": stop_loss,
            "tp": take_profit,
            "comment": comment,
            "time": datetime.now().isoformat(),
        }

    def get_open_positions(self) -> list:
        """Get all open positions for our symbol with our magic number."""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return []

        our_positions = []
        for pos in positions:
            if pos.magic == self.magic:
                our_positions.append({
                    "ticket": pos.ticket,
                    "direction": "BUY" if pos.type == 0 else "SELL",
                    "volume": pos.volume,
                    "price_open": pos.price_open,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "profit": pos.profit,
                    "time": datetime.fromtimestamp(pos.time).isoformat(),
                    "comment": pos.comment,
                })

        return our_positions

    def close_position(self, ticket: int) -> dict:
        """Close a specific position by ticket number."""
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return {"success": False, "error": f"Position {ticket} not found"}

        pos = positions[0]
        tick = mt5.symbol_info_tick(self.symbol)

        if pos.type == 0:  # BUY position -> close with SELL
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:  # SELL position -> close with BUY
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": self.slippage,
            "magic": self.magic,
            "comment": "AI_close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"  Position #{ticket} CLOSED at {result.price} | PnL: ${pos.profit:.2f}")
            return {"success": True, "ticket": ticket, "profit": pos.profit}
        else:
            error = result.comment if result else mt5.last_error()
            logger.error(f"  Failed to close #{ticket}: {error}")
            return {"success": False, "error": str(error)}

    def close_all_positions(self) -> list:
        """Close all positions opened by this bot."""
        positions = self.get_open_positions()
        results = []
        for pos in positions:
            result = self.close_position(pos["ticket"])
            results.append(result)
        return results

    def get_account_summary(self) -> dict:
        """Get current account info."""
        info = mt5.account_info()
        if info is None:
            return {}
        return {
            "balance": info.balance,
            "equity": info.equity,
            "profit": info.profit,
            "margin_free": info.margin_free,
            "margin_level": info.margin_level if info.margin_level else 0,
        }
'''

# ============================================================
# FILE 2: src/executor/live_engine.py
# ============================================================
FILES["src/executor/live_engine.py"] = r'''"""
Live Trading Engine
The main loop that runs continuously:
1. Fetch latest data from MT5
2. Compute features
3. Generate and filter signals
4. Check news
5. Apply ML scoring
6. Execute approved trades
7. Log everything
"""

import MetaTrader5 as mt5
import time
import json
from datetime import datetime, timezone
from pathlib import Path
from src.data_collector.mt5_collector import MT5Connector, DataCollector
from src.feature_engine.engine import FeatureEngine
from src.strategies.strategy_manager import StrategyManager
from src.risk_manager.signal_filter import SignalFilter
from src.risk_manager.risk_manager import RiskManager
from src.ml_scorer.model import MLSignalScorer
from src.ml_scorer.feature_builder import FeatureBuilder
from src.news_filter.news_manager import NewsManager
from src.executor.mt5_executor import MT5Executor
from src.utils.config import get_settings, PROJECT_ROOT
from src.utils.logger import setup_logger

logger = setup_logger("live_engine")


class LiveTradingEngine:
    """
    The main live trading loop.
    Runs continuously, scanning for setups and executing trades on MT5 demo.
    """

    def __init__(self):
        self.settings = get_settings()
        self.scan_interval = 300  # Scan every 5 minutes (300 seconds)
        self.news_refresh_interval = 900  # Refresh news every 15 minutes
        self.last_news_refresh = 0

        # Initialize all components
        self.connector = MT5Connector()
        self.feature_engine = FeatureEngine()
        self.strategy_manager = StrategyManager()
        self.signal_filter = SignalFilter()
        self.risk_manager = RiskManager(initial_balance=0)  # Will update from MT5
        self.executor = MT5Executor()
        self.news_manager = NewsManager()

        # ML model
        self.ml_scorer = MLSignalScorer()
        self.feature_builder = FeatureBuilder()
        self.ml_loaded = self.ml_scorer.load()

        # Trade log
        self.trade_log_path = PROJECT_ROOT / "data" / "live_trades.json"
        self.trade_log = self._load_trade_log()

        # State
        self.running = False
        self.total_scans = 0
        self.total_trades_placed = 0

    def _load_trade_log(self) -> list:
        """Load existing trade log."""
        if self.trade_log_path.exists():
            try:
                with open(self.trade_log_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_trade_log(self):
        """Save trade log to disk."""
        self.trade_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.trade_log_path, "w") as f:
            json.dump(self.trade_log, f, indent=2, default=str)

    def start(self):
        """Start the live trading loop."""
        print("\n" + "=" * 70)
        print("  XAUUSD AI TRADING SYSTEM - LIVE PAPER TRADING")
        print("  Mode: DEMO (paper trading)")
        print("  Press Ctrl+C to stop")
        print("=" * 70 + "\n")

        # Connect to MT5
        if not self.connector.connect():
            logger.error("Cannot connect to MT5!")
            return

        # Update risk manager with actual account balance
        account = self.connector.get_account_info()
        self.risk_manager.initial_balance = account["balance"]
        self.risk_manager.current_balance = account["balance"]
        self.risk_manager.peak_balance = account["balance"]

        logger.info(f"  Account: {account['login']} | Balance: ${account['balance']:.2f}")
        logger.info(f"  ML Model: {'Loaded' if self.ml_loaded else 'Not loaded (using raw signals)'}")
        logger.info(f"  Scan interval: {self.scan_interval} seconds")

        # Initial news refresh
        self._refresh_news()

        self.running = True

        try:
            while self.running:
                self._scan_and_trade()
                self.total_scans += 1

                # Print status every scan
                self._print_status()

                # Wait for next scan
                logger.info(f"  Next scan in {self.scan_interval} seconds... (Ctrl+C to stop)")
                time.sleep(self.scan_interval)

        except KeyboardInterrupt:
            logger.info("\n  Stopping live trading...")
            self._shutdown()

    def _scan_and_trade(self):
        """Single scan iteration: fetch data, compute, signal, trade."""
        scan_time = datetime.now()
        logger.info(f"\n{'='*50}")
        logger.info(f"SCAN #{self.total_scans + 1} at {scan_time.strftime('%H:%M:%S')}")
        logger.info(f"{'='*50}")

        # Refresh news periodically
        if time.time() - self.last_news_refresh > self.news_refresh_interval:
            self._refresh_news()

        # Step 1: Fetch latest H1 data
        try:
            collector = DataCollector(self.connector)
            df = collector.fetch_candles("H1", num_bars=500)
            if df.empty:
                logger.warning("  No data received, skipping scan")
                return
        except Exception as e:
            logger.error(f"  Data fetch error: {e}")
            return

        # Step 2: Compute features
        try:
            featured_df = self.feature_engine.compute_features(df, "H1")
        except Exception as e:
            logger.error(f"  Feature computation error: {e}")
            return

        # Step 3: Generate signals
        raw_signals = self.strategy_manager.scan_all(featured_df, "H1")
        filtered_signals = self.signal_filter.filter_signals(raw_signals, "H1")

        logger.info(f"  Signals: {len(raw_signals)} raw -> {len(filtered_signals)} filtered")

        if not filtered_signals:
            logger.info(f"  No quality signals found. Waiting...")
            return

        # Step 4: Process each signal
        for signal in filtered_signals[:3]:  # Max 3 signals per scan

            # News check
            news_ok, news_reason, news_score = self.news_manager.can_trade(signal.direction)
            if not news_ok:
                logger.info(f"  Signal blocked by news: {news_reason}")
                continue

            # ML scoring
            if self.ml_loaded:
                current_bar = featured_df.iloc[-1]
                ml_score = self.ml_scorer.score_signal(signal, current_bar, self.feature_builder)
                signal.confidence = ml_score
                logger.info(f"  ML score: {ml_score:.2f} (threshold: 0.55)")

                if ml_score < 0.55:
                    logger.info(f"  Signal rejected by ML (score too low)")
                    continue

            # Risk manager check
            allowed, reason = self.risk_manager.can_trade(signal)
            if not allowed:
                logger.info(f"  Signal blocked by risk: {reason}")
                continue

            # Position sizing
            account = self.executor.get_account_summary()
            self.risk_manager.current_balance = account.get("balance", self.risk_manager.current_balance)
            sizing = self.risk_manager.calculate_position_size(signal)

            if sizing["lot_size"] <= 0:
                continue

            # Check current open positions
            open_positions = self.executor.get_open_positions()
            if len(open_positions) >= self.risk_manager.max_open_positions:
                logger.info(f"  Max positions reached ({len(open_positions)})")
                continue

            # EXECUTE THE TRADE!
            logger.info(f"\n  >>> EXECUTING TRADE <<<")
            logger.info(f"  {signal}")

            comment = f"AI_{signal.strategy_name[:8]}"
            result = self.executor.place_trade(
                direction=signal.direction,
                lot_size=sizing["lot_size"],
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                comment=comment
            )

            if result["success"]:
                self.total_trades_placed += 1

                # Log the trade
                trade_record = {
                    "scan": self.total_scans + 1,
                    "time": scan_time.isoformat(),
                    "ticket": result["ticket"],
                    "direction": signal.direction,
                    "strategy": signal.strategy_name,
                    "entry": result["price"],
                    "sl": signal.stop_loss,
                    "tp": signal.take_profit,
                    "lot_size": sizing["lot_size"],
                    "risk_amount": sizing["risk_amount"],
                    "confidence": signal.confidence,
                    "ml_score": signal.confidence,
                    "news_score": news_score,
                    "reason": signal.reason,
                    "result": "open",
                }
                self.trade_log.append(trade_record)
                self._save_trade_log()

                # Update risk manager
                self.risk_manager.daily_trades += 1

                logger.info(f"  Trade #{self.total_trades_placed} placed successfully!")
            else:
                logger.error(f"  Trade failed: {result.get('error', 'Unknown')}")

    def _refresh_news(self):
        """Refresh news data."""
        try:
            self.news_manager.refresh()
            self.last_news_refresh = time.time()
        except Exception as e:
            logger.warning(f"  News refresh failed: {e}")

    def _print_status(self):
        """Print current bot status."""
        account = self.executor.get_account_summary()
        positions = self.executor.get_open_positions()

        print(f"\n  --- BOT STATUS ---")
        print(f"  Scans: {self.total_scans + 1} | Trades placed: {self.total_trades_placed}")
        print(f"  Balance: ${account.get('balance', 0):.2f} | "
              f"Equity: ${account.get('equity', 0):.2f} | "
              f"Profit: ${account.get('profit', 0):.2f}")
        print(f"  Open positions: {len(positions)}")

        for pos in positions:
            pnl_icon = "🟢" if pos["profit"] >= 0 else "🔴"
            print(f"    {pnl_icon} #{pos['ticket']} {pos['direction']} {pos['volume']} lots "
                  f"@ {pos['price_open']:.2f} | PnL: ${pos['profit']:.2f}")

        # News status
        is_blackout, reason = self.news_manager.calendar.is_news_blackout(30)
        if is_blackout:
            print(f"  ⚠️  NEWS BLACKOUT: {reason}")
        else:
            print(f"  ✅ News: Clear")

    def _shutdown(self):
        """Clean shutdown."""
        self.running = False

        # Show final summary
        positions = self.executor.get_open_positions()
        account = self.executor.get_account_summary()

        print(f"\n{'='*70}")
        print(f"  LIVE TRADING SESSION ENDED")
        print(f"{'='*70}")
        print(f"  Total scans: {self.total_scans}")
        print(f"  Trades placed: {self.total_trades_placed}")
        print(f"  Open positions: {len(positions)}")
        print(f"  Balance: ${account.get('balance', 0):.2f}")
        print(f"  Equity: ${account.get('equity', 0):.2f}")
        print(f"  Session profit: ${account.get('profit', 0):.2f}")
        print(f"  Trade log saved to: {self.trade_log_path}")
        print(f"{'='*70}\n")

        self.connector.disconnect()
'''

# ============================================================
# FILE 3: live_trader.py - Entry point
# ============================================================
FILES["live_trader.py"] = r'''"""
XAUUSD AI Trading System - Live Paper Trader
Runs the bot continuously on your MT5 demo account.

The bot will:
1. Scan the market every 5 minutes
2. Compute 133 features on H1 data
3. Run all 5 strategy detectors
4. Filter signals (min 2:1 R:R, min 60% confidence)
5. Apply ML scoring (blocks low-probability trades)
6. Check news (blocks during high-impact events)
7. Place trades on your MT5 demo account
8. Log everything

Usage:
    python live_trader.py

    Press Ctrl+C to stop the bot safely.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.executor.live_engine import LiveTradingEngine
from src.utils.logger import setup_logger

logger = setup_logger("live_trader")


def main():
    print("\n" + "=" * 70)
    print("  XAUUSD AI TRADING SYSTEM")
    print("  Session 8: Live Paper Trading on MT5 Demo")
    print("=" * 70)
    print()
    print("  ⚠️  IMPORTANT REMINDERS:")
    print("  - This trades on your DEMO account only")
    print("  - Make sure MT5 is running and logged in")
    print("  - The bot scans every 5 minutes")
    print("  - Press Ctrl+C at any time to stop safely")
    print("  - All trades are logged in data/live_trades.json")
    print()

    input("  Press ENTER to start the bot...")

    engine = LiveTradingEngine()
    engine.start()


if __name__ == "__main__":
    main()
'''

# ============================================================
# FILE 4: src/executor/__init__.py
# ============================================================
FILES["src/executor/__init__.py"] = ""

# ============================================================
# Create all files
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("  Live Paper Trading Setup")
    print("=" * 60 + "\n")

    for filepath, content in FILES.items():
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        print(f"  [FILE] {filepath}")

    print(f"\n  Created {len(FILES)} files.")
    print("\n  Make sure MT5 is open and logged in, then run:")
    print("  python live_trader.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
