"""
XAUUSD AI Trading System - Risk Manager & Backtester Setup
Session 4: Build the safety layer that filters signals and simulates trading.

Usage:
    python setup_risk_manager.py
    python run_backtest.py
"""

import os

FILES = {}

# ============================================================
# FILE 1: src/risk_manager/signal_filter.py
# ============================================================
FILES["src/risk_manager/signal_filter.py"] = r'''"""
Signal Filter
First line of defense: filters raw strategy signals based on quality rules.
Reduces thousands of signals to only tradeable setups.
"""

import pandas as pd
import numpy as np
from src.utils.config import get_settings
from src.utils.logger import setup_logger

logger = setup_logger("signal_filter")


class SignalFilter:
    """
    Filters raw signals based on quality criteria.
    This runs BEFORE the Risk Manager to remove obviously bad signals.
    """

    def __init__(self):
        settings = get_settings()
        self.risk_config = settings.get("risk", {})
        self.session_config = settings.get("sessions", {})

        # Filter thresholds
        self.min_rr = self.risk_config.get("min_risk_reward", 2.0)
        self.min_confidence = 0.60
        self.max_signals_per_bar = 1  # Only 1 signal per candle
        self.min_sl_distance_pct = 0.05  # SL must be at least 0.05% from entry
        self.max_sl_distance_pct = 2.0   # SL can't be more than 2% from entry

    def filter_signals(self, signals: list, timeframe: str = "") -> list:
        """
        Filter raw signals and return only quality setups.
        """
        if not signals:
            return []

        original_count = len(signals)
        filtered = []

        for sig in signals:
            # --- Filter 1: Minimum Risk-Reward Ratio ---
            if sig.risk_reward < self.min_rr:
                continue

            # --- Filter 2: Minimum Confidence ---
            if sig.confidence < self.min_confidence:
                continue

            # --- Filter 3: SL Distance Sanity Check ---
            if sig.entry_price <= 0:
                continue

            sl_distance_pct = abs(sig.entry_price - sig.stop_loss) / sig.entry_price * 100

            # SL too tight = will get stopped out by noise
            if sl_distance_pct < self.min_sl_distance_pct:
                continue

            # SL too wide = risking too much per trade
            if sl_distance_pct > self.max_sl_distance_pct:
                continue

            # --- Filter 4: TP must be on the correct side ---
            if sig.direction == "BUY":
                if sig.take_profit <= sig.entry_price or sig.stop_loss >= sig.entry_price:
                    continue
            elif sig.direction == "SELL":
                if sig.take_profit >= sig.entry_price or sig.stop_loss <= sig.entry_price:
                    continue
            else:
                continue

            filtered.append(sig)

        # --- Filter 5: Deduplicate — max 1 signal per timestamp per direction ---
        seen = set()
        deduped = []
        for sig in filtered:
            key = (sig.timestamp, sig.direction)
            if key not in seen:
                seen.add(key)
                deduped.append(sig)

        # Sort by confidence (best first)
        deduped.sort(key=lambda s: s.confidence, reverse=True)

        logger.info(f"  [{timeframe}] Signal Filter: {original_count} -> {len(deduped)} "
                     f"(removed {original_count - len(deduped)} low-quality signals)")

        return deduped
'''

# ============================================================
# FILE 2: src/risk_manager/risk_manager.py
# ============================================================
FILES["src/risk_manager/risk_manager.py"] = r'''"""
Risk Manager
The most critical module. Has VETO POWER over any trade signal.
Enforces position sizing, daily loss limits, drawdown protection, and session rules.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.utils.config import get_settings
from src.utils.logger import setup_logger

logger = setup_logger("risk_manager")


class RiskManager:
    """
    Enforces all risk rules. No trade can be placed without passing through here.
    """

    def __init__(self, initial_balance: float = 3000.0):
        settings = get_settings()
        self.config = settings.get("risk", {})
        self.session_config = settings.get("sessions", {})

        # Account state
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.equity = initial_balance

        # Risk parameters
        self.risk_per_trade_pct = self.config.get("risk_per_trade_pct", 0.5)
        self.max_daily_loss_pct = self.config.get("max_daily_loss_pct", 3.0)
        self.max_open_positions = self.config.get("max_open_positions", 2)
        self.max_drawdown_pct = self.config.get("max_drawdown_pct", 10.0)
        self.min_risk_reward = self.config.get("min_risk_reward", 2.0)

        # State tracking
        self.open_positions = []
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.max_daily_trades = 5
        self.current_date = None
        self.halted = False
        self.halt_reason = ""

        # Trade log
        self.trade_history = []

    def reset_daily(self, date):
        """Reset daily counters at start of new trading day."""
        if self.current_date != date:
            self.current_date = date
            self.daily_pnl = 0.0
            self.daily_trades = 0
            if self.halted and "daily" in self.halt_reason:
                self.halted = False
                self.halt_reason = ""
                logger.info(f"  Daily halt lifted for new day: {date}")

    def can_trade(self, signal) -> tuple:
        """
        Check if a trade is allowed. Returns (allowed: bool, reason: str).
        This is the VETO gate.
        """
        # --- Check 1: System halted? ---
        if self.halted:
            return False, f"System halted: {self.halt_reason}"

        # --- Check 2: Max drawdown check ---
        drawdown_pct = (self.peak_balance - self.current_balance) / self.peak_balance * 100
        if drawdown_pct >= self.max_drawdown_pct:
            self.halted = True
            self.halt_reason = f"Max drawdown {drawdown_pct:.1f}% >= {self.max_drawdown_pct}%"
            logger.warning(f"  HALTED: {self.halt_reason}")
            return False, self.halt_reason

        # --- Check 3: Daily loss limit ---
        daily_loss_pct = abs(min(self.daily_pnl, 0)) / self.current_balance * 100
        if daily_loss_pct >= self.max_daily_loss_pct:
            self.halted = True
            self.halt_reason = f"daily loss {daily_loss_pct:.1f}% >= {self.max_daily_loss_pct}%"
            logger.warning(f"  HALTED: {self.halt_reason}")
            return False, self.halt_reason

        # --- Check 4: Max daily trades ---
        if self.daily_trades >= self.max_daily_trades:
            return False, f"Max daily trades reached ({self.max_daily_trades})"

        # --- Check 5: Max open positions ---
        if len(self.open_positions) >= self.max_open_positions:
            return False, f"Max open positions ({self.max_open_positions})"

        # --- Check 6: No duplicate direction ---
        for pos in self.open_positions:
            if pos["direction"] == signal.direction:
                return False, "Already have a position in same direction"

        # --- Check 7: Minimum R:R ratio ---
        if signal.risk_reward < self.min_risk_reward:
            return False, f"R:R {signal.risk_reward:.1f} < minimum {self.min_risk_reward}"

        # --- Check 8: Minimum confidence ---
        if signal.confidence < 0.60:
            return False, f"Confidence {signal.confidence:.0%} too low"

        return True, "Approved"

    def calculate_position_size(self, signal) -> dict:
        """
        Calculate position size based on risk per trade.
        Risk amount = balance * risk_pct
        Position size = risk_amount / SL_distance_in_dollars
        """
        risk_amount = self.current_balance * (self.risk_per_trade_pct / 100)
        sl_distance = abs(signal.entry_price - signal.stop_loss)

        if sl_distance <= 0:
            return {"lot_size": 0.01, "risk_amount": 0, "sl_points": 0}

        # For XAUUSD: 1 lot = 100 oz, 1 point = $0.01 per oz
        # So 0.01 lot: $1 per $1 move in gold price
        # Position size = risk_amount / sl_distance (for 0.01 lot increments)
        raw_lots = risk_amount / (sl_distance * 100)  # 100 oz per lot

        # Round down to nearest 0.01 lot
        lot_size = max(0.01, round(raw_lots, 2))

        # Cap at reasonable size for the account
        max_lots = self.current_balance / 10000  # Rough safety cap
        lot_size = min(lot_size, max(0.01, round(max_lots, 2)))

        actual_risk = lot_size * sl_distance * 100

        return {
            "lot_size": lot_size,
            "risk_amount": round(actual_risk, 2),
            "risk_pct": round(actual_risk / self.current_balance * 100, 2),
            "sl_distance": round(sl_distance, 2),
        }

    def open_trade(self, signal, sizing: dict):
        """Record a new open position."""
        trade = {
            "id": len(self.trade_history) + 1,
            "direction": signal.direction,
            "strategy": signal.strategy_name,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "lot_size": sizing["lot_size"],
            "risk_amount": sizing["risk_amount"],
            "entry_time": signal.timestamp,
            "confidence": signal.confidence,
            "risk_reward": signal.risk_reward,
            "reason": signal.reason,
            "tags": signal.tags,
            "status": "open",
            "pnl": 0.0,
        }

        self.open_positions.append(trade)
        self.daily_trades += 1

        logger.info(f"  TRADE OPENED #{trade['id']}: {signal.direction} {sizing['lot_size']} lots "
                     f"@ {signal.entry_price:.2f} | SL: {signal.stop_loss:.2f} | "
                     f"TP: {signal.take_profit:.2f} | Risk: ${sizing['risk_amount']:.2f}")

        return trade

    def check_exits(self, current_high: float, current_low: float, current_close: float,
                    current_time=None) -> list:
        """
        Check if any open positions should be closed.
        Returns list of closed trades.
        """
        closed = []

        for pos in self.open_positions[:]:  # Copy list to allow removal
            hit_sl = False
            hit_tp = False

            if pos["direction"] == "BUY":
                hit_sl = current_low <= pos["stop_loss"]
                hit_tp = current_high >= pos["take_profit"]
            elif pos["direction"] == "SELL":
                hit_sl = current_high >= pos["stop_loss"]
                hit_tp = current_low <= pos["take_profit"]

            if hit_sl or hit_tp:
                # Calculate PnL
                if hit_tp:
                    exit_price = pos["take_profit"]
                    pos["status"] = "tp_hit"
                else:
                    exit_price = pos["stop_loss"]
                    pos["status"] = "sl_hit"

                if pos["direction"] == "BUY":
                    pnl = (exit_price - pos["entry_price"]) * pos["lot_size"] * 100
                else:
                    pnl = (pos["entry_price"] - exit_price) * pos["lot_size"] * 100

                pos["pnl"] = round(pnl, 2)
                pos["exit_price"] = exit_price
                pos["exit_time"] = current_time

                # Update account
                self.current_balance += pnl
                self.daily_pnl += pnl
                if self.current_balance > self.peak_balance:
                    self.peak_balance = self.current_balance

                self.trade_history.append(pos)
                self.open_positions.remove(pos)
                closed.append(pos)

                result = "WIN" if pnl > 0 else "LOSS"
                logger.info(f"  TRADE CLOSED #{pos['id']}: {result} ${pnl:+.2f} | "
                             f"Balance: ${self.current_balance:.2f}")

        return closed

    def get_stats(self) -> dict:
        """Get comprehensive trading statistics."""
        if not self.trade_history:
            return {"total_trades": 0}

        trades = self.trade_history
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]

        total_pnl = sum(t["pnl"] for t in trades)
        win_rate = len(wins) / len(trades) * 100 if trades else 0

        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t["pnl"]) for t in losses]) if losses else 0
        profit_factor = sum(t["pnl"] for t in wins) / sum(abs(t["pnl"]) for t in losses) if losses else 0

        max_drawdown = 0
        peak = self.initial_balance
        running_balance = self.initial_balance
        for t in trades:
            running_balance += t["pnl"]
            if running_balance > peak:
                peak = running_balance
            dd = (peak - running_balance) / peak * 100
            if dd > max_drawdown:
                max_drawdown = dd

        # Strategy breakdown
        strategy_stats = {}
        for t in trades:
            strat = t["strategy"]
            if strat not in strategy_stats:
                strategy_stats[strat] = {"wins": 0, "losses": 0, "pnl": 0}
            if t["pnl"] > 0:
                strategy_stats[strat]["wins"] += 1
            else:
                strategy_stats[strat]["losses"] += 1
            strategy_stats[strat]["pnl"] += t["pnl"]

        return {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "final_balance": round(self.current_balance, 2),
            "return_pct": round((self.current_balance - self.initial_balance) / self.initial_balance * 100, 2),
            "strategy_breakdown": strategy_stats,
        }
'''

# ============================================================
# FILE 3: src/risk_manager/backtester.py
# ============================================================
FILES["src/risk_manager/backtester.py"] = r'''"""
Backtester
Simulates trading on historical data to measure strategy performance.
This is how we know if the system works BEFORE risking real money.
"""

import pandas as pd
import numpy as np
from src.strategies.strategy_manager import StrategyManager
from src.risk_manager.signal_filter import SignalFilter
from src.risk_manager.risk_manager import RiskManager
from src.utils.logger import setup_logger

logger = setup_logger("backtester")


class Backtester:
    """
    Simulates the full trading pipeline on historical data:
    Data -> Features -> Strategies -> Filter -> Risk Manager -> Simulated Trades
    """

    def __init__(self, initial_balance: float = 3000.0):
        self.initial_balance = initial_balance

    def run(self, featured_data: dict, target_timeframes: list = None) -> dict:
        """
        Run backtest on featured data.

        Args:
            featured_data: Dict of {timeframe: DataFrame with features}
            target_timeframes: Which timeframes to trade on (default: ["H1"])

        Returns:
            Dict with backtest results and statistics
        """
        if target_timeframes is None:
            target_timeframes = ["H1"]

        strategy_manager = StrategyManager()
        signal_filter = SignalFilter()
        risk_manager = RiskManager(self.initial_balance)

        all_trades = []
        balance_curve = [self.initial_balance]

        for tf_name in target_timeframes:
            if tf_name not in featured_data:
                logger.warning(f"  {tf_name} not found in featured data, skipping")
                continue

            df = featured_data[tf_name]
            logger.info(f"\n  Backtesting on {tf_name}: {len(df)} candles...")

            # Generate all signals for this timeframe
            raw_signals = strategy_manager.scan_all(df, tf_name)
            logger.info(f"  Raw signals: {len(raw_signals)}")

            # Filter signals
            filtered_signals = signal_filter.filter_signals(raw_signals, tf_name)
            logger.info(f"  Filtered signals: {len(filtered_signals)}")

            # Build signal lookup by timestamp
            signal_map = {}
            for sig in filtered_signals:
                ts = sig.timestamp
                if ts not in signal_map:
                    signal_map[ts] = []
                signal_map[ts].append(sig)

            # Walk forward through data bar by bar
            for i in range(1, len(df)):
                current_bar = df.iloc[i]
                current_time = df.index[i]

                # Reset daily counters
                current_date = current_time.date() if hasattr(current_time, 'date') else None
                if current_date:
                    risk_manager.reset_daily(current_date)

                # Check exits on open positions
                closed = risk_manager.check_exits(
                    current_high=current_bar["high"],
                    current_low=current_bar["low"],
                    current_close=current_bar["close"],
                    current_time=current_time
                )

                for trade in closed:
                    all_trades.append(trade)

                balance_curve.append(risk_manager.current_balance)

                # Check for new signals at this bar
                if current_time in signal_map:
                    for signal in signal_map[current_time]:
                        # Ask risk manager for permission
                        allowed, reason = risk_manager.can_trade(signal)
                        if not allowed:
                            continue

                        # Calculate position size
                        sizing = risk_manager.calculate_position_size(signal)
                        if sizing["lot_size"] <= 0:
                            continue

                        # Open the trade
                        risk_manager.open_trade(signal, sizing)

        # Close any remaining open positions at last price
        if risk_manager.open_positions and len(df) > 0:
            last_bar = df.iloc[-1]
            for pos in risk_manager.open_positions[:]:
                if pos["direction"] == "BUY":
                    pnl = (last_bar["close"] - pos["entry_price"]) * pos["lot_size"] * 100
                else:
                    pnl = (pos["entry_price"] - last_bar["close"]) * pos["lot_size"] * 100
                pos["pnl"] = round(pnl, 2)
                pos["status"] = "forced_close"
                pos["exit_price"] = last_bar["close"]
                risk_manager.current_balance += pnl
                risk_manager.trade_history.append(pos)
                risk_manager.open_positions.remove(pos)

        # Get final stats
        stats = risk_manager.get_stats()
        stats["balance_curve"] = balance_curve

        return stats

    def print_results(self, stats: dict):
        """Pretty print backtest results."""
        print("\n" + "=" * 70)
        print("  BACKTEST RESULTS")
        print("=" * 70)

        if stats["total_trades"] == 0:
            print("\n  No trades were executed during the backtest period.")
            print("  This could mean filters are too strict or no high-quality setups occurred.")
            print("=" * 70 + "\n")
            return

        print(f"\n  {'Total Trades:':<25} {stats['total_trades']}")
        print(f"  {'Wins:':<25} {stats['wins']} ({stats['win_rate']}%)")
        print(f"  {'Losses:':<25} {stats['losses']}")
        print(f"  {'Total PnL:':<25} ${stats['total_pnl']:+.2f}")
        print(f"  {'Return:':<25} {stats['return_pct']:+.1f}%")
        print(f"  {'Final Balance:':<25} ${stats['final_balance']:.2f}")
        print(f"  {'Max Drawdown:':<25} {stats['max_drawdown_pct']:.1f}%")
        print(f"  {'Profit Factor:':<25} {stats['profit_factor']:.2f}")
        print(f"  {'Avg Win:':<25} ${stats['avg_win']:.2f}")
        print(f"  {'Avg Loss:':<25} ${stats['avg_loss']:.2f}")

        if stats.get("strategy_breakdown"):
            print(f"\n  --- Performance by Strategy ---")
            for strat, data in stats["strategy_breakdown"].items():
                total = data["wins"] + data["losses"]
                wr = data["wins"] / total * 100 if total > 0 else 0
                print(f"  {strat:>25}: {total} trades | "
                      f"Win rate: {wr:.0f}% | PnL: ${data['pnl']:+.2f}")

        # Rating
        print(f"\n  --- SYSTEM RATING ---")
        score = 0
        if stats["win_rate"] >= 50:
            score += 1
            print(f"  [PASS] Win rate {stats['win_rate']}% >= 50%")
        else:
            print(f"  [FAIL] Win rate {stats['win_rate']}% < 50%")

        if stats["profit_factor"] >= 1.5:
            score += 1
            print(f"  [PASS] Profit factor {stats['profit_factor']} >= 1.5")
        elif stats["profit_factor"] >= 1.0:
            print(f"  [WARN] Profit factor {stats['profit_factor']} (needs >= 1.5)")
        else:
            print(f"  [FAIL] Profit factor {stats['profit_factor']} < 1.0")

        if stats["max_drawdown_pct"] <= 10:
            score += 1
            print(f"  [PASS] Max drawdown {stats['max_drawdown_pct']}% <= 10%")
        else:
            print(f"  [FAIL] Max drawdown {stats['max_drawdown_pct']}% > 10%")

        if stats["return_pct"] > 0:
            score += 1
            print(f"  [PASS] Positive return: {stats['return_pct']:+.1f}%")
        else:
            print(f"  [FAIL] Negative return: {stats['return_pct']:+.1f}%")

        rating = ["POOR", "NEEDS WORK", "ACCEPTABLE", "GOOD", "EXCELLENT"]
        print(f"\n  Overall: {rating[score]} ({score}/4)")

        if score < 3:
            print(f"\n  NOTE: This is the first backtest with untuned parameters.")
            print(f"  The ML model (next session) will learn which signals actually work")
            print(f"  and dramatically improve these results.")

        print("=" * 70 + "\n")
'''

# ============================================================
# FILE 4: run_backtest.py - Main test script
# ============================================================
FILES["run_backtest.py"] = r'''"""
XAUUSD AI Trading System - Backtest Runner
Run the full pipeline: Data -> Features -> Strategies -> Filter -> Risk Manager -> Results

Usage:
    python run_backtest.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collector.mt5_collector import MT5Connector, DataCollector
from src.feature_engine.engine import FeatureEngine
from src.risk_manager.backtester import Backtester
from src.utils.logger import setup_logger

logger = setup_logger("run_backtest")


def main():
    print("\n" + "=" * 70)
    print("  XAUUSD AI TRADING SYSTEM")
    print("  Phase 2 - Session 4: Risk Manager & Backtester")
    print("=" * 70 + "\n")

    # --- Step 1: Get data ---
    logger.info("Step 1: Fetching data from MT5...")
    connector = MT5Connector()
    if not connector.connect():
        logger.error("Cannot connect to MT5!")
        return

    collector = DataCollector(connector)
    collector.check_symbol()
    data = collector.fetch_all_timeframes()
    connector.disconnect()

    if not data:
        logger.error("No data!")
        return

    # --- Step 2: Compute features ---
    logger.info("\nStep 2: Computing features...\n")
    engine = FeatureEngine()
    featured_data = engine.compute_multi_timeframe(data)

    # --- Step 3: Run backtest ---
    logger.info("\nStep 3: Running backtest...\n")
    backtester = Backtester(initial_balance=3000.0)

    # Backtest on H1 (best balance of signal quality and trade frequency)
    stats = backtester.run(featured_data, target_timeframes=["H1"])

    # --- Step 4: Print results ---
    backtester.print_results(stats)

    # --- Step 5: Also test H4 ---
    print("\n  Running H4 backtest for comparison...\n")
    backtester_h4 = Backtester(initial_balance=3000.0)
    stats_h4 = backtester_h4.run(featured_data, target_timeframes=["H4"])
    backtester_h4.print_results(stats_h4)

    print("\n" + "=" * 70)
    print("  Phase 2 - Session 4 COMPLETE!")
    print("  Risk Manager and Backtester are working!")
    print("  Next: Build ML Signal Scorer to improve signal quality")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
'''

# ============================================================
# FILE 5: src/risk_manager/__init__.py (update)
# ============================================================
FILES["src/risk_manager/__init__.py"] = ""

# ============================================================
# Create all files
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("  Risk Manager & Backtester Setup")
    print("=" * 60 + "\n")

    for filepath, content in FILES.items():
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        print(f"  [FILE] {filepath}")

    print(f"\n  Created {len(FILES)} files.")
    print("\n  Now run:  python run_backtest.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
