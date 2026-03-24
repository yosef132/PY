"""
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
        self.min_signal_confidence = self.config.get("min_signal_confidence", 0.60)

        # State tracking
        self.open_positions = []
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.max_daily_trades = self.config.get("max_daily_trades", 5)
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
        if signal.confidence < self.min_signal_confidence:
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
