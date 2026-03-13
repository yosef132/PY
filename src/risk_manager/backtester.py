"""
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
