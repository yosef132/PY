"""
Enhanced Backtester with ML
Runs two backtests:
1. WITHOUT ML (baseline) - raw strategy signals
2. WITH ML scoring - only take trades the ML model approves

This proves whether the ML model actually improves results.
"""

import pandas as pd
import numpy as np
from src.strategies.strategy_manager import StrategyManager
from src.risk_manager.signal_filter import SignalFilter
from src.risk_manager.risk_manager import RiskManager
from src.ml_scorer.feature_builder import FeatureBuilder
from src.ml_scorer.model import MLSignalScorer
from src.utils.logger import setup_logger

logger = setup_logger("ml_backtester")


class EnhancedBacktester:
    """
    Two-phase backtester:
    Phase A: Run without ML to collect training data
    Phase B: Train ML model, then re-run with ML filtering
    """

    def __init__(self, initial_balance: float = 3000.0):
        self.initial_balance = initial_balance

    def run_full_pipeline(self, featured_data: dict, timeframe: str = "H1",
                          train_pct: float = 0.70) -> dict:
        """
        Run the complete ML pipeline using a walk-forward train/test split.

        IMPORTANT: To avoid data snooping, the ML model is trained ONLY on the
        first `train_pct` of bars, and tested on the remaining held-out bars.
        This gives honest out-of-sample performance numbers.

        Pipeline:
        1. Split data: train window (first 70%) + test window (last 30%)
        2. Phase A: Baseline backtest on FULL data (no ML) — for comparison only
        3. Phase B: Generate training trades from TRAIN window only, train ML model
        4. Phase C: Run ML-enhanced backtest on TEST window only (out-of-sample)
        5. Compare results
        """
        if timeframe not in featured_data:
            logger.error(f"  {timeframe} not found in data")
            return {}

        df = featured_data[timeframe]
        split_idx = int(len(df) * train_pct)
        df_train = df.iloc[:split_idx]
        df_test  = df.iloc[split_idx:]

        logger.info(f"\n  Walk-forward split: {len(df_train)} train bars | "
                    f"{len(df_test)} test bars ({train_pct*100:.0f}/{(1-train_pct)*100:.0f})")

        # ========== PHASE A: Baseline on FULL data (no ML) ==========
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE A: Baseline backtest WITHOUT ML ({timeframe}, full data)")
        logger.info(f"{'='*60}")

        baseline_trades, baseline_stats = self._run_backtest(df, timeframe, use_ml=False)

        # ========== PHASE B: Train ML on TRAIN window only ==========
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE B: Training ML model on TRAIN window ({len(df_train)} bars)")
        logger.info(f"{'='*60}")

        train_trades, _ = self._run_backtest(df_train, timeframe, use_ml=False)

        if len(train_trades) < 10:
            logger.warning(f"  Only {len(train_trades)} trades in train window - not enough to train ML")
            return {
                "baseline": baseline_stats,
                "ml_enhanced": None,
                "ml_metrics": None,
                "improvement": None,
                "split": {"train_bars": len(df_train), "test_bars": len(df_test)},
            }

        feature_builder = FeatureBuilder()
        X, y, feature_names = feature_builder.build_training_data(train_trades, df_train)

        ml_scorer = MLSignalScorer()
        ml_metrics = ml_scorer.train(X, y, feature_names)

        # Save the model
        ml_scorer.save()

        # ========== PHASE C: ML-Enhanced on TEST window (out-of-sample) ==========
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE C: ML-enhanced backtest on TEST window (out-of-sample, {len(df_test)} bars)")
        logger.info(f"{'='*60}")

        ml_trades, ml_stats = self._run_backtest(
            df_test, timeframe, use_ml=True,
            ml_scorer=ml_scorer, feature_builder=feature_builder
        )

        # Baseline on same test window for fair comparison
        baseline_test_trades, baseline_test_stats = self._run_backtest(
            df_test, timeframe, use_ml=False
        )

        # ========== Compare (test window only — apples to apples) ==========
        improvement = {}
        if baseline_test_stats["total_trades"] > 0 and ml_stats["total_trades"] > 0:
            improvement = {
                "trades_reduction": baseline_test_stats["total_trades"] - ml_stats["total_trades"],
                "win_rate_change": ml_stats["win_rate"] - baseline_test_stats["win_rate"],
                "pnl_change": ml_stats["total_pnl"] - baseline_test_stats["total_pnl"],
                "drawdown_change": ml_stats["max_drawdown_pct"] - baseline_test_stats["max_drawdown_pct"],
                "profit_factor_change": ml_stats["profit_factor"] - baseline_test_stats["profit_factor"],
            }

        return {
            "baseline": baseline_test_stats,       # baseline on test window
            "baseline_full": baseline_stats,        # baseline on full data (reference)
            "ml_enhanced": ml_stats,                # ML on test window (out-of-sample)
            "ml_metrics": ml_metrics,
            "improvement": improvement,
            "split": {"train_bars": len(df_train), "test_bars": len(df_test),
                      "train_trades": len(train_trades), "test_trades": len(ml_trades)},
        }

    def _run_backtest(self, df, timeframe, use_ml=False,
                      ml_scorer=None, feature_builder=None, ml_threshold=0.55):
        """Run a single backtest pass."""

        strategy_manager = StrategyManager()
        signal_filter = SignalFilter()
        risk_manager = RiskManager(self.initial_balance)

        # Generate and filter signals
        raw_signals = strategy_manager.scan_all(df, timeframe)
        filtered_signals = signal_filter.filter_signals(raw_signals, timeframe)

        # Build signal lookup
        signal_map = {}
        for sig in filtered_signals:
            ts = sig.timestamp
            if ts not in signal_map:
                signal_map[ts] = []
            signal_map[ts].append(sig)

        # Walk forward
        for i in range(1, len(df)):
            current_bar = df.iloc[i]
            current_time = df.index[i]

            current_date = current_time.date() if hasattr(current_time, 'date') else None
            if current_date:
                risk_manager.reset_daily(current_date)

            # Check exits
            risk_manager.check_exits(
                current_high=current_bar["high"],
                current_low=current_bar["low"],
                current_close=current_bar["close"],
                current_time=current_time
            )

            # Process new signals
            if current_time in signal_map:
                for signal in signal_map[current_time]:
                    # ML FILTER: Only if ML is enabled
                    if use_ml and ml_scorer and feature_builder:
                        ml_score = ml_scorer.score_signal(signal, current_bar, feature_builder)

                        # Replace signal confidence with ML-blended score
                        signal.confidence = ml_score

                        # Skip if ML score is too low
                        if ml_score < ml_threshold:
                            continue

                    # Risk manager check
                    allowed, reason = risk_manager.can_trade(signal)
                    if not allowed:
                        continue

                    sizing = risk_manager.calculate_position_size(signal)
                    if sizing["lot_size"] <= 0:
                        continue

                    risk_manager.open_trade(signal, sizing)

        # Close remaining positions
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

        stats = risk_manager.get_stats()
        return risk_manager.trade_history, stats

    def print_comparison(self, results: dict):
        """Print side-by-side comparison of baseline vs ML-enhanced."""

        print("\n" + "=" * 70)
        print("  ML SIGNAL SCORER - RESULTS COMPARISON (OUT-OF-SAMPLE)")
        print("=" * 70)

        split = results.get("split", {})
        if split:
            print(f"\n  Walk-forward split:")
            print(f"    Train: {split.get('train_bars', '?')} bars, "
                  f"{split.get('train_trades', '?')} trades (model trained here)")
            print(f"    Test:  {split.get('test_bars', '?')} bars, "
                  f"{split.get('test_trades', '?')} trades (model NEVER saw this)")

        baseline = results.get("baseline", {})
        ml = results.get("ml_enhanced")
        ml_metrics = results.get("ml_metrics")
        improvement = results.get("improvement")

        # ML training metrics
        if ml_metrics:
            print(f"\n  --- ML MODEL TRAINING ---")
            print(f"  {'Training samples:':<25} {ml_metrics['n_samples']}")
            print(f"  {'CV Accuracy:':<25} {ml_metrics['cv_accuracy']}% (+/- {ml_metrics['cv_std']}%)")
            print(f"  {'Features used:':<25} {ml_metrics['n_features']}")
            print(f"\n  Top predictive features:")
            for feat, imp in ml_metrics.get("top_features", [])[:7]:
                bar = "#" * int(imp * 100)
                print(f"    {feat:<25} {imp:.3f} {bar}")

        # Side by side comparison
        if ml and baseline.get("total_trades", 0) > 0:
            print(f"\n  {'-'*60}")
            print(f"  {'Metric':<25} {'Baseline':>12} {'+ ML':>12} {'Change':>12}")
            print(f"  {'-'*60}")

            metrics = [
                ("Total Trades", "total_trades", "", False),
                ("Win Rate", "win_rate", "%", True),
                ("Total PnL", "total_pnl", "$", True),
                ("Return", "return_pct", "%", True),
                ("Profit Factor", "profit_factor", "", True),
                ("Max Drawdown", "max_drawdown_pct", "%", False),
                ("Avg Win", "avg_win", "$", True),
                ("Avg Loss", "avg_loss", "$", False),
            ]

            for label, key, suffix, higher_better in metrics:
                b_val = baseline.get(key, 0)
                m_val = ml.get(key, 0)
                diff = m_val - b_val

                if suffix == "$":
                    b_str = f"${b_val:.2f}"
                    m_str = f"${m_val:.2f}"
                    d_str = f"${diff:+.2f}"
                elif suffix == "%":
                    b_str = f"{b_val:.1f}%"
                    m_str = f"{m_val:.1f}%"
                    d_str = f"{diff:+.1f}%"
                else:
                    b_str = f"{b_val}"
                    m_str = f"{m_val}"
                    d_str = f"{diff:+.1f}" if isinstance(diff, float) else f"{diff:+d}"

                # Color indicator
                if key == "max_drawdown_pct":
                    indicator = " OK" if diff < 0 else " XX"
                elif higher_better:
                    indicator = " OK" if diff > 0 else " XX"
                else:
                    indicator = ""

                print(f"  {label:<25} {b_str:>12} {m_str:>12} {d_str:>10}{indicator}")

            print(f"  {'-'*60}")

            # ML strategy breakdown
            if ml.get("strategy_breakdown"):
                print(f"\n  --- ML-Enhanced Performance by Strategy ---")
                for strat, data in ml["strategy_breakdown"].items():
                    total = data["wins"] + data["losses"]
                    wr = data["wins"] / total * 100 if total > 0 else 0
                    print(f"  {strat:>25}: {total} trades | "
                          f"Win rate: {wr:.0f}% | PnL: ${data['pnl']:+.2f}")

        elif not ml:
            print(f"\n  Not enough trades to train ML model.")
            print(f"  Baseline had {baseline.get('total_trades', 0)} trades.")
            print(f"  Need at least 10+ trades for ML training.")

        # Final verdict
        print(f"\n  {'='*60}")
        if ml and improvement:
            wr_change = improvement.get("win_rate_change", 0)
            pnl_change = improvement.get("pnl_change", 0)
            dd_change = improvement.get("drawdown_change", 0)

            if wr_change > 0 and pnl_change > 0:
                print(f"  VERDICT: ML IMPROVED the system!")
                print(f"  Win rate: +{wr_change:.1f}% | PnL: ${pnl_change:+.2f} | DD: {dd_change:+.1f}%")
            elif pnl_change > 0:
                print(f"  VERDICT: ML improved PnL but win rate changed by {wr_change:+.1f}%")
            else:
                print(f"  VERDICT: ML needs more data to improve (only {baseline.get('total_trades', 0)} trades)")
                print(f"  The model will improve with more training data over time.")

        print(f"  {'='*60}\n")
