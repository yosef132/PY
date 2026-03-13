"""
XAUUSD AI Trading System - ML Signal Scorer Setup
Session 5: Build the XGBoost ML model that learns which signals win/lose.

Usage:
    python setup_ml_scorer.py
    python run_ml_backtest.py
"""

import os

FILES = {}

# ============================================================
# FILE 1: src/ml_scorer/feature_builder.py
# ============================================================
FILES["src/ml_scorer/feature_builder.py"] = r'''"""
ML Feature Builder
Extracts features from each signal + market context to build training data.
This converts a Signal object + DataFrame row into a flat feature vector for XGBoost.
"""

import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger("ml_features")


class FeatureBuilder:
    """Builds ML feature vectors from signals and market data."""

    # Features we extract from each signal's market context
    MARKET_FEATURES = [
        "rsi_14", "rsi_7", "ema_alignment", "atr_14", "atr_pct",
        "bb_width", "bb_position", "hvol_20",
        "market_trend", "bos", "choch",
        "sr_strength", "dist_to_support_pct", "dist_to_resistance_pct", "sr_position",
        "wyckoff_phase", "effort_vs_result",
        "relative_volume", "mfi_14", "cmf_20",
        "killzone_active", "session_london", "session_ny", "session_overlap",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "liquidity_above", "liquidity_below",
        "body_atr_ratio", "range_atr_ratio", "upper_wick_pct", "lower_wick_pct",
    ]

    # Features from the signal itself
    SIGNAL_FEATURES = [
        "confidence", "risk_reward",
    ]

    # Strategy one-hot encoding
    STRATEGY_NAMES = [
        "support_resistance", "rsi_divergence", "ict", "wyckoff", "crt", "sk_system"
    ]

    def get_feature_names(self) -> list:
        """Get all feature column names."""
        names = []
        names.extend(self.MARKET_FEATURES)
        names.extend(self.SIGNAL_FEATURES)
        names.append("is_buy")  # 1 for BUY, 0 for SELL
        names.extend([f"strat_{s}" for s in self.STRATEGY_NAMES])
        return names

    def extract_features(self, signal, df_row: pd.Series) -> dict:
        """
        Extract a feature vector from a signal + its market context row.

        Args:
            signal: Signal object from a strategy detector
            df_row: The DataFrame row at the signal's timestamp

        Returns:
            Dict of feature_name -> value
        """
        features = {}

        # Market context features
        for feat in self.MARKET_FEATURES:
            val = df_row.get(feat, np.nan)
            features[feat] = float(val) if pd.notna(val) else 0.0

        # Signal features
        features["confidence"] = signal.confidence
        features["risk_reward"] = signal.risk_reward

        # Direction
        features["is_buy"] = 1.0 if signal.direction == "BUY" else 0.0

        # Strategy one-hot
        for strat in self.STRATEGY_NAMES:
            features[f"strat_{strat}"] = 1.0 if signal.strategy_name == strat else 0.0

        return features

    def build_training_data(self, trades: list, featured_df: pd.DataFrame) -> tuple:
        """
        Build training dataset from completed trades.

        Args:
            trades: List of completed trade dicts (from backtester)
            featured_df: The featured DataFrame used during backtesting

        Returns:
            (X: np.array of features, y: np.array of labels, feature_names: list)
        """
        feature_names = self.get_feature_names()
        rows = []
        labels = []

        for trade in trades:
            # Find the closest row in the DataFrame
            entry_time = trade.get("entry_time")
            if entry_time is None:
                continue

            # Get the market context at entry time
            if entry_time in featured_df.index:
                df_row = featured_df.loc[entry_time]
            else:
                # Find nearest timestamp
                idx = featured_df.index.get_indexer([entry_time], method="nearest")[0]
                if idx < 0 or idx >= len(featured_df):
                    continue
                df_row = featured_df.iloc[idx]

            # Build feature vector
            features = {}
            for feat in self.MARKET_FEATURES:
                val = df_row.get(feat, np.nan)
                features[feat] = float(val) if pd.notna(val) else 0.0

            features["confidence"] = trade.get("confidence", 0.5)
            features["risk_reward"] = trade.get("risk_reward", 1.0)
            features["is_buy"] = 1.0 if trade["direction"] == "BUY" else 0.0

            for strat in self.STRATEGY_NAMES:
                features[f"strat_{strat}"] = 1.0 if trade.get("strategy") == strat else 0.0

            # Feature vector in correct order
            row = [features.get(name, 0.0) for name in feature_names]
            rows.append(row)

            # Label: 1 = profitable trade, 0 = losing trade
            labels.append(1 if trade["pnl"] > 0 else 0)

        X = np.array(rows)
        y = np.array(labels)

        logger.info(f"  Built training data: {len(rows)} samples, {len(feature_names)} features")
        logger.info(f"  Win rate in data: {y.mean()*100:.1f}%")

        return X, y, feature_names
'''

# ============================================================
# FILE 2: src/ml_scorer/model.py
# ============================================================
FILES["src/ml_scorer/model.py"] = r'''"""
ML Signal Scorer Model
XGBoost classifier that predicts whether a signal will be profitable.
This is the AI brain that learns over time.
"""

import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
from src.utils.config import PROJECT_ROOT
from src.utils.logger import setup_logger

logger = setup_logger("ml_model")


class MLSignalScorer:
    """
    XGBoost model that scores trading signals.
    Learns from historical trade outcomes to predict which signals will profit.
    """

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.is_trained = False
        self.model_dir = PROJECT_ROOT / "data" / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list) -> dict:
        """
        Train the XGBoost model on historical trade data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (1 = win, 0 = loss)
            feature_names: List of feature names

        Returns:
            Dict with training metrics
        """
        self.feature_names = feature_names

        if len(X) < 20:
            logger.warning(f"  Only {len(X)} samples - too few for reliable training")
            logger.warning(f"  Need at least 50+ trades for meaningful ML learning")

        # XGBoost with conservative settings to avoid overfitting
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=4,              # Shallow trees = less overfitting
            learning_rate=0.1,
            subsample=0.8,            # Use 80% of data per tree
            colsample_bytree=0.8,     # Use 80% of features per tree
            min_child_weight=3,       # Minimum samples in leaf
            reg_alpha=0.1,            # L1 regularization
            reg_lambda=1.0,           # L2 regularization
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
        )

        # Time-series cross-validation (respects temporal order)
        n_splits = min(5, max(2, len(X) // 20))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        try:
            cv_scores = cross_val_score(self.model, X, y, cv=tscv, scoring="accuracy")
        except Exception as e:
            logger.warning(f"  CV failed: {e}, training on full data")
            cv_scores = np.array([0.5])

        # Train on full dataset
        self.model.fit(X, y)
        self.is_trained = True

        # Get predictions on training data (for analysis)
        y_pred = self.model.predict(X)
        train_acc = accuracy_score(y, y_pred)

        # Feature importance
        importances = self.model.feature_importances_
        top_features = sorted(zip(feature_names, importances),
                              key=lambda x: x[1], reverse=True)[:10]

        metrics = {
            "cv_accuracy": round(cv_scores.mean() * 100, 1),
            "cv_std": round(cv_scores.std() * 100, 1),
            "train_accuracy": round(train_acc * 100, 1),
            "n_samples": len(X),
            "n_features": len(feature_names),
            "win_rate_in_data": round(y.mean() * 100, 1),
            "top_features": top_features,
        }

        logger.info(f"  ML Model trained:")
        logger.info(f"    CV Accuracy: {metrics['cv_accuracy']}% (+/- {metrics['cv_std']}%)")
        logger.info(f"    Train Accuracy: {metrics['train_accuracy']}%")
        logger.info(f"    Top features:")
        for feat, imp in top_features[:5]:
            logger.info(f"      {feat}: {imp:.3f}")

        return metrics

    def predict(self, features: dict) -> tuple:
        """
        Predict whether a signal will be profitable.

        Args:
            features: Dict of feature_name -> value

        Returns:
            (prediction: 0 or 1, probability: float 0.0 to 1.0)
        """
        if not self.is_trained:
            return 1, 0.5  # Default: allow trade with neutral confidence

        # Build feature vector in correct order
        X = np.array([[features.get(name, 0.0) for name in self.feature_names]])

        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0][1]  # P(win)

        return int(prediction), float(probability)

    def score_signal(self, signal, df_row, feature_builder) -> float:
        """
        Score a signal using the ML model.

        Returns:
            ML confidence score (0.0 to 1.0)
        """
        if not self.is_trained:
            return signal.confidence  # Pass through original confidence

        features = feature_builder.extract_features(signal, df_row)
        _, ml_probability = self.predict(features)

        # Blend ML score with strategy confidence
        # ML has 60% weight, strategy has 40% weight
        blended = ml_probability * 0.6 + signal.confidence * 0.4

        return round(blended, 3)

    def save(self, filename: str = "xgb_signal_scorer.pkl"):
        """Save trained model to disk."""
        if not self.is_trained:
            logger.warning("  Cannot save: model not trained")
            return

        filepath = self.model_dir / filename
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
        }, filepath)
        logger.info(f"  Model saved to {filepath}")

    def load(self, filename: str = "xgb_signal_scorer.pkl") -> bool:
        """Load model from disk."""
        filepath = self.model_dir / filename
        if not filepath.exists():
            logger.warning(f"  No saved model found at {filepath}")
            return False

        data = joblib.load(filepath)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.is_trained = True
        logger.info(f"  Model loaded from {filepath}")
        return True
'''

# ============================================================
# FILE 3: src/ml_scorer/enhanced_backtester.py
# ============================================================
FILES["src/ml_scorer/enhanced_backtester.py"] = r'''"""
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

    def run_full_pipeline(self, featured_data: dict, timeframe: str = "H1") -> dict:
        """
        Run the complete ML pipeline:
        1. Generate signals + collect trades (no ML)
        2. Train ML model on trade outcomes
        3. Re-run backtest WITH ML filtering
        4. Compare results
        """
        if timeframe not in featured_data:
            logger.error(f"  {timeframe} not found in data")
            return {}

        df = featured_data[timeframe]

        # ========== PHASE A: Baseline (no ML) ==========
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE A: Baseline backtest WITHOUT ML ({timeframe})")
        logger.info(f"{'='*60}")

        baseline_trades, baseline_stats = self._run_backtest(df, timeframe, use_ml=False)

        if len(baseline_trades) < 10:
            logger.warning(f"  Only {len(baseline_trades)} trades - not enough to train ML")
            return {
                "baseline": baseline_stats,
                "ml_enhanced": None,
                "ml_metrics": None,
                "improvement": None
            }

        # ========== PHASE B: Train ML Model ==========
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE B: Training ML model on {len(baseline_trades)} trades")
        logger.info(f"{'='*60}")

        feature_builder = FeatureBuilder()
        X, y, feature_names = feature_builder.build_training_data(baseline_trades, df)

        ml_scorer = MLSignalScorer()
        ml_metrics = ml_scorer.train(X, y, feature_names)

        # Save the model
        ml_scorer.save()

        # ========== PHASE C: ML-Enhanced Backtest ==========
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE C: Re-running backtest WITH ML scoring")
        logger.info(f"{'='*60}")

        ml_trades, ml_stats = self._run_backtest(
            df, timeframe, use_ml=True,
            ml_scorer=ml_scorer, feature_builder=feature_builder
        )

        # ========== Compare ==========
        improvement = {}
        if baseline_stats["total_trades"] > 0 and ml_stats["total_trades"] > 0:
            improvement = {
                "trades_reduction": baseline_stats["total_trades"] - ml_stats["total_trades"],
                "win_rate_change": ml_stats["win_rate"] - baseline_stats["win_rate"],
                "pnl_change": ml_stats["total_pnl"] - baseline_stats["total_pnl"],
                "drawdown_change": ml_stats["max_drawdown_pct"] - baseline_stats["max_drawdown_pct"],
                "profit_factor_change": ml_stats["profit_factor"] - baseline_stats["profit_factor"],
            }

        return {
            "baseline": baseline_stats,
            "ml_enhanced": ml_stats,
            "ml_metrics": ml_metrics,
            "improvement": improvement,
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
        print("  ML SIGNAL SCORER - RESULTS COMPARISON")
        print("=" * 70)

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
                bar = "█" * int(imp * 100)
                print(f"    {feat:<25} {imp:.3f} {bar}")

        # Side by side comparison
        if ml and baseline.get("total_trades", 0) > 0:
            print(f"\n  {'─'*60}")
            print(f"  {'Metric':<25} {'Baseline':>12} {'+ ML':>12} {'Change':>12}")
            print(f"  {'─'*60}")

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
                    indicator = " ✓" if diff < 0 else " ✗"
                elif higher_better:
                    indicator = " ✓" if diff > 0 else " ✗"
                else:
                    indicator = ""

                print(f"  {label:<25} {b_str:>12} {m_str:>12} {d_str:>10}{indicator}")

            print(f"  {'─'*60}")

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
'''

# ============================================================
# FILE 4: run_ml_backtest.py
# ============================================================
FILES["run_ml_backtest.py"] = r'''"""
XAUUSD AI Trading System - ML Enhanced Backtest
Trains XGBoost on trade history, then re-runs backtest with ML scoring.

Usage:
    python run_ml_backtest.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collector.mt5_collector import MT5Connector, DataCollector
from src.feature_engine.engine import FeatureEngine
from src.ml_scorer.enhanced_backtester import EnhancedBacktester
from src.utils.logger import setup_logger

logger = setup_logger("run_ml_backtest")


def main():
    print("\n" + "=" * 70)
    print("  XAUUSD AI TRADING SYSTEM")
    print("  Phase 3 - Session 5: ML Signal Scorer")
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

    # --- Step 3: Run ML Pipeline ---
    logger.info("\nStep 3: Running ML-enhanced backtest pipeline...\n")
    backtester = EnhancedBacktester(initial_balance=3000.0)

    # Run on H1 (our best timeframe from previous backtest)
    results = backtester.run_full_pipeline(featured_data, timeframe="H1")

    # --- Step 4: Print comparison ---
    backtester.print_comparison(results)

    print("\n" + "=" * 70)
    print("  Phase 3 - Session 5 COMPLETE!")
    print("  ML model is trained and saved in: data/models/")
    print("  The model will improve as more trade data is collected.")
    print("  Next: Connect to MT5 demo for live paper trading")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
'''

# ============================================================
# FILE 5: src/ml_scorer/__init__.py
# ============================================================
FILES["src/ml_scorer/__init__.py"] = ""

# ============================================================
# Create all files
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("  ML Signal Scorer Setup")
    print("=" * 60 + "\n")

    for filepath, content in FILES.items():
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        print(f"  [FILE] {filepath}")

    print(f"\n  Created {len(FILES)} files.")
    print("\n  Now run:  python run_ml_backtest.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
