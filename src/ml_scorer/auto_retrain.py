"""
Auto-Retraining Module
Collects live trade results and retrains the ML model weekly.
This is how the AI "learns by itself" over time.
"""

import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import MetaTrader5 as mt5
from src.data_collector.mt5_collector import MT5Connector, DataCollector
from src.feature_engine.engine import FeatureEngine
from src.strategies.strategy_manager import StrategyManager
from src.risk_manager.signal_filter import SignalFilter
from src.risk_manager.risk_manager import RiskManager
from src.ml_scorer.feature_builder import FeatureBuilder
from src.ml_scorer.model import MLSignalScorer
from src.utils.config import get_settings, PROJECT_ROOT
from src.utils.logger import setup_logger

logger = setup_logger("auto_retrain")


class AutoRetrainer:
    """
    Handles automatic ML model retraining.
    Collects trade data, trains new model, compares with old, and replaces if better.
    """

    def __init__(self):
        self.settings = get_settings()
        self.model_dir = PROJECT_ROOT / "data" / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.trade_log_path = PROJECT_ROOT / "data" / "live_trades.json"
        self.retrain_log_path = PROJECT_ROOT / "data" / "retrain_history.json"

    def collect_closed_trades(self) -> list:
        """
        Collect closed trade data from MT5 trade history.
        These are REAL results from the demo account.
        """
        if not mt5.initialize():
            logger.error("  Cannot connect to MT5 for trade history")
            return []

        # Get trade history for last 30 days
        from_date = datetime.now() - timedelta(days=30)
        to_date = datetime.now()

        # Get deals (closed trades)
        deals = mt5.history_deals_get(from_date, to_date)
        mt5.shutdown()

        if deals is None or len(deals) == 0:
            logger.info("  No closed trades found in history")
            return []

        # Filter for our bot's trades (magic number)
        magic = self.settings.get("execution", {}).get("magic_number", 123456)
        symbol = self.settings.get("symbol", "XAUUSD")

        trades = []
        for deal in deals:
            if deal.magic == magic and deal.symbol == symbol:
                trades.append({
                    "ticket": deal.order,
                    "direction": "BUY" if deal.type == 0 else "SELL",
                    "volume": deal.volume,
                    "price": deal.price,
                    "profit": deal.profit,
                    "commission": deal.commission,
                    "swap": deal.swap,
                    "time": datetime.fromtimestamp(deal.time).isoformat(),
                    "comment": deal.comment,
                })

        logger.info(f"  Collected {len(trades)} closed trades from MT5 history")
        return trades

    def retrain(self, use_backtest_data: bool = True) -> dict:
        """
        Retrain the ML model with latest data.

        Args:
            use_backtest_data: If True, use backtest simulation data.
                              If False, use only live trade data (needs 50+ trades).

        Returns:
            Dict with retraining results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"ML MODEL RETRAINING - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        logger.info(f"{'='*60}")

        # Step 1: Connect and get fresh data
        logger.info("  Step 1: Fetching fresh market data...")
        connector = MT5Connector()
        if not connector.connect():
            return {"success": False, "error": "Cannot connect to MT5"}

        collector = DataCollector(connector)
        collector.check_symbol()

        # Fetch H1 data (our primary trading timeframe)
        df = collector.fetch_candles("H1", num_bars=5000)
        connector.disconnect()

        if df.empty:
            return {"success": False, "error": "No data received"}

        # Step 2: Compute features
        logger.info("  Step 2: Computing features...")
        engine = FeatureEngine()
        featured_df = engine.compute_features(df, "H1")

        # Step 3: Generate signals and simulate trades for training data
        logger.info("  Step 3: Generating training data...")
        strategy_manager = StrategyManager()
        signal_filter = SignalFilter()
        risk_manager = RiskManager(initial_balance=100000.0)

        raw_signals = strategy_manager.scan_all(featured_df, "H1")
        filtered_signals = signal_filter.filter_signals(raw_signals, "H1")

        # Simulate trades to get outcomes
        signal_map = {}
        for sig in filtered_signals:
            if sig.timestamp not in signal_map:
                signal_map[sig.timestamp] = []
            signal_map[sig.timestamp].append(sig)

        for i in range(1, len(featured_df)):
            current_bar = featured_df.iloc[i]
            current_time = featured_df.index[i]

            current_date = current_time.date() if hasattr(current_time, 'date') else None
            if current_date:
                risk_manager.reset_daily(current_date)

            risk_manager.check_exits(
                current_high=current_bar["high"],
                current_low=current_bar["low"],
                current_close=current_bar["close"],
                current_time=current_time
            )

            if current_time in signal_map:
                for signal in signal_map[current_time]:
                    allowed, _ = risk_manager.can_trade(signal)
                    if allowed:
                        sizing = risk_manager.calculate_position_size(signal)
                        if sizing["lot_size"] > 0:
                            risk_manager.open_trade(signal, sizing)

        # Close remaining
        if risk_manager.open_positions and len(featured_df) > 0:
            last_bar = featured_df.iloc[-1]
            for pos in risk_manager.open_positions[:]:
                if pos["direction"] == "BUY":
                    pnl = (last_bar["close"] - pos["entry_price"]) * pos["lot_size"] * 100
                else:
                    pnl = (pos["entry_price"] - last_bar["close"]) * pos["lot_size"] * 100
                pos["pnl"] = round(pnl, 2)
                pos["status"] = "forced_close"
                risk_manager.current_balance += pnl
                risk_manager.trade_history.append(pos)
                risk_manager.open_positions.remove(pos)

        trades = risk_manager.trade_history
        logger.info(f"  Generated {len(trades)} simulated trades for training")

        if len(trades) < 20:
            logger.warning(f"  Only {len(trades)} trades - need at least 20 for retraining")
            return {"success": False, "error": f"Only {len(trades)} trades, need 20+"}

        # Step 4: Train new model
        logger.info("  Step 4: Training new ML model...")
        feature_builder = FeatureBuilder()
        X, y, feature_names = feature_builder.build_training_data(trades, featured_df)

        new_model = MLSignalScorer()
        new_metrics = new_model.train(X, y, feature_names)

        # Step 5: Compare with old model
        logger.info("  Step 5: Comparing with current model...")
        old_model = MLSignalScorer()
        old_loaded = old_model.load()

        should_replace = True
        comparison = {}

        if old_loaded:
            # Test both models on the same data
            old_preds = old_model.model.predict_proba(X)[:, 1]
            new_preds = new_model.model.predict_proba(X)[:, 1]

            old_accuracy = np.mean((old_preds > 0.5) == y) * 100
            new_accuracy = new_metrics["train_accuracy"]

            comparison = {
                "old_accuracy": round(old_accuracy, 1),
                "new_accuracy": round(new_accuracy, 1),
                "improvement": round(new_accuracy - old_accuracy, 1),
            }

            logger.info(f"  Old model accuracy: {old_accuracy:.1f}%")
            logger.info(f"  New model accuracy: {new_accuracy:.1f}%")

            # Only replace if new model is better (or at least not worse)
            if new_accuracy < old_accuracy - 5:
                should_replace = False
                logger.warning(f"  New model is worse! Keeping old model.")
        else:
            logger.info("  No old model found - saving new model as first version")

        # Step 6: Save if better
        if should_replace:
            # Backup old model
            old_path = self.model_dir / "xgb_signal_scorer.pkl"
            if old_path.exists():
                backup_name = f"xgb_signal_scorer_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
                old_path.rename(self.model_dir / backup_name)
                logger.info(f"  Old model backed up as {backup_name}")

            new_model.save()
            logger.info(f"  New model saved!")

        # Step 7: Log retraining
        retrain_record = {
            "timestamp": datetime.now().isoformat(),
            "trades_used": len(trades),
            "win_rate": round(np.mean(y) * 100, 1),
            "cv_accuracy": new_metrics["cv_accuracy"],
            "train_accuracy": new_metrics["train_accuracy"],
            "replaced": should_replace,
            "comparison": comparison,
            "top_features": [(f, round(i, 4)) for f, i in new_metrics["top_features"][:5]],
        }

        self._save_retrain_log(retrain_record)

        logger.info(f"\n{'='*60}")
        logger.info(f"RETRAINING COMPLETE")
        logger.info(f"  Trades used: {len(trades)}")
        logger.info(f"  CV Accuracy: {new_metrics['cv_accuracy']}%")
        logger.info(f"  Model replaced: {should_replace}")
        logger.info(f"{'='*60}")

        return {
            "success": True,
            "trades_used": len(trades),
            "metrics": new_metrics,
            "comparison": comparison,
            "replaced": should_replace,
        }

    def _save_retrain_log(self, record: dict):
        """Append to retraining history."""
        history = []
        if self.retrain_log_path.exists():
            try:
                with open(self.retrain_log_path) as f:
                    history = json.load(f)
            except Exception:
                pass

        history.append(record)

        with open(self.retrain_log_path, "w") as f:
            json.dump(history, f, indent=2, default=str)

    def get_retrain_history(self) -> list:
        """Get history of all retraining events."""
        if self.retrain_log_path.exists():
            with open(self.retrain_log_path) as f:
                return json.load(f)
        return []
