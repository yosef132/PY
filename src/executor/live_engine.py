"""
Live Trading Engine (v4)
Added: Binance exchange cross-reference signal, improved news filter.

Changes from v3:
- Added ExchangeCollector for Binance XAUUSDT futures data
- Exchange bias adjusts signal confidence (+/-0.05)
- Exchange data refreshes every 5 minutes with each scan
"""

import MetaTrader5 as mt5
import time
import json
from datetime import datetime, timezone, date
from pathlib import Path
from src.data_collector.mt5_collector import MT5Connector, DataCollector
from src.data_collector.exchange_collector import ExchangeCollector
from src.feature_engine.engine import FeatureEngine
from src.strategies.strategy_manager import StrategyManager
from src.strategies.mtf_filter import MTFFilter
from src.risk_manager.signal_filter import SignalFilter
from src.risk_manager.risk_manager import RiskManager
from src.ml_scorer.model import MLSignalScorer
from src.ml_scorer.feature_builder import FeatureBuilder
from src.news_filter.news_manager import NewsManager
from src.executor.mt5_executor import MT5Executor
from src.executor.trailing_stop import TrailingStopManager
from src.alerts.telegram_alerts import TelegramAlerts
from src.utils.config import get_settings, PROJECT_ROOT
from src.utils.logger import setup_logger

logger = setup_logger("live_engine")

# ===== TUNABLE THRESHOLDS (defaults — overridden by config at init) =====
ML_THRESHOLD = 0.55          # ML confidence threshold (default)
NEWS_BLACKOUT_MINUTES = 30   # Blackout window around high-impact news (default)
MAX_SIGNALS_PER_SCAN = 3     # Max trades to attempt per scan
# ========================================================================


class LiveTradingEngine:
    """
    Live trading loop v4 - With Binance cross-reference.
    """

    def __init__(self):
        self.settings = get_settings()
        loop_cfg = self.settings.get("loop", {})
        self.scan_interval = loop_cfg.get("scan_interval_seconds", 300)
        self.news_refresh_interval = loop_cfg.get("news_refresh_seconds", 900)
        self.h4_refresh_interval = loop_cfg.get("h4_refresh_seconds", 3600)
        self.last_news_refresh = 0
        self.last_h4_refresh = 0

        # Read thresholds from config (override module-level defaults)
        global ML_THRESHOLD, NEWS_BLACKOUT_MINUTES
        ML_THRESHOLD = self.settings.get("ml", {}).get("confidence_threshold", 0.55)
        NEWS_BLACKOUT_MINUTES = self.settings.get("risk", {}).get("news_blackout_minutes", 30)

        # Core components
        self.connector = MT5Connector()
        self.feature_engine = FeatureEngine()
        self.strategy_manager = StrategyManager()
        self.signal_filter = SignalFilter()
        self.risk_manager = RiskManager(initial_balance=0)
        self.executor = MT5Executor()
        self.news_manager = NewsManager()
        self.mtf_filter = MTFFilter()
        self.telegram = TelegramAlerts()
        self.exchange = ExchangeCollector()

        # Trailing stop manager
        from src.executor.trailing_stop import TrailingStopManager
        self.trailing_manager = TrailingStopManager()

        # ML model
        self.ml_scorer = MLSignalScorer()
        self.feature_builder = FeatureBuilder()
        self.ml_loaded = self.ml_scorer.load()

        # Shadow mode — analyse real market but never execute orders
        self.shadow_mode = self.settings.get("execution", {}).get("shadow_mode", False)

        # Trade log
        self.trade_log_path = PROJECT_ROOT / "data" / "live_trades.json"
        self.trade_log = self._load_trade_log()

        # Shadow trade log (virtual trades when shadow_mode=True)
        self.shadow_log_path = PROJECT_ROOT / "data" / "shadow_trades.json"
        self.shadow_log = self._load_shadow_log()

        # Rejection log (for debugging)
        self.rejection_log_path = PROJECT_ROOT / "data" / "rejected_signals.json"
        self.rejection_log = []

        # State
        self.running = False
        self.total_scans = 0
        self.total_trades_placed = 0
        self.total_shadow_signals = 0
        self.last_retrain_date = None   # tracks Sunday retraining
        self.session_start_balance = 0

    def _load_trade_log(self) -> list:
        if self.trade_log_path.exists():
            try:
                with open(self.trade_log_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_trade_log(self):
        self.trade_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.trade_log_path, "w") as f:
            json.dump(self.trade_log, f, indent=2, default=str)

    def _load_shadow_log(self) -> list:
        if self.shadow_log_path.exists():
            try:
                with open(self.shadow_log_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_shadow_log(self):
        self.shadow_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.shadow_log_path, "w") as f:
            json.dump(self.shadow_log, f, indent=2, default=str)

    def _save_rejection_log(self):
        self.rejection_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.rejection_log_path, "w") as f:
            json.dump(self.rejection_log[-500:], f, indent=2, default=str)  # Keep last 500

    def start(self):
        mode_label = "SHADOW MODE (read-only)" if self.shadow_mode else "LIVE PAPER TRADING v4"
        print("\n" + "=" * 70)
        print(f"  XAUUSD AI TRADING SYSTEM - {mode_label}")
        print(f"  MTF: H4+H1 | ML threshold: {ML_THRESHOLD}")
        print(f"  News blackout: {NEWS_BLACKOUT_MINUTES} min | Telegram: ON")
        if self.shadow_mode:
            print("  *** SHADOW MODE: signals logged but NO orders will be placed ***")
        print("  Press Ctrl+C to stop")
        print("=" * 70 + "\n")

        if not self.connector.connect():
            logger.error("Cannot connect to MT5!")
            self.telegram.send(
                "\U0001F6A8 <b>BOT FAILED TO START</b>\n"
                "Could not connect to MetaTrader 5.\n"
                "Check MT5 is running and credentials are correct.\n"
                f"Time: {datetime.now().strftime('%H:%M:%S')}"
            )
            return

        account = self.connector.get_account_info()
        self.risk_manager.initial_balance = account["balance"]
        self.risk_manager.current_balance = account["balance"]
        self.risk_manager.peak_balance = account["balance"]
        self.session_start_balance = account["balance"]

        logger.info(f"  Account: {account['login']} | Balance: ${account['balance']:.2f}")
        logger.info(f"  ML Model: {'Loaded' if self.ml_loaded else 'Not loaded'}")
        logger.info(f"  ML Threshold: {ML_THRESHOLD}")
        logger.info(f"  MTF Filter: H4 trend confirmation ACTIVE")
        logger.info(f"  Telegram: {'ACTIVE' if self.telegram.enabled else 'DISABLED'}")

        mode_tag = "[SHADOW - READ ONLY]" if self.shadow_mode else "[PAPER TRADING v4]"
        self.telegram.send(
            f"\U0001F680 <b>BOT STARTED {mode_tag}</b>\n"
            f"Account: {account['login']}\n"
            f"Balance: ${account['balance']:.2f}\n"
            f"ML threshold: {ML_THRESHOLD}\n"
            f"News blackout: {NEWS_BLACKOUT_MINUTES} min\n"
            + ("\U0001F441 Shadow mode: signals only, NO execution\n" if self.shadow_mode else "") +
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )

        # Pre-fetch Binance data so first scan has exchange context
        try:
            self.exchange.fetch_all()
            logger.info("  Binance data: pre-fetched OK")
        except Exception as e:
            logger.warning(f"  Binance pre-fetch failed (non-fatal): {e}")

        self._refresh_news()
        self._refresh_h4_trend()

        self._startup_health_check(account)

        self.running = True

        try:
            while self.running:
                self._check_sunday_retrain()
                self._scan_and_trade()
                self.total_scans += 1
                self._print_status()

                logger.info(f"  Next scan in {self.scan_interval} seconds... (Ctrl+C to stop)")
                time.sleep(self.scan_interval)

        except KeyboardInterrupt:
            logger.info("\n  Stopping live trading...")
            self._shutdown()

    def _startup_health_check(self, account: dict):
        """Print a health checklist before the main loop starts."""
        print("\n" + "=" * 50)
        print("  STARTUP HEALTH CHECK")
        print("=" * 50)

        # MT5 connected + account type
        trade_mode = str(account.get("trade_mode", "")).lower()
        is_demo = "demo" in trade_mode or trade_mode == "0"
        is_safe = is_demo or self.shadow_mode   # shadow mode on real account is safe
        acct_type = "DEMO" if is_demo else ("LIVE - SHADOW" if self.shadow_mode else "LIVE (!)")
        print(f"  [{'OK' if is_safe else 'WW'}] MT5 connected ({acct_type}) - Account {account.get('login')}")
        if self.shadow_mode:
            print(f"  [OK] Shadow mode ACTIVE - no orders will be placed")

        # ML model loaded
        print(f"  [{'OK' if self.ml_loaded else 'WW'}] ML model {'loaded' if self.ml_loaded else 'NOT loaded — signals use raw confidence'}")

        # Parquet files
        data_dir = PROJECT_ROOT / "data"
        parquet_files = list(data_dir.glob("*.parquet")) if data_dir.exists() else []
        print(f"  [{'OK' if parquet_files else 'WW'}] Data files: {len(parquet_files)} parquet files found")

        # .env credentials
        import os
        mt5_login_set = bool(os.getenv("MT5_LOGIN"))
        telegram_set = bool(os.getenv("TELEGRAM_TOKEN"))
        print(f"  [{'OK' if mt5_login_set else 'XX'}] MT5 credentials loaded from .env")
        print(f"  [{'OK' if telegram_set else 'WW'}] Telegram token {'loaded' if telegram_set else 'NOT set — alerts disabled'}")

        # Binance data
        binance_ok = bool(self.exchange.last_data)
        print(f"  [{'OK' if binance_ok else 'WW'}] Binance data {'available' if binance_ok else 'not available (non-fatal)'}")

        # Thresholds summary
        print(f"\n  Config:")
        print(f"    ML threshold    : {ML_THRESHOLD}")
        print(f"    News blackout   : {NEWS_BLACKOUT_MINUTES} min")
        print(f"    Risk per trade  : {self.settings.get('risk', {}).get('risk_per_trade_pct', 0.5)}%")
        print(f"    Max daily loss  : {self.settings.get('risk', {}).get('max_daily_loss_pct', 3)}%")
        print("=" * 50 + "\n")

        if not is_demo and not self.shadow_mode:
            self.telegram.send(
                "\U000026A0 <b>WARNING: Bot running on LIVE account!</b>\n"
                "Switch to demo account or enable shadow_mode before real trading."
            )

    def _check_sunday_retrain(self):
        """Auto-retrain the ML model every Sunday. Runs once per Sunday."""
        today = datetime.now().date()
        is_sunday = datetime.now().weekday() == 6
        already_ran = self.last_retrain_date == today

        if not is_sunday or already_ran:
            return

        logger.info("  Sunday auto-retrain triggered...")
        self.last_retrain_date = today

        try:
            from src.ml_scorer.auto_retrain import AutoRetrainer
            retrainer = AutoRetrainer()
            result = retrainer.retrain()

            if result.get("success"):
                msg = (
                    f"\U0001F9E0 <b>ML Model Retrained</b>\n"
                    f"Trades used: {result['trades_used']}\n"
                    f"CV Accuracy: {result['metrics']['cv_accuracy']}%\n"
                    f"Model replaced: {result['replaced']}"
                )
                # Reload model if it was replaced
                if result["replaced"]:
                    self.ml_scorer.load()
                    self.ml_loaded = True
                    logger.info("  New model loaded into live engine.")
            else:
                msg = f"\U000026A0 ML Retrain skipped: {result.get('error', 'unknown')}"

            self.telegram.send(msg)
            logger.info(f"  Auto-retrain complete: {result}")

        except Exception as e:
            logger.error(f"  Auto-retrain failed: {e}")
            self.telegram.send(f"\U000026A0 ML auto-retrain failed: {e}")

    def _update_trade_outcomes(self):
        """Check MT5 for positions that closed since last scan and update live_trades.json."""
        open_trades = [t for t in self.trade_log if t.get("result") == "open" and t.get("ticket")]
        if not open_trades:
            return

        try:
            positions = mt5.positions_get()
            open_tickets = {p.ticket for p in positions} if positions else set()
        except Exception as e:
            logger.warning(f"  Outcome check failed: {e}")
            return

        updated = 0
        for trade in open_trades:
            ticket = trade["ticket"]
            if ticket in open_tickets:
                continue  # Still open

            try:
                deals = mt5.history_deals_get(position=ticket)
                if not deals:
                    continue
                exit_deal = next((d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT), None)
                if not exit_deal:
                    continue

                pnl = exit_deal.profit
                trade["result"] = "win" if pnl > 0 else "loss"
                trade["exit_price"] = round(exit_deal.price, 2)
                trade["exit_time"] = datetime.fromtimestamp(exit_deal.time).isoformat()
                trade["pnl"] = round(pnl, 2)
                updated += 1
                logger.info(f"  Trade #{ticket} closed: {trade['result']} ${pnl:+.2f}")
            except Exception as e:
                logger.warning(f"  Could not get outcome for ticket {ticket}: {e}")

        if updated > 0:
            self._save_trade_log()
            logger.info(f"  {updated} trade outcome(s) updated in live_trades.json")

    def _refresh_h4_trend(self):
        try:
            logger.info("  Refreshing H4/D1 trend analysis...")
            collector = DataCollector(self.connector)
            featured_data = {}

            h4_df = collector.fetch_candles("H4", num_bars=200)
            if not h4_df.empty:
                featured_data["H4"] = self.feature_engine.compute_features(h4_df, "H4")

            d1_df = collector.fetch_candles("D1", num_bars=100)
            if not d1_df.empty:
                featured_data["D1"] = self.feature_engine.compute_features(d1_df, "D1")

            self.mtf_filter.analyze_higher_timeframes(featured_data)
            self.last_h4_refresh = time.time()

            status = self.mtf_filter.get_status()
            logger.info(f"  MTF: H4={status['h4_trend']} ({status['h4_strength']:.2f}) | D1={status['d1_trend']}")

        except Exception as e:
            logger.warning(f"  H4/D1 refresh failed: {e}")

    def _scan_and_trade(self):
        scan_time = datetime.now()
        logger.info(f"\n{'='*50}")
        logger.info(f"SCAN #{self.total_scans + 1} at {scan_time.strftime('%H:%M:%S')}")
        logger.info(f"{'='*50}")

        # Daily reset — clears daily_trades and daily_loss counters at midnight
        self.risk_manager.reset_daily(scan_time.date())

        # Update outcomes for any trades closed since last scan
        self._update_trade_outcomes()

        # Periodic refreshes
        if time.time() - self.last_news_refresh > self.news_refresh_interval:
            self._refresh_news()
        if time.time() - self.last_h4_refresh > self.h4_refresh_interval:
            self._refresh_h4_trend()

        # Refresh Binance cross-reference every scan
        try:
            self.exchange.fetch_all()
        except Exception as e:
            logger.warning(f"  Binance data fetch failed: {e}")

        # Step 1: Fetch H1 data
        try:
            collector = DataCollector(self.connector)
            df = collector.fetch_candles("H1", num_bars=500)
            if df.empty:
                logger.warning("  No data received")
                return

            # Stale data check — last candle must be within 3 hours
            from datetime import timezone as tz
            last_candle_time = df.index[-1]
            if last_candle_time.tzinfo is None:
                last_candle_time = last_candle_time.replace(tzinfo=tz.utc)
            staleness_hours = (datetime.now(tz.utc) - last_candle_time).total_seconds() / 3600
            if staleness_hours > 3:
                logger.warning(f"  STALE DATA: last candle is {staleness_hours:.1f}h old — skipping scan")
                return
        except Exception as e:
            logger.error(f"  Data fetch error: {e}")
            return

        # Step 2: Features
        try:
            featured_df = self.feature_engine.compute_features(df, "H1")
        except Exception as e:
            logger.error(f"  Feature error: {e}")
            return

        # Step 3: Signals
        raw_signals = self.strategy_manager.scan_all(featured_df, "H1")
        filtered_signals = self.signal_filter.filter_signals(raw_signals, "H1")

        # Recency filter: only use signals from the last 24 H1 bars (last 24 hours).
        # Strategies scan all 500 bars; without this, signals from weeks-old bars
        # would be executed at current price with stale SL/TP levels.
        cutoff_time = featured_df.index[-24]
        recent_signals = [s for s in filtered_signals
                          if s.timestamp is not None and s.timestamp >= cutoff_time]
        logger.info(f"  Signals: {len(raw_signals)} raw -> {len(filtered_signals)} filtered "
                    f"-> {len(recent_signals)} recent (last 24h)")
        filtered_signals = recent_signals
        current_bar = featured_df.iloc[-1]

        if not filtered_signals:
            logger.info(f"  No signals. Waiting...")
            return

        # Step 4: Process signals
        mtf_blocked = 0
        news_blocked = 0
        ml_blocked = 0
        risk_blocked = 0

        for signal in filtered_signals[:MAX_SIGNALS_PER_SCAN]:

            # MTF check (but allow neutral H4 — don't block)
            mtf_allowed, mtf_reason, mtf_boost = self.mtf_filter.filter_signal(signal)
            if not mtf_allowed:
                logger.info(f"  MTF BLOCKED: {mtf_reason}")
                self._log_rejection(scan_time, signal, "mtf", mtf_reason)
                mtf_blocked += 1
                continue

            if mtf_boost > 0:
                old_conf = signal.confidence
                signal.confidence = min(signal.confidence + mtf_boost, 1.0)
                logger.info(f"  MTF boost: {old_conf:.2f} -> {signal.confidence:.2f}")

            # Binance cross-reference boost/penalty
            exchange_boost = self.exchange.get_signal_boost(signal.direction)
            if exchange_boost != 0:
                old_conf = signal.confidence
                signal.confidence = max(0.0, min(signal.confidence + exchange_boost, 1.0))
                logger.info(f"  Exchange boost: {old_conf:.2f} -> {signal.confidence:.2f} "
                            f"(Binance bias={self.exchange.last_data.get('bias', 'neutral')})")

            # News check (with reduced blackout window)
            news_ok, news_reason, news_score = self.news_manager.can_trade(
                signal.direction, blackout_minutes=NEWS_BLACKOUT_MINUTES
            )
            if not news_ok:
                logger.info(f"  News blocked: {news_reason}")
                self._log_rejection(scan_time, signal, "news", news_reason)
                news_blocked += 1
                continue

            # ML scoring
            ml_score = signal.confidence
            if self.ml_loaded:
                ml_score = self.ml_scorer.score_signal(signal, current_bar, self.feature_builder)
                signal.confidence = ml_score
                logger.info(f"  ML score: {ml_score:.3f} (threshold: {ML_THRESHOLD})")

                if ml_score < ML_THRESHOLD:
                    logger.info(f"  ML rejected ({ml_score:.3f} < {ML_THRESHOLD})")
                    self._log_rejection(scan_time, signal, "ml", f"score {ml_score:.3f}")
                    ml_blocked += 1
                    continue

            # Risk check
            allowed, reason = self.risk_manager.can_trade(signal)
            if not allowed:
                logger.info(f"  Risk blocked: {reason}")
                self._log_rejection(scan_time, signal, "risk", reason)
                risk_blocked += 1
                continue

            # Position sizing
            account = self.executor.get_account_summary()
            self.risk_manager.current_balance = account.get("balance", self.risk_manager.current_balance)
            sizing = self.risk_manager.calculate_position_size(signal)

            if sizing["lot_size"] <= 0:
                continue

            # Max positions check
            open_positions = self.executor.get_open_positions()
            if len(open_positions) >= self.risk_manager.max_open_positions:
                logger.info(f"  Max positions ({len(open_positions)})")
                continue

            logger.info(f"\n  >>> SIGNAL PASSED ALL FILTERS <<<")
            logger.info(f"  Strategy: {signal.strategy_name} | {signal.direction}")
            logger.info(f"  Entry: {signal.entry_price:.2f} | SL: {signal.stop_loss:.2f} | TP: {signal.take_profit:.2f}")
            logger.info(f"  ML: {ml_score:.3f} | H4: {self.mtf_filter.h4_trend} | Lot: {sizing['lot_size']}")

            # ── SHADOW MODE ── log the signal, never place an order ──────────
            if self.shadow_mode:
                self.total_shadow_signals += 1
                shadow_record = {
                    "scan": self.total_scans + 1,
                    "time": scan_time.isoformat(),
                    "direction": signal.direction,
                    "strategy": signal.strategy_name,
                    "entry": signal.entry_price,
                    "sl": signal.stop_loss,
                    "tp": signal.take_profit,
                    "lot_size": sizing["lot_size"],
                    "risk_amount": sizing["risk_amount"],
                    "confidence": signal.confidence,
                    "ml_score": ml_score,
                    "news_score": news_score,
                    "h4_trend": self.mtf_filter.h4_trend,
                    "mtf_boost": mtf_boost,
                    "reason": signal.reason,
                    "mode": "shadow",
                }
                self.shadow_log.append(shadow_record)
                self._save_shadow_log()

                self.telegram.send(
                    "\U0001F441 <b>SHADOW SIGNAL</b> (no order placed)\n"
                    f"Strategy: {signal.strategy_name} | {signal.direction}\n"
                    f"Entry: {signal.entry_price:.2f} | SL: {signal.stop_loss:.2f} | TP: {signal.take_profit:.2f}\n"
                    f"ML: {ml_score:.3f} | H4: {self.mtf_filter.h4_trend}\n"
                    f"Lot: {sizing['lot_size']} | Risk: ${sizing['risk_amount']:.2f}"
                )
                logger.info(f"  SHADOW SIGNAL #{self.total_shadow_signals} logged (no order placed).")
                continue

            # ── LIVE / DEMO MODE ── actually execute the trade ───────────────
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
                self.risk_manager.daily_trades += 1

                # Snapshot for future ML/LSTM training
                try:
                    feat_arr = self.feature_builder.extract_features(signal, current_bar)
                    entry_features = feat_arr.tolist() if hasattr(feat_arr, 'tolist') else None
                except Exception:
                    entry_features = None

                bar_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'rsi_14', 'atr_14', 'ema_20']
                avail_cols = [c for c in bar_cols if c in featured_df.columns]
                bar_sequence = featured_df.iloc[-20:][avail_cols].round(4).values.tolist()

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
                    "ml_score": ml_score,
                    "news_score": news_score,
                    "h4_trend": self.mtf_filter.h4_trend,
                    "mtf_boost": mtf_boost,
                    "reason": signal.reason,
                    "result": "open",
                    "entry_features": entry_features,
                    "bar_sequence": bar_sequence,
                }
                self.trade_log.append(trade_record)
                self._save_trade_log()

                self.telegram.send_trade_opened({
                    "direction": signal.direction,
                    "strategy": signal.strategy_name,
                    "entry": result["price"],
                    "sl": signal.stop_loss,
                    "tp": signal.take_profit,
                    "lot_size": sizing["lot_size"],
                    "confidence": signal.confidence,
                })

                logger.info(f"  TRADE #{self.total_trades_placed} PLACED! Telegram sent.")
            else:
                logger.error(f"  Trade failed: {result.get('error')}")
                self.telegram.send_alert(f"Trade failed: {result.get('error')}")

        # Update trailing stops on open positions
        try:
            atr_val = featured_df.iloc[-1].get("atr_14", None) if not featured_df.empty else None
            self.trailing_manager.update_trailing_stops(current_atr=atr_val)
            self.trailing_manager.cleanup_closed()
        except Exception as e:
            logger.warning(f"  Trailing stop update error: {e}")

        if mtf_blocked + news_blocked + ml_blocked + risk_blocked > 0:
            logger.info(f"  Blocked: MTF={mtf_blocked} News={news_blocked} ML={ml_blocked} Risk={risk_blocked}")

    def _log_rejection(self, time_stamp, signal, reason_type, reason_detail):
        self.rejection_log.append({
            "time": time_stamp.isoformat(),
            "direction": signal.direction,
            "strategy": signal.strategy_name,
            "confidence": round(signal.confidence, 3),
            "blocked_by": reason_type,
            "reason": reason_detail,
        })
        if len(self.rejection_log) % 50 == 0:
            self._save_rejection_log()

    def _refresh_news(self):
        try:
            self.news_manager.refresh()
            self.last_news_refresh = time.time()
        except Exception as e:
            logger.warning(f"  News refresh failed: {e}")

    def _print_status(self):
        account = self.executor.get_account_summary()
        positions = self.executor.get_open_positions()
        mtf_status = self.mtf_filter.get_status()
        session_pnl = account.get("balance", 0) - self.session_start_balance

        mode_str = "SHADOW" if self.shadow_mode else "LIVE"
        print(f"\n  --- BOT STATUS [{mode_str}] ---")
        if self.shadow_mode:
            print(f"  Scans: {self.total_scans + 1} | Shadow signals: {self.total_shadow_signals}")
        else:
            print(f"  Scans: {self.total_scans + 1} | Trades: {self.total_trades_placed}")
        print(f"  Balance: ${account.get('balance', 0):.2f} | Equity: ${account.get('equity', 0):.2f} | "
              f"Session PnL: ${session_pnl:+.2f}")
        print(f"  H4 Trend: {mtf_status['h4_trend']} ({mtf_status['h4_strength']:.2f}) | "
              f"D1 Trend: {mtf_status['d1_trend']}")
        print(f"  Open positions: {len(positions)}")

        for pos in positions:
            pnl_icon = "[+]" if pos["profit"] >= 0 else "[-]"
            print(f"    {pnl_icon} #{pos['ticket']} {pos['direction']} {pos['volume']} lots "
                  f"@ {pos['price_open']:.2f} | PnL: ${pos['profit']:.2f}")

        is_blackout, reason = self.news_manager.calendar.is_news_blackout(NEWS_BLACKOUT_MINUTES)
        if is_blackout:
            print(f"  !! NEWS BLACKOUT: {reason}")
        else:
            print(f"  OK News: Clear")

    def _shutdown(self):
        self.running = False
        self._save_rejection_log()

        positions = self.executor.get_open_positions()
        account = self.executor.get_account_summary()
        session_pnl = account.get("balance", 0) - self.session_start_balance

        print(f"\n{'='*70}")
        print(f"  SESSION ENDED")
        print(f"{'='*70}")
        print(f"  Scans: {self.total_scans} | Trades: {self.total_trades_placed}")
        print(f"  Balance: ${account.get('balance', 0):.2f} | Session PnL: ${session_pnl:+.2f}")
        print(f"  Rejections logged: {len(self.rejection_log)}")
        print(f"{'='*70}\n")

        self.telegram.send(
            "\U0001F6D1 <b>BOT STOPPED</b>\n"
            f"Scans: {self.total_scans}\n"
            f"Trades: {self.total_trades_placed}\n"
            f"Balance: ${account.get('balance', 0):.2f}\n"
            f"Session PnL: ${session_pnl:+.2f}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )

        self.connector.disconnect()
