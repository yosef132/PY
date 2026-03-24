# XAUUSD AI Trading System — CLAUDE.md

This file tells Claude Code how to work in this repo. Read it before making any changes.

---

## What this project is

An automated AI trading bot for Gold (XAUUSD) on MetaTrader 5 demo account.

**Architecture:**
- 6 trading strategies generate signals (ICT, Wyckoff, RSI Divergence, S&R, CRT, SK System)
- XGBoost ML model scores each signal (binary win/loss prediction)
- 4-layer filter: MTF trend → news blackout → ML score → risk manager
- **3-tier MTF filter**: W1 (macro) → H4 (medium) → H1 (entry). W1 BULLISH + H4 BEARISH = pullback entry allowed
- Binance XAUUSDT perpetual futures used as cross-reference (funding rate, OI, bias)
- Economic calendar + headline sentiment block trades around major news events
- Trades execute on MT5 via the MetaTrader5 Python API
- Real-time Streamlit dashboard at localhost:8501

---

## Project structure

```
xauusd-ai-trader/
├── live_trader.py              # Entry point — run this to start the bot
├── dashboard.py                # Streamlit dashboard (run separately)
├── config/
│   ├── settings.yaml           # All tunable parameters (no credentials)
│   └── strategies.yaml         # Which strategies are enabled/disabled
├── .env                        # Credentials (never commit this)
├── src/
│   ├── utils/
│   │   ├── config.py           # Loads YAML + injects .env credentials
│   │   └── logger.py           # Logging setup (UTF-8 file, console)
│   ├── data_collector/
│   │   ├── mt5_collector.py    # MT5 connection + OHLCV data
│   │   └── exchange_collector.py  # Binance futures (funding, OI, bias)
│   ├── feature_engine/
│   │   ├── engine.py           # Orchestrates all 6 feature steps
│   │   ├── indicators.py       # 67 technical indicators via pandas-ta
│   │   ├── market_structure.py # Swing points, BOS, CHOCH, FVGs, OBs
│   │   ├── support_resistance.py
│   │   ├── divergence.py       # RSI divergence detection
│   │   ├── session_features.py # London/NY sessions, killzones
│   │   └── wyckoff_features.py # Springs, upthrusts, effort-vs-result
│   ├── strategies/
│   │   ├── base.py             # Signal dataclass + BaseStrategy
│   │   ├── strategy_manager.py # Runs all enabled strategies
│   │   ├── mtf_filter.py       # W1+H4+D1 three-tier trend confirmation filter
│   │   ├── ict_strategy.py
│   │   ├── wyckoff_strategy.py
│   │   ├── rsi_divergence_strategy.py
│   │   ├── support_resistance_strategy.py
│   │   ├── crt_strategy.py
│   │   └── sk_system_strategy.py
│   ├── risk_manager/
│   │   ├── risk_manager.py     # Position sizing, daily limits, drawdown
│   │   ├── signal_filter.py    # Pre-ML quality filter (RR, confidence, SL distance)
│   │   └── backtester.py
│   ├── ml_scorer/
│   │   ├── model.py            # XGBoost load/score/save
│   │   ├── feature_builder.py  # Extracts ML features from a signal+bar
│   │   ├── auto_retrain.py     # Weekly Sunday retraining
│   │   └── enhanced_backtester.py
│   ├── news_filter/
│   │   ├── news_manager.py     # Combines calendar + sentiment
│   │   ├── economic_calendar.py  # Forex Factory API + fallback
│   │   ├── headline_sentiment.py # Keyword-based gold sentiment
│   │   └── news_scraper.py     # Scrapes Google News/Kitco/Reuters RSS
│   ├── executor/
│   │   ├── live_engine.py      # Main 5-min scan loop (v4)
│   │   ├── mt5_executor.py     # Places/closes orders on MT5
│   │   └── trailing_stop.py    # Partial TP at 1R (50% closed), SL→breakeven, trail 0.5×ATR
│   └── alerts/
│       └── telegram_alerts.py  # Sends trade notifications to Telegram
└── data/
    ├── models/xgb_signal_scorer.pkl  # Trained ML model
    ├── live_trades.json        # Trade log (persisted per trade)
    └── rejected_signals.json   # Last 500 rejection reasons (for debugging)
```

---

## How to run

```bash
# Bot (keep running — Ctrl+C to stop safely)
py -u live_trader.py

# Dashboard (separate terminal)
py -m streamlit run dashboard.py

# Retrain ML model manually
py retrain_model.py

# Run a backtest
py run_ml_backtest.py
```

---

## Credentials setup

All credentials live in `.env` — never in YAML or code:

```
MT5_LOGIN=12345678
MT5_PASSWORD=yourpassword
MT5_SERVER=YourBroker-Demo
TELEGRAM_TOKEN=bot:xxxx
TELEGRAM_CHAT_ID=123456789
```

`src/utils/config.py` loads `.env` at import time via `python-dotenv` and injects values over the YAML nulls.

---

## Key config values (config/settings.yaml)

| Key | Default | What it controls |
|-----|---------|-----------------|
| `ml.confidence_threshold` | 0.65 | ML score needed to pass (read by live_engine at init) |
| `risk.news_blackout_minutes` | 30 | Minutes before/after news to block trading |
| `risk.risk_per_trade_pct` | 0.5 | % of balance risked per trade |
| `risk.max_daily_loss_pct` | 3.0 | Daily loss limit before bot stops trading |
| `risk.max_drawdown_pct` | 10.0 | Drawdown limit from peak balance |
| `risk.min_signal_confidence` | 0.60 | Pre-ML signal quality filter |
| `risk.min_risk_reward` | 2.0 | Minimum R:R ratio |
| `risk.max_trade_hours` | 48 | Close losing trades open longer than this |
| `risk.max_trade_hours_hard` | 72 | Close ALL trades open longer than this (hard limit) |
| `risk.partial_tp_enabled` | true | Close 50% at 1R, trail remaining 50% to 2R |
| `loop.scan_interval_seconds` | 300 | How often the bot scans (5 min) |
| `loop.news_refresh_seconds` | 900 | How often news is refreshed (15 min) |
| `loop.h4_refresh_seconds` | 3600 | How often W1/H4/D1 trend is refreshed (1 hour) |

---

## Live engine flow (live_engine.py)

Every `scan_interval` seconds:
1. `reset_daily()` — resets daily counters at midnight
2. `_update_trade_outcomes()` — checks MT5 for closed positions, updates live_trades.json with win/loss/pnl
3. `_check_time_exits()` — closes losing trades open > 48h, any trade open > 72h
4. Refresh news/W1+H4+D1 if their intervals elapsed
5. Fetch Binance data (funding rate, OI, bias)
6. Fetch 500 H1 candles from MT5
7. **Stale data check** — skip scan if last candle > 3 hours old
8. Compute 133 features (67 indicators + structure + S/R + sessions + divergence + Wyckoff)
9. Run all 6 strategies → raw signals
10. Signal filter (RR, confidence, SL distance, TP direction)
11. **Recency filter** — only signals from last 24 H1 bars (prevents stale signal execution)
12. For each signal (max 3 per scan):
    - **3-tier MTF filter**: W1+H4+D1 (W1 BULLISH + H4 BEARISH = pullback allowed)
    - Binance boost/penalty (±0.05 confidence)
    - News blackout check
    - ML score (XGBoost — must be ≥ threshold)
    - **Correlation check** — max 2 same-direction positions
    - Risk manager veto (daily limit, drawdown, max positions)
    - Execute on MT5 — saves entry_features + bar_sequence to trade log
13. Update trailing stops → partial TP at 1R (50% closed), SL → breakeven, trail remaining
14. Print status (shows W1/H4/D1 trend)

---

## Important rules for this codebase

- **Never use Unicode emojis in `print()` statements** — Windows cp1252 will crash. Use ASCII (`[OK]`, `[XX]`, `[!!]`) for console output. Emoji in Telegram messages (sent via requests.post) is fine.
- **Credentials only in .env** — never put login/password/tokens in YAML or Python files.
- **All tunable numbers belong in settings.yaml** — not hardcoded in Python. Read them with `.get("key", default)`.
- **Use `logger.info/warning/error` for logging**, not `print()`, except for user-facing startup/status displays.
- **The bot is always in DEMO mode** — never test on a live account without explicit confirmation.

---

## ML model

- **Stored at**: `data/models/xgb_signal_scorer.pkl`
- **Type**: XGBoost binary classifier (win=1 / loss=0)
- **Features**: ~80 features built from the last H1 bar (indicators, structure, time encoding)
- **Training**: `TimeSeriesSplit(n_splits=5)` CV — no look-ahead bias
- **Key features**: `dow_cos`, `hour_sin`, `dist_to_support_pct` (time + structure matter most)
- **Performance**: 63.8% CV accuracy, win rate 31.6% → 64.6% with ML filter
- **Retrain**: Run `py retrain_model.py` or auto-retrains every Sunday

---

## Current build status (as of 2026-03-24)

### Blueprint phase progress
| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Foundation — data, features, backtester | COMPLETE |
| Phase 2 | All 6 strategies + risk manager | COMPLETE |
| Phase 3 | XGBoost ML + news filter + live demo execution + dashboard | COMPLETE |
| Phase 4 | Self-improving: auto-retrain, LSTM, A/B testing | ~80% done |
| Phase 5 | Live trading | NOT STARTED |

### Phase 4 completed
- [x] Fixed ML overfitting: adaptive regularization (depth/trees/lambda scale with samples/features ratio)
- [x] Fixed data snooping in backtester: 70/30 walk-forward split, ML trained only on first 70%
- [x] Fixed lookahead bias in swing point detection (market_structure.py)
- [x] Fixed model comparison: CV accuracy instead of train accuracy (auto_retrain.py)
- [x] Upgraded news sentiment to FinBERT (ProsusAI/finbert) with 12/12 gold-specific accuracy
- [x] Added Shadow Mode: real account read-only analysis (set shadow_mode: true in settings.yaml)
- [x] Fixed hardcoded confidence threshold in risk_manager.py (now reads min_signal_confidence from settings.yaml)
- [x] Fixed stale signal bug: recency filter (last 24 H1 bars) prevents signals from weeks-old bars executing at current price
- [x] Fixed executor TP/SL sanity check: rejects trades where TP/SL are wrong-sided instead of silently adjusting
- [x] Added Sunday auto-retraining wired into live engine loop
- [x] Added trade outcome tracker: detects MT5 position closures, updates live_trades.json with win/loss/pnl
- [x] Added entry_features + bar_sequence saved per trade (LSTM-ready training data)
- [x] Added W1 (weekly) to MTF filter — 3-tier W1+H4+D1 system, pullback entries now allowed
- [x] Added time-based exit: losing trades closed after 48h, hard limit 72h
- [x] Added correlation protection: max 2 same-direction positions open simultaneously
- [x] Added partial TP at 1R: closes 50% at market, moves SL to breakeven, trails remaining 50%

### Phase 4 remaining work
- [ ] **DATA COLLECTION** — run bot continuously until 300+ trades logged in data/live_trades.json
- [ ] Retrain XGBoost once 300 trades collected: `py retrain_model.py` — then raise thresholds back to 0.65/0.60
- [ ] A/B testing framework: ML-filtered signals vs unfiltered signals comparison
- [ ] LSTM/Transformer model alongside XGBoost (after XGBoost is stable with 300+ trades)
- [ ] Install FinBERT in .venv: `.venv\Scripts\pip install torch transformers` (currently using keyword fallback)
- [ ] Fix news scraper: install `lxml[html_clean]` to enable Google News RSS parsing

### ML model state (last honest backtest 2026-03-23)
- Training trades: 129 | Test trades: 0 (model rejects all — needs more data)
- Honest baseline (no ML): 37 trades | win rate 35.1% | profit factor 1.14 | drawdown 11.8%
- Avg win $98.72 vs avg loss $46.96 — positive expectancy, strategies have real edge
- ML needs ~300-500 real demo trades before it can add value reliably
- CV accuracy: 66.7% = predicting majority class (loss) — expected with limited data

### Live bot state (2026-03-24)
- Bot running on demo account 92877 (Nexcbitmarket-Trade)
- Thresholds temporarily lowered for data collection: ML=0.50, min_signal_confidence=0.50
- Restore to ML=0.65, min_signal_confidence=0.60 after 300+ trades and retraining
- Trade frequency: ~2-4 per day (W1 BULLISH now allows BUY pullback entries, not just SELLs)
- W1=BULLISH | H4=BEARISH | D1=BEARISH (gold pullback in macro bull trend — as of 2026-03-24)
- Current trade count: `py -c "import json; d=json.load(open('data/live_trades.json')); print(len(d))"`

### Planned upgrades (in order)
1. **Now**: Collect 300 demo trades — keep bot running 24/7
2. **After 300 trades**: Retrain XGBoost, restore thresholds, validate ML is adding value
3. **After stable XGBoost**: Add LSTM/Transformer as a second model in ml_scorer/ pipeline
4. **A/B testing**: Compare ML-filtered vs raw signal performance once enough data exists
5. **Phase 5**: Move to live account with tight risk limits (0.25% per trade, 1.5% daily max)

---

## Known issues / technical debt

| Issue | File | Severity | Status |
|-------|------|----------|--------|
| ML model needs more data | data/live_trades.json | HIGH | Only ~129 trades — collect 300+ before raising thresholds |
| FinBERT not installed in .venv | news_filter/headline_sentiment.py | MEDIUM | Using keyword fallback; run `.venv\Scripts\pip install torch transformers` |
| Google News RSS parser missing lxml-xml | news_filter/news_scraper.py | MEDIUM | Non-fatal; install `lxml[html_clean]` to fix |
| Reuters DNS failing | news_filter/news_scraper.py | LOW | Network/broker issue — non-fatal, falls back to neutral sentiment |
| PerformanceWarning on fragmented DataFrame | session_features.py, wyckoff_features.py | LOW | Cosmetic only — no errors |
| Swing point detection uses nested loops | market_structure.py | LOW | Acceptable for 500-bar window |
| Unused import: `field` from dataclasses | strategies/base.py | LOW | Cosmetic |
| Forex Factory API fails on weekends | economic_calendar.py | LOW | Falls back to built-in schedule — expected |

---

## Testing

Before making changes to the live engine, verify with:

```bash
# Check all imports work
py -c "from src.executor.live_engine import LiveTradingEngine; print('OK')"

# Verify config loads all keys
py -c "from src.utils.config import get_settings; s=get_settings(); print(s.get('ml',{}).get('confidence_threshold'))"

# Syntax check a file
py -c "import ast; ast.parse(open('src/executor/live_engine.py').read()); print('Syntax OK')"
```

---

## Windows-specific notes

- Python launcher: use `py` not `python` or `python3`
- Always run with `py -u` (unbuffered) to see print output in real-time
- Log files are UTF-8 encoded (set in logger.py)
- Console is cp1252 — no emoji in print() statements
- MetaTrader5 package only works when MT5 terminal is installed and running
