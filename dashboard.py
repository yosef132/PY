"""
XAUUSD AI Trading System — Real-Time Monitoring Dashboard
Run: streamlit run dashboard.py
Auto-refreshes every 30 seconds.
"""

import sys
import json
import time
import warnings
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta, timezone

import streamlit as st

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XAUUSD AI Trader",
    page_icon="gold",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Dark theme enhancements */
    .metric-card {
        background: #1a1d2e;
        border: 1px solid #2d3150;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 4px 0;
    }
    .metric-label { color: #888; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.07em; }
    .metric-value { color: #fff; font-size: 1.7rem; font-weight: 700; margin: 2px 0; }
    .metric-delta-pos { color: #26a65b; font-size: 0.82rem; }
    .metric-delta-neg { color: #e74c3c; font-size: 0.82rem; }
    .profit { color: #26a65b; font-weight: 600; }
    .loss   { color: #e74c3c; font-weight: 600; }
    .neutral { color: #888; }
    .tag { background: #2d3150; color: #a0aec0; padding: 2px 8px;
           border-radius: 12px; font-size: 0.75rem; margin: 0 2px; }
    .status-dot-green { color: #26a65b; }
    .status-dot-red   { color: #e74c3c; }
    .stTabs [data-baseweb="tab"] { font-size: 0.92rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ─── Data Loaders ───────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def load_candles():
    candles = {}
    for tf in ["M5", "M15", "H1", "H4", "D1"]:
        p = PROJECT_ROOT / "data" / "historical" / f"XAUUSD_{tf}.parquet"
        if p.exists():
            candles[tf] = pd.read_parquet(p)
    return candles


@st.cache_data(ttl=10)
def load_live_trades():
    p = PROJECT_ROOT / "data" / "live_trades.json"
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    return []


@st.cache_data(ttl=10)
def load_rejection_log():
    p = PROJECT_ROOT / "data" / "rejected_signals.json"
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    return []


@st.cache_data(ttl=300)
def load_ml_model():
    p = PROJECT_ROOT / "data" / "models" / "xgb_signal_scorer.pkl"
    if p.exists():
        try:
            data = joblib.load(p)
            return data["model"], data["feature_names"]
        except Exception:
            pass
    return None, None


@st.cache_data(ttl=60)
def load_mt5_live():
    """Connect to MT5 and get live account + positions + price."""
    try:
        from src.data_collector.mt5_collector import MT5Connector, DataCollector
        conn = MT5Connector()
        if not conn.connect():
            return None, None, None
        account = conn.get_account_info()
        price   = conn.get_account_info()  # placeholder

        import MetaTrader5 as mt5
        tick = mt5.symbol_info_tick("XAUUSD")
        live_price = {"bid": tick.bid, "ask": tick.ask,
                      "spread": round(tick.ask - tick.bid, 2)} if tick else {}

        positions_raw = mt5.positions_get(symbol="XAUUSD") or []
        positions = []
        for p in positions_raw:
            positions.append({
                "ticket": p.ticket,
                "direction": "BUY" if p.type == 0 else "SELL",
                "volume": p.volume,
                "price_open": p.price_open,
                "price_current": p.price_current,
                "profit": p.profit,
                "sl": p.sl,
                "tp": p.tp,
                "comment": p.comment,
                "time": datetime.fromtimestamp(p.time).strftime("%H:%M:%S"),
            })

        conn.disconnect()
        return account, live_price, positions
    except Exception:
        return None, None, None


@st.cache_data(ttl=60)
def load_binance_data():
    try:
        from src.data_collector.exchange_collector import ExchangeCollector
        ex = ExchangeCollector()
        return ex.fetch_all()
    except Exception:
        return {}


@st.cache_data(ttl=900)
def load_news():
    try:
        from src.news_filter.news_manager import NewsManager
        mgr = NewsManager()
        mgr.refresh()
        return mgr
    except Exception:
        return None


# ─── Helper: Equity curve from trades ───────────────────────────────────────
def build_equity_curve(trades: list, start_balance: float = 100000.0) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    rows = []
    balance = start_balance
    for t in trades:
        pnl = t.get("pnl", 0) or 0
        balance += pnl
        rows.append({
            "time": t.get("time") or t.get("entry_time", ""),
            "balance": balance,
            "pnl": pnl,
            "direction": t.get("direction", ""),
            "strategy": t.get("strategy", ""),
            "result": t.get("result", ""),
        })
    return pd.DataFrame(rows)


def strategy_stats(trades: list) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    by_strat = {}
    for t in trades:
        s = t.get("strategy", "unknown")
        if s not in by_strat:
            by_strat[s] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0}
        by_strat[s]["trades"] += 1
        pnl = t.get("pnl", 0) or 0
        by_strat[s]["pnl"] += pnl
        outcome = t.get("result", "")
        if outcome == "win" or pnl > 0:
            by_strat[s]["wins"] += 1
        elif outcome == "loss" or pnl < 0:
            by_strat[s]["losses"] += 1

    rows = []
    for strat, d in by_strat.items():
        total = d["trades"]
        wr = d["wins"] / total * 100 if total > 0 else 0
        rows.append({
            "Strategy": strat,
            "Trades": total,
            "Wins": d["wins"],
            "Losses": d["losses"],
            "Win Rate": f"{wr:.0f}%",
            "PnL": d["pnl"],
        })
    return pd.DataFrame(rows).sort_values("Trades", ascending=False)


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## XAUUSD AI Trader")
    st.caption("Gold Trading Intelligence System")
    st.divider()

    page = st.radio("Navigation", [
        "Overview",
        "Live Trading",
        "Trade Journal",
        "News & Sentiment",
        "ML Intelligence",
    ], label_visibility="collapsed")

    st.divider()

    # Quick account snapshot in sidebar
    account, _, _ = load_mt5_live()
    if account:
        balance = account.get("balance", 0)
        equity  = account.get("equity",  0)
        profit  = account.get("profit",  0)
        st.metric("Balance", f"${balance:,.2f}")
        st.metric("Equity",  f"${equity:,.2f}",  f"{profit:+.2f}")
        st.caption(f"Account: {account.get('login')} | {account.get('trade_mode')}")
    else:
        st.warning("MT5 not connected")

    st.divider()
    auto_refresh = st.toggle("Auto-refresh (30s)", value=True)
    if st.button("Refresh now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")


# Auto-refresh
if auto_refresh:
    time.sleep(0.1)
    st.markdown(
        f'<meta http-equiv="refresh" content="30">',
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================
if page == "Overview":
    st.title("XAUUSD AI Trader — Overview")

    account, live_price, positions = load_mt5_live()
    candles = load_candles()
    binance = load_binance_data()
    live_trades = load_live_trades()

    # ── Row 1: Key metrics ──────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    if live_price:
        bid = live_price.get("bid", 0)
        ask = live_price.get("ask", 0)
        col1.metric("Gold Bid", f"${bid:,.2f}")
        col2.metric("Gold Ask", f"${ask:,.2f}", f"Spread: {live_price.get('spread', 0):.2f}")
    elif "H1" in candles:
        last = candles["H1"].iloc[-1]
        prev = candles["H1"].iloc[-2]
        chg  = last["close"] - prev["close"]
        col1.metric("Gold (H1 close)", f"${last['close']:,.2f}", f"{chg:+.2f}")
        col2.metric("H1 High / Low", f"{last['high']:.2f} / {last['low']:.2f}")

    if account:
        col3.metric("Balance",    f"${account.get('balance', 0):,.2f}")
        col4.metric("Equity",     f"${account.get('equity',  0):,.2f}")
        col5.metric("Open P&L",   f"${account.get('profit',  0):+,.2f}",
                    delta_color="normal" if account.get("profit", 0) >= 0 else "inverse")
    else:
        col3.metric("Balance", "—")
        col4.metric("Equity",  "—")
        col5.metric("Open P&L","—")

    st.divider()

    # ── Row 2: Open positions + Binance ─────────────────────
    left, right = st.columns([2, 1])

    with left:
        st.subheader(f"Open Positions ({len(positions or [])})")
        if positions:
            pos_df = pd.DataFrame(positions)
            pos_df["P&L"] = pos_df["profit"].apply(
                lambda x: f"**:green[+${x:.2f}]**" if x >= 0 else f"**:red[${x:.2f}]**"
            )
            st.dataframe(
                pos_df[["ticket","direction","volume","price_open","price_current","sl","tp","P&L","time","comment"]],
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No open positions")

    with right:
        st.subheader("Binance XAUUSDT")
        if binance:
            bias = binance.get("bias", "neutral")
            bias_color = "green" if bias == "bullish" else ("red" if bias == "bearish" else "gray")
            st.markdown(f"**Bias: :{bias_color}[{bias.upper()}]**")
            st.metric("Funding Rate", f"{binance.get('funding_rate_pct', 0):+.4f}%",
                      help="Negative = longs paid (bullish pressure)")
            st.metric("Open Interest", f"${binance.get('open_interest_usd', 0)/1e6:.1f}M")
            st.metric("24h Change", f"{binance.get('price_change_pct_24h', 0):+.2f}%")
            st.metric("Mark Price", f"${binance.get('mark_price', 0):,.2f}")
        else:
            st.info("Binance data unavailable")

    st.divider()

    # ── Row 3: XAUUSD Price Chart ────────────────────────────
    if "H1" in candles:
        st.subheader("XAUUSD — H1 Chart (Last 200 Candles)")
        df = candles["H1"].tail(200).copy()

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.78, 0.22], vertical_spacing=0.04)

        fig.add_trace(go.Candlestick(
            x=df.index, open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            name="XAUUSD",
            increasing_line_color="#26a65b",
            decreasing_line_color="#e74c3c",
        ), row=1, col=1)

        # EMA overlays if features exist
        for col_name, color, label in [
            ("ema_21", "#f39c12", "EMA 21"),
            ("ema_50", "#3498db", "EMA 50"),
            ("ema_200","#9b59b6", "EMA 200"),
        ]:
            if col_name in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col_name],
                    mode="lines", name=label,
                    line=dict(color=color, width=1, dash="dot"),
                ), row=1, col=1)

        vol_colors = ["#26a65b" if c >= o else "#e74c3c"
                      for c, o in zip(df["close"], df["open"])]
        fig.add_trace(go.Bar(
            x=df.index, y=df["volume"],
            name="Volume", marker_color=vol_colors, opacity=0.6,
        ), row=2, col=1)

        fig.update_layout(
            height=560, template="plotly_dark",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(orientation="h", y=1.02, x=0),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: System Health ─────────────────────────────────
    st.subheader("System Health")
    h1, h2, h3, h4 = st.columns(4)

    model, feat_names = load_ml_model()
    h1.metric("ML Model",     "Loaded" if model else "Not trained",
              f"{len(feat_names)} features" if feat_names else None)
    h2.metric("Live Trades",  len(live_trades))
    h3.metric("MT5 Status",   "Connected" if account else "Disconnected")
    h4.metric("Strategies",   "6 active")


# ============================================================
# PAGE 2: LIVE TRADING
# ============================================================
elif page == "Live Trading":
    st.title("Live Trading — Real-Time Monitor")

    account, live_price, positions = load_mt5_live()
    binance  = load_binance_data()
    candles  = load_candles()
    rejection_log = load_rejection_log()

    # ── Live price bar ──────────────────────────────────────
    if live_price:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bid",    f"${live_price.get('bid', 0):,.2f}")
        c2.metric("Ask",    f"${live_price.get('ask', 0):,.2f}")
        c3.metric("Spread", f"${live_price.get('spread', 0):.2f}")
        if account:
            c4.metric("Free Margin", f"${account.get('margin_free', 0):,.2f}")

    st.divider()

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("Open Positions")
        if positions:
            for p in positions:
                pnl  = p["profit"]
                icon = "🟢" if pnl >= 0 else "🔴"
                dir_color = "green" if p["direction"] == "BUY" else "red"
                with st.container(border=True):
                    r1, r2, r3 = st.columns([2, 2, 1])
                    r1.markdown(f"**:{dir_color}[{p['direction']}]** #{p['ticket']}")
                    r1.caption(f"Entry: ${p['price_open']:.2f}  |  Now: ${p['price_current']:.2f}")
                    r2.metric("P&L", f"${pnl:+.2f}", delta_color="normal" if pnl >= 0 else "inverse")
                    r3.metric("Lots", p["volume"])
                    if p["sl"] or p["tp"]:
                        st.caption(f"SL: ${p['sl']:.2f}  |  TP: ${p['tp']:.2f}  |  {p['comment']}")
        else:
            st.info("No open positions — bot is scanning for setups")

    with col_r:
        st.subheader("Binance Cross-Reference")
        if binance:
            bias = binance.get("bias", "neutral")
            cols = st.columns(2)
            cols[0].metric("Bias",         bias.upper())
            cols[0].metric("Funding",      f"{binance.get('funding_rate_pct', 0):+.4f}%")
            cols[1].metric("OI",           f"${binance.get('open_interest_usd', 0)/1e6:.1f}M")
            cols[1].metric("24h Change",   f"{binance.get('price_change_pct_24h', 0):+.2f}%")

            # OI gauge
            oi = binance.get("open_interest_usd", 0) / 1e6
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=binance.get("price_change_pct_24h", 0),
                title={"text": "24h Price Change %"},
                gauge={
                    "axis": {"range": [-5, 5]},
                    "bar":  {"color": "#26a65b" if binance.get("price_change_pct_24h", 0) >= 0 else "#e74c3c"},
                    "steps": [
                        {"range": [-5, -2], "color": "#2d1a1a"},
                        {"range": [-2,  2], "color": "#1a1d2e"},
                        {"range": [ 2,  5], "color": "#1a2d1a"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 2}, "value": 0},
                },
            ))
            fig_g.update_layout(height=200, template="plotly_dark",
                                margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_g, use_container_width=True)

    st.divider()

    # ── Signal rejection analysis ───────────────────────────
    st.subheader(f"Recent Signal Rejections (last {min(50, len(rejection_log))})")
    if rejection_log:
        recent = rejection_log[-50:]
        df_rej = pd.DataFrame(recent)

        # Breakdown by rejection reason
        by_reason = df_rej["blocked_by"].value_counts().reset_index()
        by_reason.columns = ["Blocked By", "Count"]

        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(by_reason, use_container_width=True, hide_index=True)
        with c2:
            fig_pie = go.Figure(go.Pie(
                labels=by_reason["Blocked By"],
                values=by_reason["Count"],
                hole=0.45,
                marker_colors=["#3498db", "#e74c3c", "#f39c12", "#9b59b6", "#26a65b"],
            ))
            fig_pie.update_layout(
                height=280, template="plotly_dark",
                showlegend=True,
                margin=dict(l=0, r=0, t=20, b=0),
                title="Why Signals Were Rejected",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Recent Rejections Table")
        df_show = df_rej.tail(20)[["time", "direction", "strategy", "confidence", "blocked_by", "reason"]]
        st.dataframe(df_show, use_container_width=True, hide_index=True)
    else:
        st.info("No rejection data yet — start the bot to collect signals.")

    # ── Multi-TF snapshot ───────────────────────────────────
    if candles:
        st.divider()
        st.subheader("Multi-Timeframe Snapshot")
        tf_cols = st.columns(len(candles))
        for i, (tf, df) in enumerate(candles.items()):
            if df.empty:
                continue
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last
            chg  = last["close"] - prev["close"]
            pct  = chg / prev["close"] * 100
            color = "normal" if chg >= 0 else "inverse"
            tf_cols[i].metric(tf, f"${last['close']:,.2f}",
                              f"{chg:+.2f} ({pct:+.2f}%)", delta_color=color)


# ============================================================
# PAGE 3: TRADE JOURNAL
# ============================================================
elif page == "Trade Journal":
    st.title("Trade Journal")

    live_trades = load_live_trades()

    if not live_trades:
        st.warning("No trades recorded yet. Run the bot to generate trades.")
        st.info("Start the bot: `python run_full_system.py`")
        st.stop()

    df_trades = pd.DataFrame(live_trades)
    total   = len(df_trades)
    wins    = len(df_trades[df_trades.get("result","") == "win"]) if "result" in df_trades.columns else 0
    open_t  = len(df_trades[df_trades.get("result","") == "open"]) if "result" in df_trades.columns else 0
    closed  = total - open_t

    # ── Summary metrics ─────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Trades",  total)
    c2.metric("Open",          open_t)
    c3.metric("Closed",        closed)

    if "pnl" in df_trades.columns:
        closed_df = df_trades[df_trades.get("result","") != "open"]
        total_pnl = closed_df["pnl"].sum() if "pnl" in closed_df.columns else 0
        wins_count = len(closed_df[closed_df["pnl"] > 0]) if "pnl" in closed_df.columns else 0
        wr = wins_count / len(closed_df) * 100 if len(closed_df) > 0 else 0
        c4.metric("Win Rate", f"{wr:.1f}%",
                  delta_color="normal" if wr >= 50 else "inverse")
        c5.metric("Total PnL", f"${total_pnl:+,.2f}",
                  delta_color="normal" if total_pnl >= 0 else "inverse")
    else:
        c4.metric("Win Rate", "—")
        c5.metric("Total PnL", "—")

    st.divider()

    # ── Equity curve ─────────────────────────────────────────
    start_bal = 100000.0
    if account := load_mt5_live()[0]:
        start_bal = account.get("balance", 100000.0)

    equity_df = build_equity_curve(live_trades, start_bal)
    if not equity_df.empty and "balance" in equity_df.columns:
        st.subheader("Equity Curve")
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=list(range(len(equity_df))),
            y=equity_df["balance"],
            mode="lines",
            line=dict(color="#26a65b", width=2),
            fill="tozeroy",
            fillcolor="rgba(38,166,91,0.08)",
            name="Balance",
        ))
        fig_eq.add_hline(y=start_bal, line_dash="dash", line_color="#888",
                         annotation_text="Start Balance")
        fig_eq.update_layout(
            height=320, template="plotly_dark",
            yaxis_title="Balance ($)",
            xaxis_title="Trade #",
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig_eq, use_container_width=True)

    st.divider()

    # ── Strategy performance ─────────────────────────────────
    if "strategy" in df_trades.columns:
        st.subheader("Performance by Strategy")
        strat_df = strategy_stats(live_trades)
        if not strat_df.empty:
            c_table, c_chart = st.columns([1, 1])
            with c_table:
                # Color PnL column
                def color_pnl(val):
                    color = "#26a65b" if val >= 0 else "#e74c3c"
                    return f"color: {color}"
                styled = strat_df.style.applymap(color_pnl, subset=["PnL"])
                st.dataframe(strat_df, use_container_width=True, hide_index=True)
            with c_chart:
                if "PnL" in strat_df.columns:
                    colors = ["#26a65b" if v >= 0 else "#e74c3c" for v in strat_df["PnL"]]
                    fig_s = go.Figure(go.Bar(
                        x=strat_df["Strategy"],
                        y=strat_df["PnL"],
                        marker_color=colors,
                        text=[f"${v:+.0f}" for v in strat_df["PnL"]],
                        textposition="auto",
                    ))
                    fig_s.update_layout(
                        height=300, template="plotly_dark",
                        yaxis_title="PnL ($)",
                        margin=dict(l=0, r=0, t=20, b=0),
                        title="PnL by Strategy",
                    )
                    st.plotly_chart(fig_s, use_container_width=True)

    st.divider()

    # ── Trade table ──────────────────────────────────────────
    st.subheader(f"All Trades ({total})")
    display_cols = [c for c in ["time","direction","strategy","entry","sl","tp",
                                 "lot_size","confidence","ml_score","result","pnl",
                                 "h4_trend","reason"]
                    if c in df_trades.columns]
    st.dataframe(df_trades[display_cols].tail(100).iloc[::-1],
                 use_container_width=True, hide_index=True)


# ============================================================
# PAGE 4: NEWS & SENTIMENT
# ============================================================
elif page == "News & Sentiment":
    st.title("News & Sentiment Intelligence")

    if st.button("Refresh News Now", use_container_width=True):
        st.cache_data.clear()

    with st.spinner("Loading news data..."):
        mgr = load_news()

    if mgr is None:
        st.error("Could not load news module.")
        st.stop()

    sent = mgr.sentiment
    direction = sent.get("direction", "neutral")
    score     = sent.get("avg_score", 0)

    # ── Sentiment summary ────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    dir_color = "green" if direction == "bullish" else ("red" if direction == "bearish" else "gray")
    c1.markdown(f"### Sentiment\n**:{dir_color}[{direction.upper()}]**")
    c1.metric("Score", f"{score:+.3f}")
    c2.metric("Bullish Headlines", sent.get("bullish_count", 0))
    c3.metric("Bearish Headlines", sent.get("bearish_count", 0))
    c4.metric("Neutral Headlines", sent.get("neutral_count", 0))

    # ── Blackout status ──────────────────────────────────────
    is_blackout, blackout_reason = mgr.calendar.is_news_blackout(15)
    if is_blackout:
        st.error(f"NEWS BLACKOUT ACTIVE: {blackout_reason}")
    else:
        st.success("No news blackout — trading is allowed")

    st.divider()

    # ── Economic calendar ────────────────────────────────────
    col_cal, col_news = st.columns([1, 1])

    with col_cal:
        st.subheader(f"Economic Calendar — Next 48h ({len(mgr.calendar.events)} events)")
        if mgr.calendar.events:
            cal_rows = []
            for ev in mgr.calendar.events:
                cal_rows.append({
                    "Time (UTC)": ev.event_time.strftime("%m-%d %H:%M"),
                    "Currency":   ev.currency,
                    "Event":      ev.name,
                    "Impact":     ev.impact,
                    "Gold Impact": "*" * ev.gold_impact,
                })
            cal_df = pd.DataFrame(cal_rows)
            # Highlight High impact
            def hl_impact(row):
                if row["Impact"] == "High":
                    return ["background-color: #2d1a1a"] * len(row)
                elif row["Impact"] == "Medium":
                    return ["background-color: #2d2a1a"] * len(row)
                return [""] * len(row)
            st.dataframe(cal_df.style.apply(hl_impact, axis=1),
                         use_container_width=True, hide_index=True, height=400)
        else:
            st.info("No upcoming events (weekend or API temporarily unavailable)")

    with col_news:
        st.subheader(f"Latest Headlines ({len(mgr.headlines)})")
        if mgr.headlines:
            from src.news_filter.headline_sentiment import HeadlineSentiment
            scorer = HeadlineSentiment()
            for h in mgr.headlines[:15]:
                result = scorer.score_headline(h)
                s = result["score"]
                if s > 0.2:
                    st.markdown(f":green[+{s:.2f}] {h}")
                elif s < -0.2:
                    st.markdown(f":red[{s:.2f}] {h}")
                else:
                    st.markdown(f":gray[{s:+.2f}] {h}")
        else:
            st.info("No headlines fetched")

    # ── Sentiment gauge ──────────────────────────────────────
    st.divider()
    st.subheader("Sentiment Gauge")
    fig_sent = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={"reference": 0, "relative": False},
        title={"text": "Gold Sentiment Score (-1 Bearish / +1 Bullish)"},
        gauge={
            "axis": {"range": [-1, 1], "tickwidth": 1},
            "bar":  {"color": "#26a65b" if score >= 0 else "#e74c3c", "thickness": 0.25},
            "steps": [
                {"range": [-1.0, -0.5], "color": "#3d1414"},
                {"range": [-0.5, -0.2], "color": "#3d2814"},
                {"range": [-0.2,  0.2], "color": "#1a1d2e"},
                {"range": [ 0.2,  0.5], "color": "#142814"},
                {"range": [ 0.5,  1.0], "color": "#1a3d1a"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))
    fig_sent.update_layout(height=250, template="plotly_dark",
                           margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_sent, use_container_width=True)


# ============================================================
# PAGE 5: ML INTELLIGENCE
# ============================================================
elif page == "ML Intelligence":
    st.title("ML Signal Scorer — XGBoost Intelligence")

    model, feature_names = load_ml_model()

    if model is None:
        st.warning("No trained ML model found.")
        st.info("Run: `python run_ml_backtest.py` to train the model")
        st.stop()

    model_path = PROJECT_ROOT / "data" / "models" / "xgb_signal_scorer.pkl"
    modified   = datetime.fromtimestamp(model_path.stat().st_mtime)

    # ── Model metadata ───────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Type",      "XGBoost")
    c2.metric("Features",        len(feature_names))
    c3.metric("Trees",           model.n_estimators)
    c4.metric("Last Trained",    modified.strftime("%Y-%m-%d %H:%M"))

    st.divider()

    # ── Feature importance ───────────────────────────────────
    importances = model.feature_importances_
    top_n = 20
    top_pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:top_n]
    names, imps = zip(*top_pairs)

    c_chart, c_table = st.columns([2, 1])

    with c_chart:
        st.subheader(f"Top {top_n} Predictive Features")
        colors = px.colors.sequential.Plasma_r[:len(names)]
        fig_imp = go.Figure(go.Bar(
            x=imps,
            y=names,
            orientation="h",
            marker=dict(
                color=imps,
                colorscale="Plasma",
                showscale=True,
                colorbar=dict(title="Importance"),
            ),
        ))
        fig_imp.update_layout(
            height=580, template="plotly_dark",
            xaxis_title="Feature Importance",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with c_table:
        st.subheader("Feature Rankings")
        df_feat = pd.DataFrame({
            "Feature":    names,
            "Importance": [round(i, 4) for i in imps],
            "Rank":       list(range(1, len(names) + 1)),
        })
        st.dataframe(df_feat[["Rank","Feature","Importance"]],
                     use_container_width=True, hide_index=True, height=580)

    st.divider()

    # ── Key insights ─────────────────────────────────────────
    st.subheader("What the AI Has Learned")
    top3 = [names[0], names[1], names[2]]

    insight_map = {
        "dow_cos":              "Day of week matters — gold trends differ by weekday",
        "hour_sin":             "Time of day is critical — London/NY killzones confirmed",
        "dist_to_support_pct":  "Distance from support is a key predictor — near support = better longs",
        "dist_to_resistance_pct": "Distance from resistance predicts reversal probability",
        "ema_alignment":        "EMA stack alignment strongly predicts trend continuation",
        "sr_position":          "Position relative to S/R zone is highly predictive",
        "effort_vs_result":     "Wyckoff effort vs result: confirms or denies momentum",
        "atr_14":               "Volatility level affects strategy success rates",
        "rsi_14":               "RSI level at signal time influences outcome",
    }

    st.markdown("**Top features the model relies on:**")
    for feat in top3:
        insight = insight_map.get(feat, f"`{feat}` — market context feature")
        st.markdown(f"- **`{feat}`**: {insight}")

    st.divider()

    # ── Model settings ───────────────────────────────────────
    with st.expander("Model Configuration"):
        settings = {
            "n_estimators": model.n_estimators,
            "max_depth":    model.max_depth,
            "learning_rate":model.learning_rate,
            "subsample":    model.subsample,
            "colsample_bytree": model.colsample_bytree,
            "reg_alpha":    model.reg_alpha,
            "reg_lambda":   model.reg_lambda,
        }
        for k, v in settings.items():
            st.text(f"  {k:<20}: {v}")


# ─── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "XAUUSD AI Trading System v2.0  |  "
    "Python + XGBoost + pandas-ta + Binance  |  "
    "Demo account only — not financial advice"
)
