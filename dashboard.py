"""
XAUUSD AI Trading System - Real-Time Monitoring Dashboard
Launch: streamlit run dashboard.py
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import time

# Page config
st.set_page_config(
    page_title="XAUUSD AI Trader",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric > div { background-color: #1e1e2e; padding: 10px; border-radius: 8px; }
    .big-number { font-size: 2.5rem; font-weight: bold; }
    .profit { color: #00d26a; }
    .loss { color: #f23645; }
    .neutral { color: #888888; }
    div[data-testid="stSidebar"] { background-color: #0e1117; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=30)
def load_data():
    """Load trading data from saved files."""
    data_dir = PROJECT_ROOT / "data" / "historical"
    trades_path = PROJECT_ROOT / "data" / "backtest_results.json"
    
    # Load candle data
    candles = {}
    for tf in ["M5", "M15", "H1", "H4", "D1"]:
        filepath = data_dir / f"XAUUSD_{tf}.parquet"
        if filepath.exists():
            candles[tf] = pd.read_parquet(filepath)
    
    # Load backtest results if available
    trades = []
    stats = {}
    if trades_path.exists():
        with open(trades_path) as f:
            results = json.load(f)
            trades = results.get("trades", [])
            stats = results.get("stats", {})
    
    return candles, trades, stats


def run_live_scan():
    """Run a live scan of current market conditions."""
    try:
        from src.data_collector.mt5_collector import MT5Connector, DataCollector
        from src.feature_engine.engine import FeatureEngine
        from src.strategies.strategy_manager import StrategyManager
        from src.risk_manager.signal_filter import SignalFilter
        from src.news_filter.news_manager import NewsManager
        
        connector = MT5Connector()
        if not connector.connect():
            return None, None, None, None, None
        
        collector = DataCollector(connector)
        collector.check_symbol()
        
        # Get latest price
        price = collector.get_latest_price()
        
        # Get H1 data (last 500 candles for speed)
        df = collector.fetch_candles("H1", num_bars=500)
        connector.disconnect()
        
        if df.empty:
            return price, None, None, None, None
        
        # Compute features
        engine = FeatureEngine()
        featured_df = engine.compute_features(df, "H1")
        
        # Get signals
        manager = StrategyManager()
        raw_signals = manager.scan_all(featured_df, "H1")
        
        # Filter
        signal_filter = SignalFilter()
        filtered = signal_filter.filter_signals(raw_signals, "H1")
        
        # News check
        news_mgr = NewsManager()
        news_mgr.refresh()
        news_status = news_mgr.get_status()
        
        return price, featured_df, filtered, news_status, raw_signals
        
    except Exception as e:
        st.error(f"Live scan error: {e}")
        return None, None, None, None, None


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/bar-chart-emoji.png", width=60)
    st.title("XAUUSD AI Trader")
    st.caption("Gold Trading Intelligence System")
    
    st.divider()
    
    page = st.radio("📊 Navigation", [
        "🏠 Overview",
        "📈 Live Market",
        "📰 News Intelligence",
        "🎯 Signals",
        "📊 Backtest Results",
        "🧠 ML Model Info",
    ])
    
    st.divider()
    st.caption(f"Last update: {datetime.now().strftime('%H:%M:%S')}")
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ============================================================
# PAGE: Overview
# ============================================================
if page == "🏠 Overview":
    st.title("🥇 XAUUSD AI Trading System")
    st.caption("Real-time monitoring dashboard for your gold trading bot")
    
    # Try live data
    col1, col2, col3, col4 = st.columns(4)
    
    candles, trades, stats = load_data()
    
    # Current price from latest data
    if "H1" in candles and len(candles["H1"]) > 0:
        latest = candles["H1"].iloc[-1]
        prev = candles["H1"].iloc[-2] if len(candles["H1"]) > 1 else latest
        price_change = latest["close"] - prev["close"]
        price_pct = price_change / prev["close"] * 100
        
        col1.metric("💰 Gold Price", f"${latest['close']:.2f}", f"{price_change:+.2f} ({price_pct:+.2f}%)")
    else:
        col1.metric("💰 Gold Price", "No data", "Connect MT5")
    
    if stats:
        col2.metric("📊 Total Trades", stats.get("total_trades", 0))
        
        wr = stats.get("win_rate", 0)
        col3.metric("🎯 Win Rate", f"{wr:.1f}%", 
                     "Good" if wr > 50 else "Needs work",
                     delta_color="normal" if wr > 50 else "inverse")
        
        pnl = stats.get("total_pnl", 0)
        col4.metric("💵 Total PnL", f"${pnl:+.2f}",
                     delta_color="normal" if pnl > 0 else "inverse")
    else:
        col2.metric("📊 Total Trades", "Run backtest first")
        col3.metric("🎯 Win Rate", "-")
        col4.metric("💵 Total PnL", "-")
    
    st.divider()
    
    # Price chart
    if "H1" in candles:
        st.subheader("📈 XAUUSD H1 Chart (Last 200 Candles)")
        df = candles["H1"].tail(200)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.75, 0.25],
                           vertical_spacing=0.05)
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            name="XAUUSD",
            increasing_line_color="#00d26a",
            decreasing_line_color="#f23645"
        ), row=1, col=1)
        
        # Volume
        colors = ["#00d26a" if c >= o else "#f23645" 
                  for c, o in zip(df["close"], df["open"])]
        fig.add_trace(go.Bar(
            x=df.index, y=df["volume"], name="Volume",
            marker_color=colors, opacity=0.5
        ), row=2, col=1)
        
        fig.update_layout(
            height=600,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # System status
    st.subheader("⚙️ System Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        st.markdown("**Modules Built:**")
        modules = [
            ("✅", "Data Collector (MT5)"),
            ("✅", "Feature Engine (133 features)"),
            ("✅", "Strategy Detectors (5 active)"),
            ("✅", "Signal Filter + Risk Manager"),
            ("✅", "ML Signal Scorer (XGBoost)"),
            ("✅", "News Sentiment Filter"),
            ("✅", "Monitoring Dashboard"),
            ("⬜", "Live Paper Trading"),
            ("⬜", "Weekly Auto-Retraining"),
        ]
        for icon, name in modules:
            st.text(f"  {icon} {name}")
    
    with status_col2:
        st.markdown("**Active Strategies:**")
        strategies = [
            ("🟢", "ICT (Order Blocks, FVGs)"),
            ("🟢", "Wyckoff (Springs, Upthrusts)"),
            ("🟢", "RSI Divergence"),
            ("🟢", "Support & Resistance"),
            ("🟢", "CRT (Candle Range Theory)"),
            ("⚪", "SK System (disabled)"),
        ]
        for icon, name in strategies:
            st.text(f"  {icon} {name}")
    
    with status_col3:
        st.markdown("**Risk Rules:**")
        rules = [
            "Risk per trade: 0.5%",
            "Max daily loss: 3%",
            "Max drawdown: 10%",
            "Max positions: 2",
            "Min R:R ratio: 2.0",
            "News blackout: 30 min",
            "Sessions: London + NY",
        ]
        for rule in rules:
            st.text(f"  📋 {rule}")


# ============================================================
# PAGE: Live Market
# ============================================================
elif page == "📈 Live Market":
    st.title("📈 Live Market Analysis")
    
    if st.button("🔄 Run Live Scan (connects to MT5)", use_container_width=True):
        with st.spinner("Connecting to MT5 and scanning market..."):
            price, featured_df, signals, news_status, raw_signals = run_live_scan()
        
        if price:
            col1, col2, col3 = st.columns(3)
            col1.metric("Bid", f"${price['bid']:.2f}")
            col2.metric("Ask", f"${price['ask']:.2f}")
            col3.metric("Spread", f"${price['spread']:.2f}")
        
        if signals:
            st.subheader(f"🎯 {len(signals)} Filtered Signals Found")
            for sig in signals[:10]:
                direction_icon = "🟢" if sig.direction == "BUY" else "🔴"
                st.markdown(
                    f"{direction_icon} **{sig.direction}** @ ${sig.entry_price:.2f} | "
                    f"SL: ${sig.stop_loss:.2f} | TP: ${sig.take_profit:.2f} | "
                    f"R:R {sig.risk_reward:.1f} | Conf: {sig.confidence:.0%} | "
                    f"Strategy: {sig.strategy_name}"
                )
        elif signals is not None:
            st.info("No high-quality signals at this moment. The bot is waiting for better setups.")
        
        if news_status:
            st.subheader("📰 News Status")
            if news_status["is_blackout"]:
                st.error(f"⚠️ NEWS BLACKOUT: {news_status['blackout_reason']}")
            else:
                st.success("✅ No news blackout — trading allowed")
            
            st.text(f"Sentiment: {news_status['sentiment']['direction']} "
                    f"({news_status['sentiment'].get('avg_score', 0):.2f})")
            st.text(f"Events today: {news_status['total_events_today']}")
    else:
        st.info("Click 'Run Live Scan' to connect to MT5 and analyze the current market.")


# ============================================================
# PAGE: News Intelligence
# ============================================================
elif page == "📰 News Intelligence":
    st.title("📰 News Intelligence")
    
    if st.button("🔄 Fetch Latest News", use_container_width=True):
        with st.spinner("Fetching economic calendar and headlines..."):
            try:
                from src.news_filter.news_manager import NewsManager
                mgr = NewsManager()
                mgr.refresh()
                status = mgr.get_status()
                
                # Blackout status
                if status["is_blackout"]:
                    st.error(f"⚠️ NEWS BLACKOUT ACTIVE: {status['blackout_reason']}")
                else:
                    st.success("✅ No news blackout — trading is allowed")
                
                # Sentiment
                sent = status["sentiment"]
                sent_color = "green" if sent["direction"] == "bullish" else ("red" if sent["direction"] == "bearish" else "gray")
                st.markdown(f"### Headline Sentiment: :{sent_color}[{sent['direction'].upper()}] ({sent.get('avg_score', 0):.2f})")
                
                # Events
                st.subheader(f"📅 Today's Events ({status['total_events_today']})")
                for event in mgr.calendar.events:
                    stars = "⭐" * event.gold_impact
                    st.text(f"  {stars} {event.event_time.strftime('%H:%M')} UTC | "
                           f"{event.currency} | {event.name} ({event.impact})")
                
                # Headlines
                if mgr.cached_headlines:
                    st.subheader(f"📰 Latest Headlines ({len(mgr.cached_headlines)})")
                    from src.news_filter.headline_sentiment import HeadlineSentiment
                    scorer = HeadlineSentiment()
                    for h in mgr.cached_headlines[:15]:
                        result = scorer.score_headline(h)
                        if result["score"] > 0.2:
                            st.markdown(f"🟢 **{h}** `({result['score']:+.2f})`")
                        elif result["score"] < -0.2:
                            st.markdown(f"🔴 **{h}** `({result['score']:+.2f})`")
                        else:
                            st.markdown(f"⚪ {h} `({result['score']:+.2f})`")
                            
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("Click 'Fetch Latest News' to get the current economic calendar and headlines.")


# ============================================================
# PAGE: Signals
# ============================================================
elif page == "🎯 Signals":
    st.title("🎯 Strategy Signals")
    st.info("Run a live scan from the 'Live Market' page to see current signals, "
            "or run `python run_strategies.py` in your terminal.")
    
    # Show strategy info
    st.subheader("Active Strategy Descriptions")
    
    strategies_info = {
        "ICT": "Trades Order Block and FVG entries during London/NY killzones after BOS/CHOCH confirmation. Best performer in backtests.",
        "Wyckoff": "Detects Springs (false breakdowns) and Upthrusts (false breakouts) at accumulation/distribution boundaries.",
        "RSI Divergence": "Catches reversals when price and RSI disagree. Strongest near S/R levels.",
        "Support & Resistance": "Enters on confirmed bounces off key horizontal levels with 2+ touches.",
        "CRT": "Candle Range Theory — trades liquidity sweeps of previous candle ranges. Targets opposite extreme.",
    }
    
    for name, desc in strategies_info.items():
        with st.expander(f"📋 {name}"):
            st.write(desc)


# ============================================================
# PAGE: Backtest Results
# ============================================================
elif page == "📊 Backtest Results":
    st.title("📊 Backtest Results")
    st.caption("Run `python run_ml_backtest.py` to generate backtest data")
    
    candles, trades, stats = load_data()
    
    if stats:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trades", stats.get("total_trades", 0))
        col2.metric("Win Rate", f"{stats.get('win_rate', 0):.1f}%")
        col3.metric("Profit Factor", f"{stats.get('profit_factor', 0):.2f}")
        col4.metric("Max Drawdown", f"{stats.get('max_drawdown_pct', 0):.1f}%")
        
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Total PnL", f"${stats.get('total_pnl', 0):+.2f}")
        col6.metric("Return", f"{stats.get('return_pct', 0):+.1f}%")
        col7.metric("Avg Win", f"${stats.get('avg_win', 0):.2f}")
        col8.metric("Avg Loss", f"${stats.get('avg_loss', 0):.2f}")
        
        # Balance curve
        if "balance_curve" in stats:
            st.subheader("💰 Equity Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=stats["balance_curve"],
                mode="lines",
                line=dict(color="#00d26a", width=2),
                fill="tozeroy",
                fillcolor="rgba(0,210,106,0.1)"
            ))
            fig.add_hline(y=3000, line_dash="dash", line_color="gray",
                         annotation_text="Starting Balance")
            fig.update_layout(
                height=400, template="plotly_dark",
                yaxis_title="Balance ($)",
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Strategy breakdown
        if "strategy_breakdown" in stats:
            st.subheader("📋 Performance by Strategy")
            breakdown = stats["strategy_breakdown"]
            
            strat_data = []
            for strat, data in breakdown.items():
                total = data["wins"] + data["losses"]
                wr = data["wins"] / total * 100 if total > 0 else 0
                strat_data.append({
                    "Strategy": strat,
                    "Trades": total,
                    "Wins": data["wins"],
                    "Losses": data["losses"],
                    "Win Rate": f"{wr:.0f}%",
                    "PnL": f"${data['pnl']:+.2f}",
                })
            
            st.dataframe(pd.DataFrame(strat_data), use_container_width=True, hide_index=True)
    else:
        st.warning("No backtest results found. Run `python run_ml_backtest.py` first.")


# ============================================================
# PAGE: ML Model Info
# ============================================================
elif page == "🧠 ML Model Info":
    st.title("🧠 ML Model Information")
    
    model_path = PROJECT_ROOT / "data" / "models" / "xgb_signal_scorer.pkl"
    
    if model_path.exists():
        st.success("✅ ML Model is trained and saved!")
        st.text(f"Model file: {model_path}")
        st.text(f"File size: {model_path.stat().st_size / 1024:.1f} KB")
        
        try:
            import joblib
            data = joblib.load(model_path)
            model = data["model"]
            feature_names = data["feature_names"]
            
            st.subheader("Model Details")
            st.text(f"Type: XGBoost Classifier")
            st.text(f"Features: {len(feature_names)}")
            st.text(f"Trees: {model.n_estimators}")
            st.text(f"Max depth: {model.max_depth}")
            
            # Feature importance
            st.subheader("🔑 Top Predictive Features")
            importances = model.feature_importances_
            top = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:15]
            
            fig = go.Figure(go.Bar(
                x=[imp for _, imp in top],
                y=[name for name, _ in top],
                orientation="h",
                marker_color="#2E75B6"
            ))
            fig.update_layout(
                height=500, template="plotly_dark",
                xaxis_title="Importance",
                yaxis=dict(autorange="reversed"),
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("All Features")
            st.dataframe(
                pd.DataFrame(top, columns=["Feature", "Importance"]).round(4),
                use_container_width=True, hide_index=True
            )
            
        except Exception as e:
            st.error(f"Error loading model: {e}")
    else:
        st.warning("No ML model found. Run `python run_ml_backtest.py` first to train the model.")


# Footer
st.divider()
st.caption("XAUUSD AI Trading System v1.0 | Built with Python, XGBoost, and pandas-ta | ⚠️ Demo only — not financial advice")
