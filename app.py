import os
import sys
import time
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_SCRIPT_DIR, ".env"), override=True)
except ImportError:
    pass

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

from services.sentiment import analyze_single, analyze_batch, analyze_by_ticker
from services.ai_service import generate_insights, chat_with_data
from services.data_service import parse_csv, get_texts_list, load_sample_data
from services.news_service import fetch_stock_headlines, has_news_api_key
from services.ticker_service import (
    EXCHANGE_OPTIONS,
    NSE_POPULAR,
    BSE_POPULAR,
    US_POPULAR,
    normalize_ticker,
    tradingview_widget_html,
)
from services.price_service import fetch_live_quote
from services.finnhub_service import get_company_profile, get_quote, get_company_news, get_market_news, has_finnhub_key
from services.alpha_vantage_service import get_daily_ohlc, get_sma, has_alpha_key
from ui.shapegrid_bg import render_shapegrid_background

# ── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSentimentIQ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── SHAPEGRID BACKGROUND ────────────────────────────────────────────────────
render_shapegrid_background()

# ── APP-LEVEL CSS (PREMIUM DARK UI) ──────────────────────────────────────────
st.markdown(
    """
<style>
/* Base overrides */
html {
    background: #000000 !important;
}
body, [data-testid="stAppViewContainer"], .main, header[data-testid="stHeader"] {
    background: transparent !important;
}

/* ── Glassmorphism Cards ── */
.bullish-card, .bearish-card, .neutral-card, .opportunity-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    border-radius: 12px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    padding: 16px;
    margin-bottom: 8px;
    color: #f1f5f9;
}
.bullish-card { border-left: 4px solid #10b981; }
.bearish-card { border-left: 4px solid #ef4444; }
.neutral-card { border-left: 4px solid #64748b; }
.opportunity-card { border-left: 4px solid #a78bfa; }

/* ── Premium Header ── */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=Inter:wght@400;500;600&display=swap');

@keyframes gradient-text-animation {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.app-header-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: 3.4rem !important;
    font-weight: 800 !important;
    letter-spacing: -1.5px !important;
    /* React Bits GradientText implementation (Pure CSS) */
    background: linear-gradient(to right, #5227FF, #FF9FFC, #B497CF, #5227FF) !important;
    background-size: 300% 100% !important;
    animation: gradient-text-animation 8s ease-in-out infinite !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    line-height: 1.1 !important;
    margin-bottom: 0 !important;
    display: inline-block;
}
.app-header-sub {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.15rem !important;
    color: #94a3b8 !important;
    margin-top: 6px !important;
}
.badge {
    display: inline-flex !important;
    align-items: center !important;
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 6px !important;
    padding: 3px 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
    color: #94a3b8 !important;
    margin-bottom: 6px !important;
}

/* ── Tab Bar Styling ── */
[data-testid="stTabs"] {
    background: rgba(255,255,255,0.02);
    border-radius: 12px;
    padding: 8px;
}
[data-testid="stTabs"] button[data-baseweb="tab"] {
    color: #e2e8f0 !important;
    background: transparent;
    font-weight: 600;
    border: none;
    border-bottom: 2px solid transparent !important;
    border-radius: 0;
}
[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 2px solid #c084fc !important;
}
div[role="tablist"] {
    border-bottom: 1px solid rgba(255,255,255,0.08);
}

/* ── Metric Cards ── */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px;
}
div[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 800 !important;
}
div[data-testid="stMetricDelta"] svg[title="m-up"] + div { color: #10b981 !important; }
div[data-testid="stMetricDelta"] svg[title="m-down"] + div { color: #ef4444 !important; }

/* ── Buttons ── */
button[kind="primary"] {
    background: rgba(99, 102, 241, 0.1) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    color: #c084fc !important;
    font-weight: 600 !important;
    transition: all 0.2s ease;
}
button[kind="primary"]:hover {
    background: rgba(99, 102, 241, 0.2) !important;
    border: 1px solid rgba(99, 102, 241, 0.5) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.2) !important;
    color: #ffffff !important;
}
button[kind="secondary"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
}

/* ── Input Fields ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.2) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    background: rgba(15,23,42,0.8) !important;
}
th {
    background: rgba(99,102,241,0.15) !important;
    color: #a78bfa !important;
}
tr:nth-child(even) {
    background: rgba(255,255,255,0.02) !important;
}

/* ── Chat Messages ── */
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border-radius: 16px;
    padding: 12px;
}
.stChatMessage:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(255,255,255,0.06) !important;
    color: #f1f5f9 !important;
    border-radius: 16px;
    padding: 12px;
}
.stChatAvatar {
    width: 36px !important;
    height: 36px !important;
    background: rgba(255,255,255,0.1) !important;
    backdrop-filter: blur(4px);
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 3px; }

/* ── Dividers & Alerts ── */
hr { border-color: rgba(255,255,255,0.08) !important; }

[data-testid="stAlert"] {
    backdrop-filter: blur(12px);
}
[data-testid="stAlert"]:has(div:contains("success")), .stSuccess {
    background: rgba(16,185,129,0.12) !important;
    border: 1px solid rgba(16,185,129,0.3) !important;
}
[data-testid="stAlert"]:has(div:contains("error")), .stError {
    background: rgba(239,68,68,0.12) !important;
    border: 1px solid rgba(239,68,68,0.3) !important;
}
[data-testid="stAlert"]:has(div:contains("warning")), .stWarning {
    background: rgba(245,158,11,0.12) !important;
    border: 1px solid rgba(245,158,11,0.3) !important;
}
[data-testid="stAlert"]:has(div:contains("info")), .stInfo {
    background: rgba(99,102,241,0.12) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
}
</style>
    """,
    unsafe_allow_html=True,
)

# ── SESSION STATE DEFAULTS ───────────────────────────────────────────────────
DEFAULTS = {
    "batch_results": None,
    "batch_summary": None,
    "ticker_summary": None,
    "batch_df": None,
    "chat_history": [],
    "single_result": None,
    "ai_insights": None,
    "live_results": None,
    "live_summary": None,
    "live_quote": None,
    "live_ticker_meta": None,
    "pending_chat_question": None,
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── HELPERS ──────────────────────────────────────────────────────────────────
def signal_color(signal: str) -> str:
    if signal == "Bullish": return "#10b981"
    if signal == "Bearish": return "#ef4444"
    return "#64748b"

def signal_emoji(signal: str) -> str:
    if signal == "Bullish": return "🟢"
    if signal == "Bearish": return "🔴"
    return "⚪"

def get_market_signal_from_compound(compound: float) -> str:
    if compound >= 0.05: return "Bullish"
    if compound <= -0.05: return "Bearish"
    return "Neutral"

def enrich_results(df: pd.DataFrame, results: list) -> list:
    df = df.reset_index(drop=True)
    for i in range(min(len(df), len(results))):
        results[i]["ticker"] = df.loc[i, "ticker"] if "ticker" in df.columns else "UNKNOWN"
        results[i]["source"] = df.loc[i, "source"] if "source" in df.columns else "Unknown"
        if "publishedAt" in df.columns:
            results[i]["publishedAt"] = df.loc[i, "publishedAt"]
        elif "date" in df.columns:
            results[i]["publishedAt"] = df.loc[i, "date"]
    return results

def process_batch_dataframe(df: pd.DataFrame):
    texts = get_texts_list(df)
    results, summary = analyze_batch(texts)
    results = enrich_results(df.reset_index(drop=True), results)
    ticker_summary = analyze_by_ticker(results)
    st.session_state.batch_df = df
    st.session_state.batch_results = results
    st.session_state.batch_summary = summary
    st.session_state.ticker_summary = ticker_summary
    st.session_state.ai_insights = None
    st.session_state.chat_history = []

def market_health_display(summary: dict) -> str:
    avg = summary.get("avg_compound", 0.0)
    score = round((avg + 1) / 2 * 100, 1)
    return f"{score}/100"

def insight_container(itype: str, title: str, body: str):
    content = f"**{title}**\n\n{body}"
    if itype == "bullish": st.success(content)
    elif itype == "bearish": st.error(content)
    elif itype == "warning": st.warning(content)
    else: st.info(content)

def render_kpi_cards(summary: dict):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Total Headlines", summary["total"])
    with c2: st.metric("Bullish %", f"{summary['bullish_pct']}%", delta=f"{summary['bullish_count']} headlines")
    with c3: st.metric("Bearish %", f"{summary['bearish_pct']}%", delta=f"-{summary['bearish_count']} headlines")
    with c4: st.metric("Overall Signal", summary.get("overall_signal", "MIXED ⚪"))
    with c5: st.metric("Market Health Score", market_health_display(summary))

# ── HEADER ───────────────────────────────────────────────────────────────────
h_left, h_right = st.columns([3, 1])
with h_left:
    st.markdown(
        '<div style="padding-top: 0.5rem;">'
        '<div class="app-header-title">StockSentimentIQ</div>'
        '<div class="app-header-sub">Real-time Stock Market Sentiment Analysis powered by AI</div>'
        '</div>',
        unsafe_allow_html=True,
    )
with h_right:
    st.markdown(
        '<div style="text-align: right; padding-top: 1rem;">'
        '<span class="badge">Groq LLaMA 3.3 70B</span><br>'
        '<span class="badge">VADER NLP</span>'
        '</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Analyze Headline",
    "Batch Analysis",
    "AI Market Insights",
    "Chat with Data",
    "Live Market",
    "Market Overview",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYZE HEADLINE
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Analyze a single market headline")
    headline = st.text_area(
        "Financial news headline", height=120, key="single_headline_input",
        placeholder="e.g. NVIDIA reports blowout quarter driven by AI chip demand beating all analyst expectations"
    )
    ticker_opt = st.text_input("Ticker (optional)", placeholder="NVDA", key="single_ticker")

    if st.button("Analyze Headline", use_container_width=True, type="primary"):
        if not (headline or "").strip():
            st.warning("Enter a market headline to analyze.")
        else:
            st.session_state.single_result = analyze_single(headline)

    examples = [
        ("AAPL beats earnings", "Apple beats earnings expectations with record iPhone sales driving massive revenue growth"),
        ("Fed rate fears", "Federal Reserve signals more rate hikes ahead sending markets into freefall today"),
        ("NVDA blowout", "NVIDIA reports blowout quarter driven by insane AI chip demand from hyperscale data centers"),
        ("TSLA miss", "Tesla deliveries disappoint analysts as production issues continue to plague the company"),
    ]
    st.markdown("**Quick examples:**")
    ex_cols = st.columns(4)
    for col, (label, text) in zip(ex_cols, examples):
        with col:
            if st.button(label, use_container_width=True, key=f"ex_{label}"):
                st.session_state.single_result = analyze_single(text)
                st.rerun()

    if st.session_state.single_result:
        r = st.session_state.single_result
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Signal", f"{signal_emoji(r['signal'])} {r['signal']}")
        with m2: st.metric("Strength", r["signal_strength"])
        with m3: st.metric("Sentiment Score", r["sentiment_score"])
        with m4: st.metric("Confidence", f"{r['confidence']}%")

        if ticker_opt:
            st.caption(f"Ticker context: **{ticker_opt.upper()}**")

        st.progress(float(r["positive"]), text=f"Bullish Score: {float(r['positive']):.0%}")
        st.progress(float(r["negative"]), text=f"Bearish Score: {float(r['negative']):.0%}")
        st.progress(float(r["neutral"]), text=f"Neutral Score:  {float(r['neutral']):.0%}")

        compound = r["compound"]
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=compound,
            number={"font": {"size": 28, "color": "#f1f5f9"}},
            title={"text": "Sentiment Gauge", "font": {"size": 20, "color": "#f1f5f9"}},
            gauge={
                "axis": {"range": [-1, 1], "tickcolor": "rgba(255,255,255,0.1)"},
                "bar": {"color": signal_color(get_market_signal_from_compound(compound))},
                "bgcolor": "rgba(15,30,60,0.6)",
                "bordercolor": "rgba(255,255,255,0.1)",
                "steps": [
                    {"range": [-1, -0.05], "color": "rgba(239,68,68,0.1)"},
                    {"range": [-0.05, 0.05], "color": "rgba(100,116,139,0.1)"},
                    {"range": [0.05, 1], "color": "rgba(16,185,129,0.1)"},
                ],
                "threshold": {
                    "line": {"color": "#f1f5f9", "width": 4},
                    "thickness": 0.85, "value": compound,
                },
            },
        ))
        fig.add_annotation(
            x=0.5, y=0.1, text=f"Strong {r['signal']} Signal" if abs(compound) >= 0.5 else f"{r['signal']} Signal",
            showarrow=False, font=dict(size=18, color=signal_color(r['signal']))
        )
        fig.update_layout(
            margin=dict(t=40, b=20, l=20, r=20), height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,30,60,0.6)",
        )
        st.plotly_chart(fig, use_container_width=True)

        sig_class = "bullish-card" if r["signal"] == "Bullish" else "bearish-card" if r["signal"] == "Bearish" else "neutral-card"
        st.markdown(f'<div class="{sig_class}"><strong>{r["signal"]}</strong> market tone — score {r["sentiment_score"]}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Batch analysis of market headlines")
    up_col, sample_col = st.columns([2, 1])
    with up_col:
        uploaded = st.file_uploader("Upload headlines CSV (text, ticker, source)", type=["csv"])
    with sample_col:
        st.markdown("")
        if st.button("Load Sample Data", use_container_width=True):
            with st.spinner("Analyzing market headlines..."):
                df = load_sample_data()
                process_batch_dataframe(df)
            st.success(f"Loaded {len(df)} sample headlines.")
            st.rerun()

    if uploaded is not None:
        try:
            with st.spinner("Analyzing market headlines..."):
                df = parse_csv(uploaded)
                process_batch_dataframe(df)
            st.success(f"Analyzed {len(st.session_state.batch_results)} headlines.")
            st.rerun()
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Failed to process CSV: {e}")

    if st.session_state.batch_results and st.session_state.batch_summary:
        summary = st.session_state.batch_summary
        results = st.session_state.batch_results
        df      = st.session_state.batch_df

        render_kpi_cards(summary)

        # Charts
        c1, c2 = st.columns(2)
        with c1:
            # Donut Chart
            pie_df = pd.DataFrame({
                "Signal": ["Bullish", "Bearish", "Neutral"],
                "Count": [summary["bullish_count"], summary["bearish_count"], summary["neutral_count"]],
            })
            fig_pie = px.pie(
                pie_df, names="Signal", values="Count", color="Signal",
                color_discrete_map={"Bullish": "#10b981", "Bearish": "#ef4444", "Neutral": "#64748b"},
                hole=0.45, title="Headline Sentiment Mix"
            )
            fig_pie.update_traces(hovertemplate="%{label}<br>Count: %{value}<br>Share: %{percent}<extra></extra>")
            fig_pie.add_annotation(text=summary.get("overall_signal", "MIXED ⚪"), x=0.5, y=0.5, font_size=18, showarrow=False)
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,30,60,0.6)",
                font=dict(color="#f1f5f9"), margin=dict(t=40, b=20, l=10, r=10), height=320
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            # Histogram
            compounds = [r["compound"] for r in results]
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=[c for c in compounds if c >= 0], name="Bullish", marker_color="#10b981"))
            fig_hist.add_trace(go.Histogram(x=[c for c in compounds if c < 0], name="Bearish", marker_color="#ef4444"))
            fig_hist.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
            fig_hist.add_annotation(x=-0.5, y=1, yref="paper", text="Bearish Zone", font=dict(color="#ef4444"), showarrow=False)
            fig_hist.add_annotation(x=0.5, y=1, yref="paper", text="Bullish Zone", font=dict(color="#10b981"), showarrow=False)
            fig_hist.update_layout(
                barmode="overlay", title="Score Distribution", bargap=0.1,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,30,60,0.6)",
                font=dict(color="#f1f5f9"), margin=dict(t=40, b=20, l=10, r=10), height=320,
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"), yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            # Ticker Bar
            ticker_summary = st.session_state.ticker_summary
            if ticker_summary:
                tdf = pd.DataFrame([{"Ticker": v["ticker"], "Avg": v["avg_compound"]} for v in ticker_summary.values()])
                tdf = tdf.sort_values("Avg", ascending=False)
                fig_bar = go.Figure(go.Bar(
                    x=tdf["Avg"], y=tdf["Ticker"], orientation="h",
                    marker=dict(color=tdf["Avg"], colorscale="RdYlGn", cmin=-1, cmax=1, line=dict(color="white", width=0.5)),
                    text=[f"{v:+.2f}" for v in tdf["Avg"]], textposition="outside",
                ))
                fig_bar.update_layout(
                    title="Sentiment by Ticker",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,30,60,0.6)",
                    font=dict(color="#f1f5f9"), margin=dict(t=40, b=20, l=10, r=10), height=320,
                    xaxis=dict(tickvals=[-0.5, 0, 0.5], gridcolor="rgba(255,255,255,0.05)"), yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No ticker data available for chart.")

        with c4:
            # Confidence Scatter
            s_df = pd.DataFrame(results)
            if not s_df.empty:
                s_df["short_text"] = s_df["text"].str[:80] + "..."
                fig_scat = px.scatter(
                    s_df, x="compound", y="confidence", color="signal",
                    color_discrete_map={"Bullish": "#10b981", "Bearish": "#ef4444", "Neutral": "#64748b"},
                    hover_data={"short_text": True, "compound": True, "confidence": True, "signal": False},
                    title="Confidence Distribution"
                )
                fig_scat.update_traces(marker=dict(size=8))
                fig_scat.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,30,60,0.6)",
                    font=dict(color="#f1f5f9"), margin=dict(t=40, b=20, l=10, r=10), height=320,
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"), yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
                )
                st.plotly_chart(fig_scat, use_container_width=True)

        # Timeline (if publishedAt exists)
        if "publishedAt" in df.columns:
            tl_df = pd.DataFrame(results)
            if "publishedAt" in tl_df.columns:
                tl_df["date"] = pd.to_datetime(tl_df["publishedAt"], errors="coerce").dt.date
                tl_df = tl_df.dropna(subset=["date"])
                if not tl_df.empty:
                    daily = tl_df.groupby("date")["compound"].mean().reset_index()
                    fig_line = go.Figure()
                    # Color segments
                    fig_line.add_trace(go.Scatter(
                        x=daily["date"], y=daily["compound"], mode="lines+markers",
                        line=dict(color="#a78bfa", width=3),
                        marker=dict(size=8, color=np.where(daily["compound"]>0, "#10b981", "#ef4444"))
                    ))
                    fig_line.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                    fig_line.update_layout(
                        title="Average Sentiment Timeline",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,30,60,0.6)",
                        font=dict(color="#f1f5f9"), margin=dict(t=40, b=20, l=10, r=10), height=250,
                        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"), yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
                    )
                    st.plotly_chart(fig_line, use_container_width=True)

        st.markdown("#### Filtered headline table")
        f1, f2, f3 = st.columns([1, 1, 2])
        with f1: sig_filter = st.selectbox("Signal", ["All", "Bullish", "Bearish", "Neutral"])
        with f2:
            tickers = sorted({r.get("ticker", "UNKNOWN") for r in results})
            ticker_filter = st.multiselect("Ticker", tickers, default=tickers)
        with f3: search = st.text_input("Search headlines", placeholder="Filter by keyword...")

        table_df = pd.DataFrame(results)
        if sig_filter != "All": table_df = table_df[table_df["signal"] == sig_filter]
        if ticker_filter: table_df = table_df[table_df["ticker"].isin(ticker_filter)]
        if search.strip(): table_df = table_df[table_df["text"].str.contains(search.strip(), case=False, na=False)]

        display = table_df[["text", "ticker", "signal", "sentiment_score", "confidence", "source"]].rename(columns={
            "text": "Headline", "ticker": "Ticker", "signal": "Signal",
            "sentiment_score": "Sentiment Score", "confidence": "Confidence %", "source": "Source",
        })
        st.dataframe(display, use_container_width=True, height=360)

        csv_bytes = display.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Results as CSV", data=csv_bytes, file_name="stock_sentiment_headlines.csv",
            mime="text/csv", use_container_width=True,
        )
    else:
        st.info("Upload a CSV or load sample data to run batch market sentiment analysis.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — AI MARKET INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### AI market insights (Groq Llama 3.3 70B)")
    if not st.session_state.batch_summary:
        st.info("Run **Batch Analysis** first to enable AI market insights.")
    else:
        summary = st.session_state.batch_summary
        with st.expander("Current dataset stats", expanded=False):
            st.json(summary)

        if st.button("Generate AI Analysis", use_container_width=True, type="primary"):
            with st.spinner("🤖 Generating AI market analysis with LLaMA 3.3 70B..."):
                st.session_state.ai_insights = generate_insights(summary)

        insights = st.session_state.ai_insights
        if insights:
            risk = insights.get("risk_level", "MEDIUM").upper()
            if risk == "LOW": st.success(f"**Risk Level:** {risk}")
            elif risk == "HIGH": st.error(f"**Risk Level:** {risk}")
            else: st.warning(f"**Risk Level:** {risk}")

            st.info(f"**Executive summary:** {insights.get('executive_summary', '')}")

            market_call = insights.get("market_call", "")
            if market_call:
                st.markdown(f'<div class="opportunity-card"><strong>Market call:</strong> {market_call}</div>', unsafe_allow_html=True)

            cards = insights.get("insights", [])[:4]
            r1c1, r1c2 = st.columns(2)
            r2c1, r2c2 = st.columns(2)
            for slot, item in zip([r1c1, r1c2, r2c1, r2c2], cards):
                with slot:
                    title = f"{item.get('icon','📌')} {item.get('title','Insight')}"
                    insight_container(item.get("type", "opportunity"), title, item.get("body", ""))

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — CHAT WITH DATA
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Chat with your market headline dataset")
    if not st.session_state.batch_summary:
        st.info("Run **Batch Analysis** first to enable data chat.")
    else:
        summary = st.session_state.batch_summary

        if st.session_state.pending_chat_question:
            q = st.session_state.pending_chat_question
            st.session_state.pending_chat_question = None
            st.session_state.chat_history.append({"role": "user", "content": q})
            with st.spinner("🤖 Analyzing..."):
                answer = chat_with_data(q, summary, st.session_state.chat_history[:-1])
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        st.markdown("**Quick questions:**")
        starters = [
            "What is the overall market signal?", "Which stocks are most bearish?",
            "What percentage of headlines are bullish?", "What are the top risk factors?",
            "How many strong conviction signals exist?", "What is the average sentiment score?",
        ]
        s1, s2, s3 = st.columns(3)
        s4, s5, s6 = st.columns(3)
        for col, q in zip([s1, s2, s3, s4, s5, s6], starters):
            with col:
                if st.button(q, use_container_width=True, key=f"chat_starter_{q[:24]}"):
                    st.session_state.pending_chat_question = q
                    st.rerun()

        if prompt := st.chat_input("Ask anything about your market headline data..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.spinner("🤖 Analyzing..."):
                answer = chat_with_data(prompt, summary, st.session_state.chat_history[:-1])
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — LIVE MARKET
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Live market — any stock (NSE, BSE, US & global)")
    auto_refresh = st.checkbox("🔄 Auto-refresh every 30 seconds")
    if auto_refresh:
        # Simple loop limitation to avoid infinite recursion hanging Streamlit Cloud
        if 'refresh_count' not in st.session_state: st.session_state.refresh_count = 0
        if st.session_state.refresh_count < 10:
            time.sleep(30)
            st.session_state.refresh_count += 1
            st.rerun()
        else:
            st.warning("Auto-refresh paused after 10 loops to save API quota.")

    ex_col, sym_col = st.columns([1, 2])
    with ex_col:
        exchange = st.selectbox("Stock exchange", list(EXCHANGE_OPTIONS.keys()))
    with sym_col:
        picks = {}
        if "NSE" in exchange: picks = {name: sym for name, sym in NSE_POPULAR}
        elif "BSE" in exchange: picks = {name: sym for name, sym in BSE_POPULAR}
        elif exchange.startswith("US"): picks = {name: sym for name, sym in US_POPULAR}

        if picks:
            pick_name = st.selectbox("Quick pick (optional)", ["Type symbol below"] + list(picks.keys()))
            if pick_name != "Type symbol below": st.session_state.live_symbol_prefill = picks[pick_name]

        symbol_input = st.text_input("Symbol", value=st.session_state.get("live_symbol_prefill", ""), placeholder="e.g. AAPL, TSLA", key="live_symbol_input")

    news_extra = st.text_input("News search term (optional)", placeholder="Company name for better headlines")

    if st.button("Fetch Live Price & Headlines", use_container_width=True, type="primary"):
        try:
            meta = normalize_ticker(symbol_input, exchange)
            st.session_state.live_ticker_meta = meta
            tv_sym = meta["tradingview_symbol"]
            yahoo_sym = meta["yahoo_symbol"]
            bare_sym = yahoo_sym.split(".")[0]

            with st.spinner("🔄 Fetching real-time price from Finnhub..."):
                quote = get_quote(bare_sym)
                if quote.get("ok"):
                    quote["source"] = "Finnhub"
                else:
                    quote = fetch_live_quote(yahoo_sym)
                    if quote.get("ok"):
                        quote["source"] = "Yahoo Finance"
            st.session_state.live_quote = quote

            with st.spinner("📰 Loading latest company news..."):
                news_q = (news_extra or meta["news_query"]).strip()
                news_results = []
                if has_finnhub_key() and exchange.startswith("US"):
                    news_results = get_company_news(bare_sym)
                if not news_results and has_news_api_key():
                    raw = fetch_stock_headlines(news_q, max_results=20, exchange_label=exchange)
                    if raw:
                        texts = [h["text"] for h in raw]
                        r, summ = analyze_batch(texts)
                        for i, h in enumerate(raw):
                            if i < len(r):
                                r[i]["source"] = h.get("source", "Unknown")
                                r[i]["url"] = h.get("url", "")
                                r[i]["publishedAt"] = h.get("publishedAt", "")
                                r[i]["ticker"] = meta["display_ticker"]
                        news_results = r
                
                st.session_state.live_results = news_results
        except ValueError as e:
            st.error(str(e))

    meta = st.session_state.live_ticker_meta
    quote = st.session_state.live_quote

    if quote and quote.get("ok"):
        st.markdown(f"**Data Source:** {quote.get('source', 'Unknown')}")
        
        c1, c2 = st.columns([1, 3])
        with c1:
            prof = get_company_profile(meta["yahoo_symbol"].split(".")[0]) if meta and has_finnhub_key() else {}
            st.markdown('<div class="neutral-card">', unsafe_allow_html=True)
            if prof.get("ok"):
                if prof["logo"]: st.image(prof["logo"], width=60)
                st.markdown(f"### **{prof['name']}**")
                st.markdown(f"<span class='badge'>{prof['industry']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Market Cap:** {prof['market_cap']}<br>**Country:** {prof['country']}<br>**Exchange:** {prof['exchange']}", unsafe_allow_html=True)
            else:
                st.markdown(f"### **{quote.get('name', meta['yahoo_symbol'] if meta else '')}**")
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            curr = quote.get("currency", "") or (meta["currency"] if meta else "")
            with m1: st.metric("Current Price", f"{quote['price']:.2f} {curr}")
            with m2: st.metric("Change", f"{quote['change']:+.2f}")
            with m3: st.metric("Change %", f"{quote['change_pct']:+.2f}%")
            with m4: st.metric("Day High", f"{quote.get('high', 0):.2f}")
            with m5: st.metric("Day Low", f"{quote.get('low', 0):.2f}")
            with m6: st.metric("Volume", f"{quote.get('volume', 0):,}")
            
        with st.spinner("📊 Processing candlestick data..."):
            ohlc = pd.DataFrame()
            if has_alpha_key() and meta and exchange.startswith("US"):
                ohlc = get_daily_ohlc(meta["yahoo_symbol"].split(".")[0])
            if ohlc.empty and "history" in quote:
                hist = quote["history"]
                if all(c in hist.columns for c in ["open", "high", "low", "close"]):
                    ohlc = hist

            if not ohlc.empty:
                fig_c = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.03)
                fig_c.add_trace(go.Candlestick(
                    x=ohlc["date"], open=ohlc["open"], high=ohlc["high"], low=ohlc["low"], close=ohlc["close"],
                    increasing_line_color="#10b981", decreasing_line_color="#ef4444", name="Price"
                ), row=1, col=1)
                
                # SMA Overlay
                if has_alpha_key() and meta and exchange.startswith("US"):
                    sma = get_sma(meta["yahoo_symbol"].split(".")[0])
                    if not sma.empty:
                        fig_c.add_trace(go.Scatter(x=sma["date"], y=sma["sma"], mode="lines", line=dict(color="#a78bfa", width=1.5), name="20-day SMA"), row=1, col=1)

                fig_c.add_trace(go.Bar(
                    x=ohlc["date"], y=ohlc["volume"], marker_color="rgba(59,130,246,0.3)", name="Volume"
                ), row=2, col=1)
                
                fig_c.update_layout(
                    title="Price History", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,30,60,0.6)",
                    font=dict(color="#f1f5f9"), margin=dict(t=40, b=20, l=10, r=10), height=500,
                    xaxis_rangeslider_visible=False,
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1W", step="week", stepmode="backward"),
                                dict(count=1, label="1M", step="month", stepmode="backward"),
                                dict(count=3, label="3M", step="month", stepmode="backward"),
                                dict(count=6, label="6M", step="month", stepmode="backward"),
                                dict(count=1, label="1Y", step="year", stepmode="backward"),
                            ]),
                            bgcolor="#1e293b", activecolor="#6366f1"
                        )
                    )
                )
                fig_c.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
                fig_c.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
                st.plotly_chart(fig_c, use_container_width=True)
            else:
                st.warning("OHLC data unavailable for this ticker.")

        if meta:
            tv_sym = meta["tradingview_symbol"]
            with st.expander("Advanced TradingView Chart"):
                components.html(tradingview_widget_html(tv_sym, height=450), height=460, scrolling=True)

    if st.session_state.live_results:
        st.markdown("#### Live News Sentiment")
        res = st.session_state.live_results
        
        # Trend chart
        dates = []
        scores = []
        for r in res:
            dt = r.get("datetime") or r.get("publishedAt") or r.get("date")
            if dt:
                dates.append(pd.to_datetime(dt[:10]))
                scores.append(r["compound"])
        if dates:
            tdf = pd.DataFrame({"Date": dates, "Score": scores}).groupby("Date").mean().reset_index()
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=tdf["Date"], y=tdf["Score"], mode="lines+markers", line=dict(color="#6366f1", width=2)))
            fig_trend.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
            fig_trend.update_layout(
                title="News Sentiment Trend (last 7 days)", height=250,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,30,60,0.6)",
                font=dict(color="#f1f5f9"), margin=dict(t=40, b=20, l=10, r=10),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"), yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        for item in res:
            sig = item.get("sentiment_signal", item.get("signal", "Neutral"))
            css = "bullish-card" if sig == "Bullish" else "bearish-card" if sig == "Bearish" else "neutral-card"
            url = item.get("url", "")
            date = item.get("datetime") or item.get("publishedAt", "")[:10]
            link = f' <a href="{url}" target="_blank" style="color:#a78bfa">Read →</a>' if url else ""
            headline = item.get("headline", item.get("text", ""))
            score = item.get("compound", item.get("sentiment_score", 0))
            
            st.markdown(
                f'<div class="{css}">'
                f'<strong>{signal_emoji(sig)} {sig}</strong> · '
                f'<span style="color:{signal_color(sig)}">{score}</span> · '
                f'{item.get("source","Unknown")} · {date}'
                f'<br><br>{headline}{link}'
                f'<div style="height:2px;background:{"#10b981" if sig=="Bullish" else "#ef4444" if sig=="Bearish" else "#64748b"};width:{abs(score)*100}%;margin-top:8px"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — MARKET OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### Global Market Overview")
    
    m1, m2 = st.columns([1, 2])
    with m1:
        st.markdown("#### Market Mood Meter")
        with st.spinner("Fetching general market news..."):
            m_news = get_market_news("general")
        if m_news:
            avg = sum(n["compound"] for n in m_news) / len(m_news)
            score = (avg + 1) / 2 * 100
            
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                number={"font": {"size": 28, "color": "#f1f5f9"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "rgba(255,255,255,0.1)"},
                    "bar": {"color": "#6366f1"},
                    "bgcolor": "rgba(15,30,60,0.6)",
                    "bordercolor": "rgba(255,255,255,0.1)",
                    "steps": [
                        {"range": [0, 20], "color": "#7f1d1d"},
                        {"range": [21, 40], "color": "#ef4444"},
                        {"range": [41, 60], "color": "#64748b"},
                        {"range": [61, 80], "color": "#10b981"},
                        {"range": [81, 100], "color": "#14532d"},
                    ],
                }
            ))
            fig_g.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_g, use_container_width=True)
            
            lbl = "Extreme Greed 🤑" if score > 80 else "Greed 😊" if score > 60 else "Neutral 😐" if score > 40 else "Fear 😰" if score > 20 else "Extreme Fear 😱"
            st.markdown(f"<center><h3>{lbl}</h3></center>", unsafe_allow_html=True)
        else:
            st.info("Market Mood unavailable (requires Finnhub API Key).")

    with m2:
        st.markdown("#### Today's Top Movers")
        US_WATCHLIST = ["AAPL","MSFT","NVDA","TSLA","AMZN","GOOGL","META","JPM","NFLX","COIN"]
        movers = []
        with st.spinner("Fetching Top Movers..."):
            for s in US_WATCHLIST:
                q = get_quote(s)
                if q.get("ok"):
                    movers.append({"Ticker": s, "Change%": q["change_pct"]})
        
        if movers:
            mdf = pd.DataFrame(movers).sort_values("Change%", ascending=True)
            fig_m = go.Figure(go.Bar(
                x=mdf["Change%"], y=mdf["Ticker"], orientation="h",
                marker_color=np.where(mdf["Change%"] > 0, "#10b981", "#ef4444"),
                text=[f"{v:+.2f}%" for v in mdf["Change%"]], textposition="outside"
            ))
            fig_m.update_layout(
                height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,30,60,0.6)",
                font=dict(color="#f1f5f9"), margin=dict(t=20, b=20, l=10, r=20),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"), yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_m, use_container_width=True)

    st.markdown("#### Sector Sentiment")
    SECTORS = {
        "Technology": "tech stocks AI semiconductor",
        "Energy": "oil gas crude energy stocks",
        "Finance": "banking stocks Fed interest rates",
        "Healthcare": "pharma biotech FDA approval",
        "Consumer": "retail consumer spending inflation",
    }
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    slots = [sc1, sc2, sc3, sc4, sc5]
    for (sec, kw), slot in zip(SECTORS.items(), slots):
        with slot:
            if has_news_api_key():
                sh = fetch_stock_headlines(kw, max_results=5)
                if sh:
                    res, s_sum = analyze_batch([x["text"] for x in sh])
                    sig = s_sum["overall_signal"].split(" ")[0].capitalize()
                    css = "bullish-card" if sig == "Bullish" else "bearish-card" if sig == "Bearish" else "neutral-card"
                    st.markdown(f'<div class="{css}"><center><b>{sec}</b><br>{signal_emoji(sig)} {sig}</center></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="neutral-card"><center><b>{sec}</b><br>⚪ N/A</center></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="neutral-card"><center><b>{sec}</b><br>⚪ Key Req</center></div>', unsafe_allow_html=True)

    st.markdown("#### Ticker Coverage Heat Map")
    all_res = []
    if st.session_state.batch_results: all_res.extend(st.session_state.batch_results)
    if st.session_state.live_results: all_res.extend(st.session_state.live_results)
    
    if all_res:
        t_df = pd.DataFrame(all_res)
        if "ticker" in t_df.columns:
            counts = t_df.groupby("ticker").agg(Count=("text", "count"), AvgScore=("compound", "mean")).reset_index()
            counts = counts[counts["ticker"] != "UNKNOWN"]
            if not counts.empty:
                fig_tm = px.treemap(
                    counts, path=["ticker"], values="Count", color="AvgScore",
                    color_continuous_scale="RdYlGn", range_color=[-1, 1],
                    title="Most Mentioned Tickers"
                )
                fig_tm.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f1f5f9"), height=400, margin=dict(t=40, b=10, l=10, r=10))
                st.plotly_chart(fig_tm, use_container_width=True)
            else:
                st.info("No ticker data available for heatmap.")
    else:
        st.info("Run Batch Analysis or Live Market to populate the heatmap.")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center>StockSentimentIQ | Premium AI Market Analytics<br>"
    "<small> For educational purposes only. Not financial advice.</small></center>",
    unsafe_allow_html=True,
)