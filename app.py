import os
import sys

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
import plotly.express as px
import plotly.graph_objects as go
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
from ui.dotfield_bg import render_dotfield_background
import streamlit.components.v1 as components

# ── PAGE CONFIG — must be first Streamlit call ──────────────────────────────
st.set_page_config(
    page_title="StockSentimentIQ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── DOTFIELD BACKGROUND — call immediately after set_page_config ────────────
render_dotfield_background()

# ── APP-LEVEL CSS ────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Sentiment cards ── */
.bullish-card {
    border-left: 4px solid #16a34a;
    background: #14532d44;
    padding: 16px;
    border-radius: 8px;
    margin-bottom: 8px;
    color: #f1f5f9;
}
.bearish-card {
    border-left: 4px solid #dc2626;
    background: #7f1d1d44;
    padding: 16px;
    border-radius: 8px;
    margin-bottom: 8px;
    color: #f1f5f9;
}
.neutral-card {
    border-left: 4px solid #6b7280;
    background: #1e293b99;
    padding: 16px;
    border-radius: 8px;
    margin-bottom: 8px;
    color: #f1f5f9;
}
.opportunity-card {
    border-left: 4px solid #6366f1;
    background: #312e8144;
    padding: 16px;
    border-radius: 8px;
    margin-bottom: 8px;
    color: #f1f5f9;
}

/* ── Metric value size ── */
div[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}

/* ── Tab styling ── */
[data-testid="stTabs"] button[data-baseweb="tab"] {
    color: #94a3b8 !important;
    font-weight: 500;
}
[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
    color: #6366f1 !important;
    border-bottom: 2px solid #6366f1 !important;
}

/* ── Input fields ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: #1e293bcc !important;
    color: #f1f5f9 !important;
    border-color: #334155 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    background: #1e293bcc !important;
}

/* ── Success / error / warning / info boxes ── */
[data-testid="stAlert"] {
    background: #1e293bcc !important;
}

/* ── Header ── */
.app-header-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #f1f5f9;
    letter-spacing: -0.5px;
    line-height: 1.1;
}
.app-header-sub {
    font-size: 1rem;
    color: #94a3b8;
    margin-top: 4px;
}
.badge {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    font-weight: 600;
    color: #a78bfa;
    margin-bottom: 6px;
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
    if signal == "Bullish":
        return "#16a34a"
    if signal == "Bearish":
        return "#dc2626"
    return "#6b7280"


def signal_emoji(signal: str) -> str:
    if signal == "Bullish":
        return "🟢"
    if signal == "Bearish":
        return "🔴"
    return "⚪"


def get_market_signal_from_compound(compound: float) -> str:
    if compound >= 0.05:
        return "Bullish"
    if compound <= -0.05:
        return "Bearish"
    return "Neutral"


def enrich_results(df: pd.DataFrame, results: list) -> list:
    df = df.reset_index(drop=True)
    for i in range(min(len(df), len(results))):
        results[i]["ticker"] = df.loc[i, "ticker"]
        results[i]["source"] = df.loc[i, "source"]
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


# ── KPI CARDS ────────────────────────────────────────────────────────────────
def render_kpi_cards(summary: dict):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Headlines", summary["total"])
    with c2:
        st.metric(
            "Bullish %",
            f"{summary['bullish_pct']}%",
            delta=f"{summary['bullish_count']} headlines",
            delta_color="normal",
        )
    with c3:
        st.metric(
            "Bearish %",
            f"{summary['bearish_pct']}%",
            delta=f"{summary['bearish_count']} headlines",
            delta_color="inverse",
        )
    with c4:
        sig = summary.get("overall_signal", "MIXED ⚪")
        st.metric("Overall Signal", sig)
    with c5:
        st.metric("Market Health Score", market_health_display(summary))


# ── CHARTS ───────────────────────────────────────────────────────────────────
def render_sentiment_charts(summary: dict, results: list, df: pd.DataFrame):
    c1, c2 = st.columns(2)

    with c1:
        pie_df = pd.DataFrame({
            "Signal": ["Bullish", "Bearish", "Neutral"],
            "Count": [
                summary["bullish_count"],
                summary["bearish_count"],
                summary["neutral_count"],
            ],
        })
        fig_pie = px.pie(
            pie_df,
            names="Signal",
            values="Count",
            color="Signal",
            color_discrete_map={
                "Bullish": "#16a34a",
                "Bearish": "#dc2626",
                "Neutral": "#9ca3af",
            },
            hole=0.45,
            title="Headline Sentiment Mix",
        )
        fig_pie.update_layout(
            margin=dict(t=40, b=20, l=10, r=10),
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f1f5f9"),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        compounds = [r["compound"] for r in results]
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=[c for c in compounds if c >= 0],
            name="Bullish zone",
            marker_color="#2563eb",
            opacity=0.75,
        ))
        fig_hist.add_trace(go.Histogram(
            x=[c for c in compounds if c < 0],
            name="Bearish zone",
            marker_color="#dc2626",
            opacity=0.75,
        ))
        fig_hist.update_layout(
            barmode="overlay",
            title="Sentiment Score Distribution",
            xaxis_title="Sentiment Score",
            yaxis_title="Headlines",
            margin=dict(t=40, b=20, l=10, r=10),
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1e293b",
            font=dict(color="#f1f5f9"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)


def render_ticker_sentiment_chart(results: list):
    ticker_summary = st.session_state.ticker_summary or analyze_by_ticker(results)
    if not ticker_summary:
        st.info("No ticker data available for chart.")
        return

    tdf = pd.DataFrame([
        {"Ticker": v["ticker"], "Avg Sentiment Score": v["avg_compound"]}
        for v in ticker_summary.values()
    ])
    tdf = tdf.sort_values("Avg Sentiment Score", ascending=False)
    colors = [
        "#16a34a" if x > 0 else ("#dc2626" if x < 0 else "#9ca3af")
        for x in tdf["Avg Sentiment Score"]
    ]

    fig_bar = go.Figure(go.Bar(
        x=tdf["Avg Sentiment Score"],
        y=tdf["Ticker"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}" for v in tdf["Avg Sentiment Score"]],
        textposition="outside",
    ))
    fig_bar.update_layout(
        title="Sentiment by Stock Ticker",
        margin=dict(t=50, b=30, l=10, r=30),
        height=max(320, len(tdf) * 36),
        xaxis_title="Avg Sentiment Score",
        yaxis_title="Ticker",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1e293b",
        font=dict(color="#f1f5f9"),
    )
    fig_bar.add_vline(x=0, line_dash="dash", line_color="#64748b")
    st.plotly_chart(fig_bar, use_container_width=True)


def render_gauge(compound: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=compound,
        number={"suffix": "", "font": {"size": 28, "color": "#f1f5f9"}},
        title={"text": "Sentiment Gauge", "font": {"size": 20, "color": "#f1f5f9"}},
        gauge={
            "axis": {"range": [-1, 1], "tickcolor": "#94a3b8"},
            "bar": {"color": signal_color(get_market_signal_from_compound(compound))},
            "bgcolor": "#1e293b",
            "bordercolor": "#334155",
            "steps": [
                {"range": [-1, -0.05], "color": "#7f1d1d"},
                {"range": [-0.05, 0.05], "color": "#854d0e"},
                {"range": [0.05, 1], "color": "#14532d"},
            ],
            "threshold": {
                "line": {"color": "#f1f5f9", "width": 4},
                "thickness": 0.85,
                "value": compound,
            },
        },
    ))
    fig.update_layout(
        margin=dict(t=40, b=20, l=20, r=20),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def insight_container(itype: str, title: str, body: str):
    content = f"**{title}**\n\n{body}"
    if itype == "bullish":
        st.success(content)
    elif itype == "bearish":
        st.error(content)
    elif itype == "warning":
        st.warning(content)
    else:
        st.info(content)


# ── HEADER ───────────────────────────────────────────────────────────────────
h_left, h_right = st.columns([3, 1])
with h_left:
    st.markdown(
        '<p class="app-header-title">📈 StockSentimentIQ</p>'
        '<p class="app-header-sub">Real-time Stock Market Sentiment Analysis powered by AI</p>',
        unsafe_allow_html=True,
    )
with h_right:
    st.markdown("")
    st.markdown('<span class="badge">⚡ Groq Llama 3.3 70B</span>', unsafe_allow_html=True)
    st.markdown('<span class="badge">🧠 VADER NLP</span>', unsafe_allow_html=True)

st.divider()

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📰 Analyze Headline",
    "📊 Batch Analysis",
    "🤖 AI Market Insights",
    "💬 Chat with Data",
    "🔴 Live Market",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYZE HEADLINE
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Analyze a single market headline")
    headline = st.text_area(
        "Financial news headline",
        height=120,
        placeholder=(
            "e.g. NVIDIA reports blowout quarter driven by AI chip demand "
            "beating all analyst expectations"
        ),
        key="single_headline_input",
    )
    ticker_opt = st.text_input("Ticker (optional)", placeholder="NVDA", key="single_ticker")

    if st.button("Analyze Headline", use_container_width=True, type="primary"):
        if not (headline or "").strip():
            st.warning("Enter a market headline to analyze.")
        else:
            st.session_state.single_result = analyze_single(headline)

    examples = [
        ("AAPL beats earnings",
         "Apple beats earnings expectations with record iPhone sales driving massive revenue growth"),
        ("Fed rate fears",
         "Federal Reserve signals more rate hikes ahead sending markets into freefall today"),
        ("NVDA blowout",
         "NVIDIA reports blowout quarter driven by insane AI chip demand from hyperscale data centers"),
        ("TSLA miss",
         "Tesla deliveries disappoint analysts as production issues continue to plague the company"),
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
        with m1:
            st.metric("Signal", f"{signal_emoji(r['signal'])} {r['signal']}")
        with m2:
            st.metric("Strength", r["signal_strength"])
        with m3:
            st.metric("Sentiment Score", r["sentiment_score"])
        with m4:
            st.metric("Confidence", f"{r['confidence']}%")

        if ticker_opt:
            st.caption(f"Ticker context: **{ticker_opt.upper()}**")

        st.progress(float(r["positive"]),
                    text=f"Bullish Score: {float(r['positive']):.0%}")
        st.progress(float(r["negative"]),
                    text=f"Bearish Score: {float(r['negative']):.0%}")
        st.progress(float(r["neutral"]),
                    text=f"Neutral Score:  {float(r['neutral']):.0%}")

        render_gauge(r["compound"])

        if r["signal"] == "Bullish":
            st.success(f"**Bullish** market tone — sentiment score {r['sentiment_score']}")
        elif r["signal"] == "Bearish":
            st.error(f"**Bearish** market tone — sentiment score {r['sentiment_score']}")
        else:
            st.warning(f"**Neutral** market tone — sentiment score {r['sentiment_score']}")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Batch analysis of market headlines")
    up_col, sample_col = st.columns([2, 1])
    with up_col:
        uploaded = st.file_uploader(
            "Upload headlines CSV (text, ticker, source)", type=["csv"]
        )
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
        render_ticker_sentiment_chart(results)
        render_sentiment_charts(summary, results, df)

        st.markdown("#### Filtered headline table")
        f1, f2, f3 = st.columns([1, 1, 2])
        with f1:
            sig_filter = st.selectbox("Signal", ["All", "Bullish", "Bearish", "Neutral"])
        with f2:
            tickers = sorted({r.get("ticker", "UNKNOWN") for r in results})
            ticker_filter = st.multiselect("Ticker", tickers, default=tickers)
        with f3:
            search = st.text_input("Search headlines", placeholder="Filter by keyword...")

        table_df = pd.DataFrame(results)
        if sig_filter != "All":
            table_df = table_df[table_df["signal"] == sig_filter]
        if ticker_filter:
            table_df = table_df[table_df["ticker"].isin(ticker_filter)]
        if search.strip():
            table_df = table_df[
                table_df["text"].str.contains(search.strip(), case=False, na=False)
            ]

        display = table_df[
            ["text", "ticker", "signal", "sentiment_score", "confidence", "source"]
        ].rename(columns={
            "text": "Headline",
            "ticker": "Ticker",
            "signal": "Signal",
            "sentiment_score": "Sentiment Score",
            "confidence": "Confidence %",
            "source": "Source",
        })
        st.dataframe(display, use_container_width=True, height=360)

        csv_bytes = display.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Results as CSV",
            data=csv_bytes,
            file_name="stock_sentiment_headlines.csv",
            mime="text/csv",
            use_container_width=True,
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
            with st.spinner("Generating Wall Street-style insights with Llama 3.3 70B..."):
                st.session_state.ai_insights = generate_insights(summary)

        insights = st.session_state.ai_insights
        if insights:
            risk = insights.get("risk_level", "MEDIUM").upper()
            if risk == "LOW":
                st.success(f"**Risk Level:** {risk}")
            elif risk == "HIGH":
                st.error(f"**Risk Level:** {risk}")
            else:
                st.warning(f"**Risk Level:** {risk}")

            st.info(f"**Executive summary:** {insights.get('executive_summary', '')}")

            market_call = insights.get("market_call", "")
            if market_call:
                st.markdown(
                    f'<div class="opportunity-card">'
                    f'<strong>Market call:</strong> {market_call}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            cards = insights.get("insights", [])[:4]
            r1c1, r1c2 = st.columns(2)
            r2c1, r2c2 = st.columns(2)
            for slot, item in zip([r1c1, r1c2, r2c1, r2c2], cards):
                with slot:
                    title = f"{item.get('icon','📌')} {item.get('title','Insight')}"
                    insight_container(
                        item.get("type", "opportunity"),
                        title,
                        item.get("body", ""),
                    )

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — CHAT WITH DATA
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Chat with your market headline dataset")
    if not st.session_state.batch_summary:
        st.info("Run **Batch Analysis** first to enable data chat.")
    else:
        summary = st.session_state.batch_summary

        # Handle starter-question button clicks
        if st.session_state.pending_chat_question:
            q = st.session_state.pending_chat_question
            st.session_state.pending_chat_question = None
            st.session_state.chat_history.append({"role": "user", "content": q})
            with st.spinner("Analyzing..."):
                answer = chat_with_data(
                    q, summary, st.session_state.chat_history[:-1]
                )
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )
            st.rerun()

        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Starter question buttons
        st.markdown("**Quick questions:**")
        starters = [
            "What is the overall market signal?",
            "Which stocks are most bearish?",
            "What percentage of headlines are bullish?",
            "What are the top risk factors?",
            "How many strong conviction signals exist?",
            "What is the average sentiment score?",
        ]
        s1, s2, s3 = st.columns(3)
        s4, s5, s6 = st.columns(3)
        for col, q in zip([s1, s2, s3, s4, s5, s6], starters):
            with col:
                if st.button(q, use_container_width=True, key=f"chat_starter_{q[:24]}"):
                    st.session_state.pending_chat_question = q
                    st.rerun()

        # Chat input
        if prompt := st.chat_input("Ask anything about your market headline data..."):
            st.session_state.chat_history.append(
                {"role": "user", "content": prompt}
            )
            with st.spinner("Analyzing..."):
                answer = chat_with_data(
                    prompt, summary, st.session_state.chat_history[:-1]
                )
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )
            st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — LIVE MARKET
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Live market — any stock (NSE, BSE, US & global)")
    st.caption(
        "Prices via **Yahoo Finance** (e.g. `RELIANCE.NS` for NSE, `RELIANCE.BO` for BSE). "
        "Chart via **TradingView** embed. News via **NewsAPI** when configured."
    )

    ex_col, sym_col = st.columns([1, 2])
    with ex_col:
        exchange = st.selectbox(
            "Stock exchange",
            list(EXCHANGE_OPTIONS.keys()),
            help="NSE/BSE append .NS / .BO automatically",
        )
    with sym_col:
        if "NSE" in exchange:
            picks = {name: sym for name, sym in NSE_POPULAR}
        elif "BSE" in exchange:
            picks = {name: sym for name, sym in BSE_POPULAR}
        elif exchange.startswith("US"):
            picks = {name: sym for name, sym in US_POPULAR}
        else:
            picks = {}

        if picks:
            pick_name = st.selectbox(
                "Quick pick (optional)",
                ["Type symbol below"] + list(picks.keys()),
            )
            if pick_name != "Type symbol below":
                st.session_state.live_symbol_prefill = picks[pick_name]

        symbol_input = st.text_input(
            "Symbol",
            value=st.session_state.get("live_symbol_prefill", ""),
            placeholder="e.g. RELIANCE, TCS, AAPL, TSLA",
            help="NSE: RELIANCE → RELIANCE.NS | BSE: RELIANCE → RELIANCE.BO",
            key="live_symbol_input",
        )

    news_extra = st.text_input(
        "News search term (optional)",
        placeholder="Company name for better headlines, e.g. Reliance Industries",
    )

    if st.button(
        "Fetch Live Price & Headlines", use_container_width=True, type="primary"
    ):
        try:
            meta = normalize_ticker(symbol_input, exchange)
            st.session_state.live_ticker_meta = meta

            with st.spinner(f"Loading price data for **{meta['yahoo_symbol']}**..."):
                quote = fetch_live_quote(meta["yahoo_symbol"])
            if quote.get("ok") and quote.get("tradingview_symbol"):
                meta["tradingview_symbol"] = quote["tradingview_symbol"]
                st.session_state.live_ticker_meta = meta
            st.session_state.live_quote = quote

            news_q = (news_extra or meta["news_query"]).strip()
            if has_news_api_key():
                with st.spinner(f"Fetching headlines for **{news_q}**..."):
                    raw = fetch_stock_headlines(
                        news_q,
                        max_results=20,
                        exchange_label=exchange,
                    )
                if raw:
                    texts = [h["text"] for h in raw]
                    results, summary = analyze_batch(texts)
                    for i, h in enumerate(raw):
                        if i < len(results):
                            results[i]["source"]      = h.get("source", "Unknown")
                            results[i]["url"]          = h.get("url", "")
                            results[i]["publishedAt"]  = h.get("publishedAt", "")
                            results[i]["ticker"]       = meta["display_ticker"]
                    st.session_state.live_results  = results
                    st.session_state.live_summary  = summary
                else:
                    st.session_state.live_results  = None
                    st.session_state.live_summary  = None
                    st.warning(
                        "No headlines returned. "
                        "Try a different search term or check NewsAPI quota."
                    )
            else:
                st.session_state.live_results = None
                st.session_state.live_summary = None
                st.info(
                    "Add **NEWS_API_KEY** in `.env` for live headlines "
                    "([newsapi.org](https://newsapi.org))."
                )
        except ValueError as e:
            st.error(str(e))

    meta  = st.session_state.live_ticker_meta
    quote = st.session_state.live_quote

    if quote and quote.get("ok"):
        curr      = quote.get("currency", "")
        sym_label = meta["yahoo_symbol"] if meta else quote.get("yahoo_symbol", "")
        st.markdown(f"#### {quote.get('name', sym_label)} (`{sym_label}`)")

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.metric("Last price", f"{quote['price']:,.2f} {curr}")
        with p2:
            st.metric(
                "Change",
                f"{quote['change']:+,.2f}",
                f"{quote['change_pct']:+.2f}%",
            )
        with p3:
            st.metric("Volume", f"{quote['volume']:,}")
        with p4:
            st.metric("As of", quote.get("as_of", "—"))

        hist = quote.get("history")
        if hist is not None and not hist.empty:
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=hist["date"],
                y=hist["close"],
                mode="lines",
                fill="tozeroy",
                line=dict(color="#6366f1", width=2),
                name="Close",
            ))
            fig_price.update_layout(
                title="Price history (last ~1 month)",
                xaxis_title="Date",
                yaxis_title=f"Price ({curr})",
                height=360,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#1e293b",
                font=dict(color="#f1f5f9"),
                margin=dict(t=50, b=40, l=50, r=20),
            )
            st.plotly_chart(fig_price, use_container_width=True)

        if meta or quote.get("tradingview_symbol"):
            tv_sym = quote.get("tradingview_symbol") or meta["tradingview_symbol"]
            st.markdown("#### TradingView chart")
            st.caption(
                f"Chart symbol: **{tv_sym}** "
                f"(matches **{quote.get('name', sym_label)}**)"
            )
            components.html(
                tradingview_widget_html(tv_sym, height=450),
                height=460,
                scrolling=True,
            )

    elif quote and not quote.get("ok"):
        st.error(quote.get("error", "Could not load price data."))
        st.info(
            "**Tips:** NSE → use exchange **NSE** and symbol `RELIANCE` "
            "(becomes `RELIANCE.NS`). "
            "BSE → **BSE** and `RELIANCE.BO`. US → `AAPL`. "
            "Run: `pip install yfinance`"
        )

    if st.session_state.live_summary and st.session_state.live_results:
        st.markdown("#### Headline sentiment")
        render_kpi_cards(st.session_state.live_summary)
        for item in st.session_state.live_results:
            sig = item["signal"]
            css = (
                "bullish-card" if sig == "Bullish"
                else ("bearish-card" if sig == "Bearish" else "neutral-card")
            )
            url  = item.get("url", "")
            date = item.get("publishedAt", "")[:10]
            link = (
                f' <a href="{url}" target="_blank">Read article →</a>'
                if url else ""
            )
            st.markdown(
                f'<div class="{css}">'
                f'<strong>{signal_emoji(sig)} {sig}</strong> · '
                f'Score: <code>{item["sentiment_score"]}</code> · '
                f'{item.get("source","Unknown")} · {date}'
                f'<br><br>{item["text"]}{link}'
                f'</div>',
                unsafe_allow_html=True,
            )

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center>StockSentimentIQ | Groq Llama 3.3 70B + VADER NLP<br>"
    "<small> For educational purposes only. Not financial advice.</small></center>",
    unsafe_allow_html=True,
)