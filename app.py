import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_SCRIPT_DIR, ".env"))
except ImportError:
    pass

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Sentiment vs Price Analyzer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    from data_fetcher import fetch_news, fetch_stock_data
    from sentiment_engine import analyze_sentiment, aggregate_daily_sentiment, merge_and_correlate
except Exception as e:
    st.error("Could not load app modules. Fix the error below and restart.")
    st.code(str(e))
    st.stop()

st.markdown("""
<style>
    /* ── Main background ── */
    .stApp { background-color: #020202 !important; }
    section[data-testid="stMain"] { background-color: #020202 !important; }
    .main .block-container { background-color: #020202 !important; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] { background-color: #0f0f0f !important; border-right: 1px solid #00c48c44; }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span { color: #e0e0e0 !important; }

    /* ── General text ── */
    p, span, li { color: #e0e0e0; }

    /* ── Headings ── */
    h1, h2, h3, h4 { color: #00c48c !important; }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: #0a1a13 !important;
        border: 1px solid #00c48c33 !important;
        border-radius: 10px !important;
        padding: 12px !important;
    }
    [data-testid="stMetricValue"] { color: #00c48c !important; font-size: 1.4rem !important; }
    [data-testid="stMetricLabel"] { color: #aaaaaa !important; }
    [data-testid="stMetricDelta"] { color: #00c48c !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #003d2b, #005a3e) !important;
        color: #00c48c !important;
        border: 1px solid #00c48c !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #005a3e, #007a52) !important; }

    /* ── Input fields ── */
    .stTextInput input, .stSelectbox > div > div {
        background-color: #111111 !important;
        color: #e0e0e0 !important;
        border: 1px solid #00c48c44 !important;
    }

    /* ── Radio ── */
    .stRadio label { color: #e0e0e0 !important; }

    /* ── Slider ── */
    [data-testid="stSlider"] div[role="slider"] { background: #00c48c !important; }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] { border: 1px solid #00c48c22 !important; border-radius: 8px !important; }

    /* ── Dividers ── */
    hr { border-color: #00c48c22 !important; }

    /* ── Caption ── */
    .stCaption, caption { color: #666666 !important; }

    /* ── Header ── */
    header[data-testid="stHeader"] { background-color: #020202 !important; }

    /* ── Info/warning/success ── */
    [data-testid="stAlert"] { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ── Indian Stocks — NSE & BSE ──
NSE_BSE_STOCKS = {
    # Nifty 50 / Large Cap
    "Reliance Industries":       "RELIANCE.NS",
    "TCS":                       "TCS.NS",
    "HDFC Bank":                 "HDFCBANK.NS",
    "Infosys":                   "INFY.NS",
    "ICICI Bank":                "ICICIBANK.NS",
    "SBI":                       "SBIN.NS",
    "Wipro":                     "WIPRO.NS",
    "Bajaj Finance":             "BAJFINANCE.NS",
    "Tata Motors":               "TATAMOTORS.NS",
    "Tata Steel":                "TATASTEEL.NS",
    "HCL Technologies":          "HCLTECH.NS",
    "Asian Paints":              "ASIANPAINT.NS",
    "Maruti Suzuki":             "MARUTI.NS",
    "Sun Pharma":                "SUNPHARMA.NS",
    "Adani Ports":               "ADANIPORTS.NS",
    "Adani Enterprises":         "ADANIENT.NS",
    "Larsen & Toubro":           "LT.NS",
    "Axis Bank":                 "AXISBANK.NS",
    "Kotak Mahindra Bank":       "KOTAKBANK.NS",
    "ITC":                       "ITC.NS",
    "Power Grid":                "POWERGRID.NS",
    "NTPC":                      "NTPC.NS",
    "ONGC":                      "ONGC.NS",
    "Bharti Airtel":             "BHARTIARTL.NS",
    "Tech Mahindra":             "TECHM.NS",
    "Titan Company":             "TITAN.NS",
    "UltraTech Cement":          "ULTRACEMCO.NS",
    "Nestle India":              "NESTLEIND.NS",
    "JSW Steel":                 "JSWSTEEL.NS",
    "Hindalco":                  "HINDALCO.NS",
    # Mid Cap
    "Zomato":                    "ZOMATO.NS",
    "Paytm":                     "PAYTM.NS",
    "Nykaa":                     "NYKAA.NS",
    "Delhivery":                 "DELHIVERY.NS",
    "Vedanta":                   "VEDL.NS",
    "Tata Power":                "TATAPOWER.NS",
    "Indigo (IndiGo)":           "INDIGO.NS",
    "Trent":                     "TRENT.NS",
    "Dixon Technologies":        "DIXON.NS",
    "Havells India":             "HAVELLS.NS",
}

# ── SIDEBAR ──
with st.sidebar:
    st.title("📰 Sentilytics")
    st.markdown("---")
    st.subheader("🔍 Stock Settings")

    search_mode = st.radio("How to select stock?", ["Choose from list", "Type any ticker"], horizontal=True)

    if search_mode == "Choose from list":
        selected_name = st.selectbox("Select NSE/BSE Stock", list(NSE_BSE_STOCKS.keys()))
        ticker        = NSE_BSE_STOCKS[selected_name]
        company_name  = selected_name
        st.caption(f"Ticker: `{ticker}`")
    else:
        st.markdown("**Enter NSE or BSE ticker:**")
        custom_ticker = st.text_input("Ticker symbol", placeholder="e.g. IRCTC.NS or ZOMATO.NS").upper().strip()
        company_name  = st.text_input("Company name (for news search)", placeholder="e.g. IRCTC or Zomato").strip()
        ticker        = custom_ticker if custom_ticker else "RELIANCE.NS"
        selected_name = company_name if company_name else ticker
        st.caption("💡 NSE stocks → add `.NS`  |  BSE stocks → add `.BO`")
        if not custom_ticker:
            st.warning("Enter a ticker above.")

    days = st.slider("Analysis Period (days)", min_value=7, max_value=30, value=20)

    st.markdown("---")
    st.subheader("🔑 NewsAPI Key")
    api_key = st.text_input(
        "Paste your API key",
        value=os.environ.get("NEWS_API_KEY", ""),
        type="password",
        help="Free at newsapi.org. Leave blank for demo mode.",
        placeholder="Leave blank for demo mode"
    )
    if not api_key:
        st.info("🎮 Running in **demo mode** with synthetic data.")

    st.markdown("---")
    analyze_btn = st.button("Analyze", use_container_width=True, type="primary")
    st.markdown("---")
    st.caption("Built with Python · Pandas · VADER NLP · Plotly · Streamlit")
    st.caption("by Birjyot Singh Sahiwal")


# ── MAIN ──
st.title("📰 News Sentiment vs Stock Price Analyzer")
st.markdown("*Does financial news sentiment predict NSE/BSE stock price movement?*")
st.markdown("---")

if not analyze_btn:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🗞️ Step 1: Scrape News\nFetches real headlines for your stock using NewsAPI — or runs on demo data.")
    with col2:
        st.markdown("### 🧠 Step 2: NLP Scoring\nVADER sentiment model scores each headline from **-1 (negative)** to **+1 (positive)**.")
    with col3:
        st.markdown("### 📊 Step 3: Correlate\nCompares daily sentiment with actual NSE/BSE price movement.")
    st.info("👈 Select a stock and hit **Analyze** to get started.")
    st.stop()


# ── PIPELINE ──
with st.spinner(f"📡 Fetching news for **{company_name}**..."):
    news_df = fetch_news(company_name, api_key or "", days=days)

with st.spinner(f"📈 Loading NSE price data for **{ticker}**..."):
    stock_df = fetch_stock_data(ticker, days=days)

with st.spinner("🧠 Running sentiment analysis..."):
    scored_df  = analyze_sentiment(news_df)
    daily_sent = aggregate_daily_sentiment(scored_df)
    merged, same_corr, lag_corr = merge_and_correlate(daily_sent, stock_df)

if not api_key:
    st.warning("⚠️ **Demo Mode:** Showing synthetic headlines. Add a NewsAPI key for live data.")


# ── METRICS ──
st.subheader("📊 Key Metrics")

total_headlines = len(scored_df)
pos_pct    = (scored_df["sentiment_label"] == "🟢 Positive").mean() * 100
neg_pct    = (scored_df["sentiment_label"] == "🔴 Negative").mean() * 100
avg_sent   = scored_df["compound_score"].mean()
curr_price = stock_df["close"].iloc[-1] if len(stock_df) > 0 else 0
price_change = ((stock_df["close"].iloc[-1] / stock_df["close"].iloc[0]) - 1) * 100 if len(stock_df) > 1 else 0

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("📰 Headlines Analyzed", f"{total_headlines}")
with col2:
    sentiment_label = "Positive 🟢" if avg_sent > 0.05 else ("Negative 🔴" if avg_sent < -0.05 else "Neutral 🟡")
    st.metric("Overall Sentiment", sentiment_label, f"Score: {avg_sent:.3f}")
with col3:
    st.metric("🟢 Positive News", f"{pos_pct:.1f}%")
with col4:
    st.metric("🔴 Negative News", f"{neg_pct:.1f}%")
with col5:
    st.metric(f"{selected_name} Price", f"₹{curr_price:,.2f}", f"{price_change:+.2f}% over {days}d")

st.markdown("---")


# ── CORRELATION ──
st.subheader("🔗 Sentiment–Price Correlation")

corr_col1, corr_col2 = st.columns(2)
with corr_col1:
    strength  = "Strong" if abs(same_corr) > 0.5 else ("Moderate" if abs(same_corr) > 0.2 else "Weak")
    direction = "Positive" if same_corr > 0 else "Negative"
    st.metric("Same-Day Correlation", f"{same_corr:.3f}", f"{strength} {direction} relationship")
    st.caption("How much today's sentiment matches today's price move")
with corr_col2:
    lag_strength  = "Strong" if abs(lag_corr) > 0.5 else ("Moderate" if abs(lag_corr) > 0.2 else "Weak")
    lag_direction = "Positive" if lag_corr > 0 else "Negative"
    st.metric("1-Day Lag Correlation (Predictive)", f"{lag_corr:.3f}", f"{lag_strength} {lag_direction} predictive signal")
    st.caption("How much today's sentiment predicts tomorrow's price move")

if abs(lag_corr) > 0.4:
    st.success(f"📌 **Finding:** Strong predictive signal ({lag_corr:.3f}) — news sentiment correlates meaningfully with next-day price for {selected_name}.")
elif abs(lag_corr) > 0.2:
    st.info(f"📌 **Finding:** Moderate signal ({lag_corr:.3f}) — some predictive relationship exists.")
else:
    st.warning(f"📌 **Finding:** Weak signal ({lag_corr:.3f}) — news sentiment alone isn't a strong predictor here. Markets are complex!")

st.markdown("---")


# ── MAIN CHART ──
st.subheader("📈 Sentiment Score vs Stock Price (₹) Over Time")

if not merged.empty:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Daily Sentiment Score (Averaged from Headlines)",
            f"{selected_name} — Closing Price (₹)"
        ),
        row_heights=[0.45, 0.55]
    )

    colors = ["#00c48c" if v >= 0 else "#ff4b4b" for v in daily_sent["avg_sentiment"]]
    fig.add_trace(go.Bar(
        x=daily_sent["date"], y=daily_sent["avg_sentiment"],
        marker_color=colors, name="Daily Sentiment", opacity=0.7,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=daily_sent["date"], y=daily_sent["rolling_sentiment"],
        mode="lines", line=dict(color="#ffa500", width=2, dash="dot"),
        name="3-Day Rolling Avg",
    ), row=1, col=1)

    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3, row=1, col=1)

    fig.add_trace(go.Scatter(
        x=stock_df["date"], y=stock_df["close"],
        mode="lines+markers",
        line=dict(color="#00c48c", width=2),
        marker=dict(size=4),
        name=f"{selected_name} (₹)",
        fill="tozeroy",
        fillcolor="rgba(0, 196, 140, 0.08)",
    ), row=2, col=1)

    fig.update_layout(
        height=520,
        template="plotly_dark",
        plot_bgcolor="#020202",
        paper_bgcolor="#020202",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Sentiment Score", row=1, col=1, range=[-1, 1])
    fig.update_yaxes(title_text="Price (₹)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Not enough overlapping dates between news and price data to plot.")

st.markdown("---")


# ── SCATTER ──
st.subheader("🔵 Scatter: Daily Sentiment Score vs Price Return (%)")
st.caption("Each dot = one trading day. Trend line shows the directional relationship.")

if not merged.empty and len(merged) >= 3:
    x_vals = merged["avg_sentiment"].values
    y_vals = merged["pct_change"].values
    m, b   = np.polyfit(x_vals, y_vals, 1)
    tx     = np.linspace(x_vals.min(), x_vals.max(), 100)
    ty     = m * tx + b

    fig2 = px.scatter(
        merged, x="avg_sentiment", y="pct_change",
        color="avg_sentiment",
        color_continuous_scale=["#ff4b4b", "#888888", "#00c48c"],
        range_color=[-0.5, 0.5],
        labels={"avg_sentiment": "Avg Sentiment Score", "pct_change": "Price Change (%)"},
        hover_data={"date": True, "avg_sentiment": ":.3f", "pct_change": ":.2f"},
        template="plotly_dark",
    )
    fig2.add_trace(go.Scatter(
        x=tx, y=ty, mode="lines",
        line=dict(color="#ffa500", width=2, dash="dot"),
        name="Trend", showlegend=False,
    ))
    fig2.update_layout(
        height=380, plot_bgcolor="#020202", paper_bgcolor="#020202",
        coloraxis_showscale=False, margin=dict(l=40, r=20, t=20, b=40),
    )
    fig2.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.3)
    fig2.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")


# ── PIE + BAR ──
st.subheader("🥧 News Sentiment Breakdown")

pie_col, bar_col = st.columns([1, 2])
with pie_col:
    fig3 = go.Figure(data=[go.Pie(
        labels=["Positive", "Negative", "Neutral"],
        values=[
            (scored_df["sentiment_label"] == "🟢 Positive").sum(),
            (scored_df["sentiment_label"] == "🔴 Negative").sum(),
            (scored_df["sentiment_label"] == "🟡 Neutral").sum(),
        ],
        hole=0.5,
        marker_colors=["#00c48c", "#ff4b4b", "#ffa500"],
    )])
    fig3.update_layout(height=280, template="plotly_dark", paper_bgcolor="#020202",
                       showlegend=True, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)

with bar_col:
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=daily_sent["date"], y=daily_sent["positive_count"],
                          name="Positive", marker_color="#00c48c", opacity=0.8))
    fig4.add_trace(go.Bar(x=daily_sent["date"], y=daily_sent["negative_count"],
                          name="Negative", marker_color="#ff4b4b", opacity=0.8))
    fig4.update_layout(
        barmode="stack", height=280, template="plotly_dark",
        paper_bgcolor="#020202", plot_bgcolor="#020202",
        title="Daily Headline Volume by Sentiment",
        margin=dict(l=20, r=10, t=40, b=30),
        legend=dict(orientation="h"), yaxis_title="# Headlines",
    )
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")


# ── HEADLINE TABLE ──
st.subheader("📋 All Headlines with Sentiment Scores")

col_f1, col_f2 = st.columns([1, 2])
with col_f1:
    filter_sentiment = st.selectbox("Filter by sentiment", ["All", "🟢 Positive", "🔴 Negative", "🟡 Neutral"])
with col_f2:
    sort_by = st.selectbox("Sort by", ["Date (newest first)", "Highest Score", "Lowest Score"])

display_df = scored_df.copy()
if filter_sentiment != "All":
    display_df = display_df[display_df["sentiment_label"] == filter_sentiment]
if sort_by == "Date (newest first)":
    display_df = display_df.sort_values("date", ascending=False)
elif sort_by == "Highest Score":
    display_df = display_df.sort_values("compound_score", ascending=False)
else:
    display_df = display_df.sort_values("compound_score", ascending=True)

display_df["date"] = display_df["date"].dt.strftime("%b %d, %Y")
st.dataframe(
    display_df[["date", "headline", "sentiment_label", "compound_score", "source"]].rename(columns={
        "date": "Date", "headline": "Headline", "sentiment_label": "Sentiment",
        "compound_score": "Score", "source": "Source"
    }).reset_index(drop=True),
    use_container_width=True, height=350,
)
st.caption(f"Showing {len(display_df)} of {len(scored_df)} headlines")