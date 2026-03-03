# 📰 News Sentiment vs Stock Price Analyzer

A full-stack NLP web application that scrapes financial news headlines, scores them using sentiment analysis, and correlates the results with live NSE/BSE stock price data to find predictive patterns.s

**🔴 Live Demo →** [sentiment-analyzer.streamlit.app](https://sentiment-analyzer.streamlit.app)

---

## 🧠 What Problem Does It Solve?

Investors and analysts read hundreds of news headlines daily about stocks — forming gut feelings about market direction. This process is **slow, biased, and unscalable**.

This app **automates and quantifies** that process:
- Scrapes real financial news headlines via NewsAPI
- Scores each headline from **-1 (very negative)** to **+1 (very positive)** using NLP
- Plots sentiment against actual NSE/BSE stock price movement
- Calculates **same-day and 1-day lag correlation** — answering: *does today's news predict tomorrow's price?*

---

## 📊 Features

- 🔍 **40+ preset NSE/BSE stocks** or type any custom ticker
- 📡 **Live news scraping** via NewsAPI (or demo mode without a key)
- 🧠 **VADER NLP pipeline** — scores every headline individually
- 📈 **Dual-axis chart** — sentiment bars overlaid with stock price (₹)
- 🔵 **Scatter plot** with trend line — sentiment score vs daily return
- 🥧 **Sentiment breakdown** — pie chart + daily volume bar chart
- 📋 **Filterable headline table** — filter by sentiment, sort by score or date
- 🔗 **Correlation metrics** — same-day and predictive 1-day lag scores
- 🎮 **Demo mode** — fully functional without any API key

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data Collection | `requests`, `yfinance`, NewsAPI |
| NLP Pipeline | `vaderSentiment`, custom financial lexicon |
| Data Processing | `pandas`, `numpy` |
| Visualization | `plotly` |
| Web App | `streamlit` |
| Deployment | Streamlit Community Cloud |

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/sentiment-analyzer.git
cd sentiment-analyzer

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## 🔑 NewsAPI Key 

The app runs with a key. For live headlines:

1. Get a free key at [newsapi.org](https://newsapi.org) (takes 2 mins)
2. Paste it in the sidebar when the app runs

---

## 📁 Project Structure

```
sentiment-analyzer/
├── app.py                 # Streamlit frontend — all UI and charts
├── data_fetcher.py        # Data layer — NewsAPI + yfinance fetching
├── sentiment_engine.py    # NLP layer — VADER scoring + correlation
├── requirements.txt       # Python dependencies
└── README.md
```

Each file has a single responsibility — clean separation of concerns.

---

## 💡 How It Works

```
User selects stock + date range
        ↓
NewsAPI fetches headlines for that company
        ↓
VADER NLP scores each headline (-1 to +1)
        ↓
Scores aggregated to daily averages
        ↓
yfinance fetches actual NSE/BSE price data
        ↓
Pandas merges both on date
        ↓
Correlation calculated (same-day + 1-day lag)
        ↓
Plotly renders interactive dashboard
```

---

## 📈 Sample Output

| Metric | Example Value |
|---|---|
| Headlines Analyzed | 90 |
| Overall Sentiment | Positive (0.41) |
| Positive News | 96.7% |
| Same-Day Correlation | 0.312 |
| 1-Day Lag Correlation | 0.284 |
| Current Price | ₹2,450.30 |

---

## 🌐 Supported Stocks

Any stock listed on **NSE or BSE**:
- NSE stocks → append `.NS` → e.g. `ZOMATO.NS`, `IRCTC.NS`
- BSE stocks → append `.BO` → e.g. `RELIANCE.BO`

40+ presets included (Reliance, TCS, HDFC Bank, Infosys, Zomato, Paytm, and more)

---

## 📌 Context

Built as an extension of data analysis work done during a finance internship. The project automates the manual process of reading financial news and estimating market sentiment — replacing gut-feel with an NLP-driven, data-backed correlation pipeline.

---
