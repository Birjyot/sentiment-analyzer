# StockSentimentIQ — AI-Powered Stock Market Sentiment Analyzer

StockSentimentIQ is a production-grade Streamlit application that scores financial news headlines using **VADER NLP** (with finance-tuned lexicon boosters), visualizes **bullish / bearish / neutral** market sentiment, and generates **Wall Street-style insights** via **Groq Llama 3.3 70B**. Upload headline CSVs, analyze live news from Reuters/Bloomberg/CNBC/MarketWatch, or chat with your dataset using an AI market analyst.

---

## Features

| Tab | Description |
|-----|-------------|
| 📰 **Analyze Headline** | Score a single market headline with VADER — signal, strength, gauge chart |
| 📊 **Batch Analysis** | Upload CSV (`text`, `ticker`, `source`) — KPIs, charts, filterable table, CSV export |
| 🤖 **AI Market Insights** | Groq generates 4 investment insights, executive summary, market call, risk level |
| 💬 **Chat with Data** | Ask questions about your uploaded headline dataset |
| 🔴 **Live News** | Fetch live headlines via NewsAPI for major tickers |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Web UI | Streamlit |
| NLP | VADER (`vaderSentiment`) + finance lexicon boosters |
| AI | Groq API — `llama-3.3-70b-versatile` |
| Charts | Plotly |
| Data | pandas |
| Live news | NewsAPI (`requests`) |
| Config | python-dotenv |

---

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Birjyot/sentiment-analyzer.git
   cd sentiment-analyzer
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys**
   ```bash
   copy .env.example .env       # Windows
   cp .env.example .env         # Mac/Linux
   ```
   Edit `.env`:
   - `GROQ_API_KEY` — **required** for AI Insights & Chat ([console.groq.com](https://console.groq.com))
   - `NEWS_API_KEY` — optional for Live News tab ([newsapi.org](https://newsapi.org))

5. **Run the app**
   ```bash
   streamlit run app.py
   ```
   Open **http://localhost:8501**

---

## CSV Format

```csv
text,ticker,source
"Apple beats earnings expectations...",AAPL,Reuters
"Fed signals more rate hikes...",SPY,Bloomberg
```

Supported text column names: `text`, `headline`, `title`, `news`, `content`, `article`, `description`, `summary`, `message`.

---

## Project Structure

```
sentiment-analyzer/
├── app.py                 # Main Streamlit application
├── services/
│   ├── sentiment.py       # VADER + finance boosters
│   ├── ai_service.py      # Groq market insights & chat
│   ├── data_service.py    # CSV parsing
│   └── news_service.py    # NewsAPI live headlines
├── sample_data.csv        # 22 sample market headlines
├── requirements.txt
├── .env.example
└── README.md
```

---

## Disclaimer

**For educational purposes only. Not financial advice.** Sentiment scores and AI insights are research tools — not buy/sell recommendations. Always do your own due diligence before making investment decisions.
