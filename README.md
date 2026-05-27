# 📈 StockSentimentIQ — Premium Edition

A state-of-the-art AI-powered market intelligence dashboard that analyzes financial headlines, live market data, and sentiment signals in real time. 

Built with Streamlit, Plotly, Groq LLaMA 3.3 70B, and VADER NLP, featuring a custom glassmorphism dark theme UI and interactive canvas backgrounds.

![App Preview](https://via.placeholder.com/1000x500.png?text=StockSentimentIQ+Premium+Dashboard)

## ✨ Features

- **Live Market Feed:** Real-time stock prices, candlestick charts with 20-day SMA, and TradingView embeds for major exchanges (US, NSE, BSE).
- **Market Overview:** Live Fear/Greed gauge, Top Movers visualization, and Sector Sentiment tracking.
- **Batch Headline Analysis:** Upload CSVs of headlines or run the sample dataset to view pie charts, distribution histograms, and sentiment timelines.
- **AI Market Insights:** Generate high-level market calls, risk assessments, and executive summaries using **LLaMA 3.3 70B** running on Groq.
- **Chat with Data:** Talk directly to your dataset and ask complex questions using AI.
- **Premium Aesthetics:** Custom CSS overrides for frosted glass cards, gradient text, dynamic dotfield backgrounds, and seamless responsive layouts.

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/sentiment-analyzer.git
cd sentiment-analyzer
```

### 2. Install dependencies
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
Copy the example environment file and add your keys:
```bash
cp .env.example .env
```
Open `.env` and add:
- **Groq API Key:** (Required) For LLaMA 3.3 AI insights.
- **Finnhub API Key:** (Highly Recommended) For real-time US stock quotes and market news.
- **Alpha Vantage Key:** (Recommended) For OHLC candlestick data and technical indicators.
- **NewsAPI Key:** (Optional) For alternative news fallback.

### 4. Run the app
```bash
streamlit run app.py
```

## 🧠 Architecture Stack

- **Frontend:** Streamlit, Custom CSS (Glassmorphism), Vanilla JS (Interactive Background Canvas)
- **Data Visualization:** Plotly Express & Plotly Graph Objects
- **Sentiment Engine:** VADER Sentiment Analysis (NLTK) + Finance Domain Boosters
- **AI/LLM:** Groq API (LLaMA 3.3 70B)
- **Data Sources:** Finnhub, Alpha Vantage, Yahoo Finance (`yfinance`), NewsAPI

## 📝 License
MIT License. Created for educational and market research purposes. Not financial advice.
