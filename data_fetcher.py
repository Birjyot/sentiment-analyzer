import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def fetch_news(query: str, api_key: str, days: int = 30) -> pd.DataFrame:
    if not api_key:
        return _fallback_news(query, days)

    end_date   = datetime.today()
    start_date = end_date - timedelta(days=days)

    url = "https://newsapi.org/v2/everything"
    params = {
        "q"        : query,
        "from"     : start_date.strftime("%Y-%m-%d"),
        "to"       : end_date.strftime("%Y-%m-%d"),
        "language" : "en",
        "sortBy"   : "publishedAt",
        "pageSize" : 100,
        "apiKey"   : api_key,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        articles = response.json().get("articles", [])
        records = []
        for a in articles:
            if a.get("title") and a["title"] != "[Removed]":
                records.append({
                    "date"    : pd.to_datetime(a["publishedAt"]).normalize(),
                    "headline": a["title"],
                    "source"  : a.get("source", {}).get("name", "Unknown"),
                })
        df = pd.DataFrame(records)
        if df.empty:
            return _fallback_news(query, days)
        return df.sort_values("date").reset_index(drop=True)
    except Exception as e:
        print(f"NewsAPI error: {e}")
        return _fallback_news(query, days)


def _fallback_news(query: str, days: int = 30) -> pd.DataFrame:
    np.random.seed(42)
    end_date = datetime.today()

    positive = [
        f"{query} reports record quarterly earnings, beats analyst expectations",
        f"{query} announces major expansion plan, shares surge",
        f"{query} secures landmark partnership deal with global firm",
        f"{query} raises full-year guidance amid strong demand",
        f"Analysts upgrade {query} to Buy citing strong fundamentals",
        f"{query} launches innovative product to strong market reception",
        f"{query} posts better-than-expected revenue growth this quarter",
        f"Institutional investors increase stake in {query}",
    ]
    negative = [
        f"{query} misses earnings estimates, shares fall in after-hours trading",
        f"{query} faces regulatory scrutiny over business practices",
        f"{query} cuts workforce by 8% amid restructuring efforts",
        f"Analysts downgrade {query} citing margin pressure concerns",
        f"{query} reports supply chain disruptions affecting production",
        f"{query} faces class action lawsuit from shareholders",
        f"{query} warns of slowing growth in key markets",
        f"CEO of {query} resigns amid controversy",
    ]
    neutral = [
        f"{query} to present at upcoming investor conference next week",
        f"{query} appoints new Chief Financial Officer",
        f"{query} completes scheduled share buyback program",
        f"{query} files annual report with regulatory authorities",
        f"What analysts are saying about {query} this quarter",
        f"{query} holds annual general meeting, no major changes announced",
    ]

    records = []
    for i in range(days):
        date = (end_date - timedelta(days=days - i)).date()
        n = np.random.randint(2, 5)
        for _ in range(n):
            r = np.random.random()
            pool = positive if r < 0.40 else (negative if r < 0.70 else neutral)
            records.append({
                "date"    : pd.Timestamp(date),
                "headline": np.random.choice(pool),
                "source"  : np.random.choice(["Reuters", "Bloomberg", "CNBC", "Economic Times", "Mint"]),
            })

    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


def fetch_stock_data(ticker: str, days: int = 30) -> pd.DataFrame:
    try:
        import yfinance as yf
        end   = datetime.today()
        start = end - timedelta(days=days + 10)
        hist  = yf.Ticker(ticker).history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d")
        )
        if hist.empty:
            return _fallback_stock(days)
        hist = hist.reset_index()
        hist["date"]       = pd.to_datetime(hist["Date"]).dt.normalize()
        hist["close"]      = hist["Close"].round(2)
        hist["pct_change"] = hist["Close"].pct_change().round(4) * 100
        hist["volume"]     = hist["Volume"]
        return hist[["date", "close", "pct_change", "volume"]].dropna().tail(days)
    except Exception as e:
        print(f"yfinance error: {e}")
        return _fallback_stock(days)


def _fallback_stock(days: int = 30) -> pd.DataFrame:
    np.random.seed(99)
    end_date = datetime.today()
    dates    = pd.bdate_range(end=end_date, periods=days)
    returns  = np.random.normal(0.001, 0.015, days)
    prices   = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        "date"      : dates,
        "close"     : prices.round(2),
        "pct_change": (returns * 100).round(4),
        "volume"    : np.random.randint(5_000_000, 50_000_000, days),
    })
