# services/news_service.py
import os
from datetime import datetime, timedelta
from typing import Optional
import requests
from dotenv import load_dotenv

load_dotenv()

NEWS_API_URL = "https://newsapi.org/v2/everything"

US_FINANCE_DOMAINS = "reuters.com,bloomberg.com,cnbc.com,marketwatch.com"
INDIA_FINANCE_DOMAINS = (
    "economictimes.indiatimes.com,moneycontrol.com,livemint.com,"
    "business-standard.com,financialexpress.com,thehindubusinessline.com"
)
GLOBAL_FINANCE_DOMAINS = f"{US_FINANCE_DOMAINS},{INDIA_FINANCE_DOMAINS}"


def has_news_api_key() -> bool:
    key = os.getenv("NEWS_API_KEY", "").strip()
    if not key:
        return False
    placeholders = {"your_newsapi_key_here", "your_key_here", "xxx", "paste_key_here"}
    return key.lower() not in placeholders


def _domains_for_exchange(exchange_label: str) -> Optional[str]:
    """Return domain filter or None for unrestricted global search."""
    if "NSE" in exchange_label or "BSE" in exchange_label:
        return INDIA_FINANCE_DOMAINS
    if exchange_label.startswith("US"):
        return US_FINANCE_DOMAINS
    return None


def fetch_stock_headlines(
    query: str,
    max_results: int = 20,
    exchange_label: str = "US (NYSE / NASDAQ)",
) -> list:
    if not has_news_api_key():
        return []

    api_key = os.getenv("NEWS_API_KEY")
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=14)

    params = {
        "q": query,
        "from": start_date.strftime("%Y-%m-%d"),
        "to": end_date.strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": min(max_results, 100),
        "apiKey": api_key,
    }

    domains = _domains_for_exchange(exchange_label)
    if domains:
        params["domains"] = domains

    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=12)
        data = response.json()
        if data.get("status") != "ok":
            return []
        articles = data.get("articles", [])
        headlines = []
        for article in articles:
            title = article.get("title")
            if not title or title == "[Removed]":
                continue
            headlines.append({
                "text": title,
                "source": article.get("source", {}).get("name", "Unknown"),
                "url": article.get("url", ""),
                "publishedAt": article.get("publishedAt", ""),
            })
            if len(headlines) >= max_results:
                break
        return headlines
    except Exception:
        return []
