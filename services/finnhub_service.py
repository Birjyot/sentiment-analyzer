# services/finnhub_service.py
"""Finnhub.io API wrapper — real-time US stock quotes, company news & profiles.

Free tier: 60 API calls/minute.  Register at https://finnhub.io for a key.
All functions fail gracefully if FINNHUB_API_KEY is missing or invalid.
"""

import os
from datetime import datetime, timedelta

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
BASE = "https://finnhub.io/api/v1"


def has_finnhub_key() -> bool:
    """Return True if a usable Finnhub API key is configured."""
    if not FINNHUB_KEY:
        return False
    placeholders = {"your_finnhub_api_key_here", "your_key_here", "xxx", "paste_key_here", ""}
    return FINNHUB_KEY.strip().lower() not in placeholders


# ── REAL-TIME QUOTE ───────────────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def get_quote(symbol: str) -> dict:
    """Fetch real-time price data from Finnhub /quote endpoint.

    Returns dict with: price, change, change_pct, high, low, open, prev_close, ok, error
    """
    if not has_finnhub_key():
        return {"ok": False, "error": "FINNHUB_API_KEY not configured."}

    try:
        resp = requests.get(
            f"{BASE}/quote",
            params={"symbol": symbol.upper(), "token": FINNHUB_KEY.strip()},
            timeout=10,
        )
        data = resp.json()

        # Finnhub returns c=0 when symbol is invalid
        if data.get("c", 0) == 0 and data.get("d") is None:
            return {"ok": False, "error": f"No quote data for '{symbol}'. Check symbol."}

        return {
            "ok": True,
            "price": data.get("c", 0),
            "change": data.get("d", 0),
            "change_pct": data.get("dp", 0),
            "high": data.get("h", 0),
            "low": data.get("l", 0),
            "open": data.get("o", 0),
            "prev_close": data.get("pc", 0),
        }
    except Exception as e:
        return {"ok": False, "error": f"Finnhub quote error: {e}"}


# ── COMPANY NEWS ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def get_company_news(symbol: str, from_date: str = "", to_date: str = "") -> list:
    """Fetch recent company news articles from Finnhub.

    from_date / to_date format: "2024-01-01".  Defaults to last 7 days.
    Returns list of dicts: headline, summary, source, url, datetime, sentiment_signal.
    Each headline is analyzed with VADER before returning.
    """
    if not has_finnhub_key():
        return []

    if not to_date:
        to_date = datetime.utcnow().strftime("%Y-%m-%d")
    if not from_date:
        from_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

    try:
        resp = requests.get(
            f"{BASE}/company-news",
            params={
                "symbol": symbol.upper(),
                "from": from_date,
                "to": to_date,
                "token": FINNHUB_KEY.strip(),
            },
            timeout=12,
        )
        articles = resp.json()
        if not isinstance(articles, list):
            return []

        # Lazy import to avoid circular dependency
        from services.sentiment import analyze_single

        results = []
        for art in articles[:30]:  # cap at 30
            headline = art.get("headline", "").strip()
            if not headline:
                continue

            sentiment = analyze_single(headline)
            ts = art.get("datetime", 0)
            dt_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else ""

            results.append({
                "headline": headline,
                "summary": art.get("summary", ""),
                "source": art.get("source", "Unknown"),
                "url": art.get("url", ""),
                "datetime": dt_str,
                "date": dt_str[:10] if dt_str else "",
                "sentiment_signal": sentiment["signal"],
                "sentiment_score": sentiment["sentiment_score"],
                "compound": sentiment["compound"],
                "confidence": sentiment["confidence"],
            })

        return results
    except Exception:
        return []


# ── COMPANY PROFILE ───────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_company_profile(symbol: str) -> dict:
    """Fetch company profile from Finnhub /stock/profile2.

    Returns dict: name, logo, industry, market_cap, country, exchange, ok.
    """
    if not has_finnhub_key():
        return {"ok": False, "error": "FINNHUB_API_KEY not configured."}

    try:
        resp = requests.get(
            f"{BASE}/stock/profile2",
            params={"symbol": symbol.upper(), "token": FINNHUB_KEY.strip()},
            timeout=10,
        )
        data = resp.json()

        if not data.get("name"):
            return {"ok": False, "error": f"No profile for '{symbol}'."}

        # Format market cap nicely
        mc = data.get("marketCapitalization", 0)  # millions
        if mc >= 1_000_000:
            mc_str = f"{mc / 1_000_000:.2f}T"
        elif mc >= 1_000:
            mc_str = f"{mc / 1_000:.2f}B"
        else:
            mc_str = f"{mc:.0f}M"

        return {
            "ok": True,
            "name": data.get("name", ""),
            "logo": data.get("logo", ""),
            "industry": data.get("finnhubIndustry", ""),
            "market_cap": mc_str,
            "market_cap_raw": mc,
            "country": data.get("country", ""),
            "exchange": data.get("exchange", ""),
            "ticker": data.get("ticker", symbol),
            "ipo": data.get("ipo", ""),
            "weburl": data.get("weburl", ""),
        }
    except Exception as e:
        return {"ok": False, "error": f"Finnhub profile error: {e}"}


# ── SYMBOL SEARCH ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def search_symbol(query: str) -> list:
    """Search for a stock symbol by company name.

    Returns list of dicts: symbol, description, type.
    """
    if not has_finnhub_key():
        return []

    try:
        resp = requests.get(
            f"{BASE}/search",
            params={"q": query, "token": FINNHUB_KEY.strip()},
            timeout=10,
        )
        data = resp.json()
        results = []
        for item in data.get("result", [])[:15]:
            results.append({
                "symbol": item.get("symbol", ""),
                "description": item.get("description", ""),
                "type": item.get("type", ""),
            })
        return results
    except Exception:
        return []


# ── GENERAL MARKET NEWS ───────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def get_market_news(category: str = "general") -> list:
    """Fetch general market news from Finnhub /news.

    category options: general, forex, crypto, merger.
    Returns list of dicts with headline, summary, source, url, datetime.
    Each headline is analyzed with VADER.
    """
    if not has_finnhub_key():
        return []

    try:
        resp = requests.get(
            f"{BASE}/news",
            params={"category": category, "token": FINNHUB_KEY.strip()},
            timeout=12,
        )
        articles = resp.json()
        if not isinstance(articles, list):
            return []

        from services.sentiment import analyze_single

        results = []
        for art in articles[:30]:
            headline = art.get("headline", "").strip()
            if not headline:
                continue

            sentiment = analyze_single(headline)
            ts = art.get("datetime", 0)
            dt_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else ""

            results.append({
                "headline": headline,
                "summary": art.get("summary", ""),
                "source": art.get("source", "Unknown"),
                "url": art.get("url", ""),
                "datetime": dt_str,
                "date": dt_str[:10] if dt_str else "",
                "sentiment_signal": sentiment["signal"],
                "sentiment_score": sentiment["sentiment_score"],
                "compound": sentiment["compound"],
                "confidence": sentiment["confidence"],
            })

        return results
    except Exception:
        return []
