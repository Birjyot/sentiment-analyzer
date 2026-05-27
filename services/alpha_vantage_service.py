# services/alpha_vantage_service.py
"""Alpha Vantage API wrapper — historical OHLC, SMA, and RSI data.

Free tier: 25 requests/day.  Register at https://www.alphavantage.co for a key.
All functions fail gracefully if ALPHA_VANTAGE_API_KEY is missing.
"""

import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

ALPHA_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
BASE = "https://www.alphavantage.co/query"


def has_alpha_key() -> bool:
    """Return True if a usable Alpha Vantage API key is configured."""
    if not ALPHA_KEY:
        return False
    placeholders = {"your_alpha_vantage_api_key_here", "your_key_here", "xxx", "paste_key_here", ""}
    return ALPHA_KEY.strip().lower() not in placeholders


# ── DAILY OHLC ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_daily_ohlc(symbol: str, outputsize: str = "compact") -> pd.DataFrame:
    """Fetch daily OHLC data via TIME_SERIES_DAILY_ADJUSTED.

    outputsize: 'compact' (last 100 days) or 'full' (20+ years).
    Returns DataFrame with columns: date, open, high, low, close, volume.
    Returns empty DataFrame on failure.
    """
    if not has_alpha_key():
        return pd.DataFrame()

    try:
        resp = requests.get(
            BASE,
            params={
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol.upper(),
                "outputsize": outputsize,
                "apikey": ALPHA_KEY.strip(),
            },
            timeout=15,
        )
        data = resp.json()

        ts_key = "Time Series (Daily)"
        if ts_key not in data:
            # Could be rate limited or bad symbol
            return pd.DataFrame()

        rows = []
        for date_str, values in data[ts_key].items():
            rows.append({
                "date": pd.to_datetime(date_str),
                "open": float(values.get("1. open", 0)),
                "high": float(values.get("2. high", 0)),
                "low": float(values.get("3. low", 0)),
                "close": float(values.get("4. close", 0)),
                "volume": int(float(values.get("5. volume", 0))),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# ── SIMPLE MOVING AVERAGE ─────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_sma(symbol: str, interval: str = "daily", period: int = 20) -> pd.DataFrame:
    """Fetch SMA data from Alpha Vantage.

    Returns DataFrame with columns: date, sma.
    Overlay on candlestick chart as a purple line.
    """
    if not has_alpha_key():
        return pd.DataFrame()

    try:
        resp = requests.get(
            BASE,
            params={
                "function": "SMA",
                "symbol": symbol.upper(),
                "interval": interval,
                "time_period": period,
                "series_type": "close",
                "apikey": ALPHA_KEY.strip(),
            },
            timeout=15,
        )
        data = resp.json()

        ta_key = "Technical Analysis: SMA"
        if ta_key not in data:
            return pd.DataFrame()

        rows = []
        for date_str, values in data[ta_key].items():
            rows.append({
                "date": pd.to_datetime(date_str),
                "sma": float(values.get("SMA", 0)),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# ── RELATIVE STRENGTH INDEX ──────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_rsi(symbol: str, interval: str = "daily", period: int = 14) -> pd.DataFrame:
    """Fetch RSI data from Alpha Vantage.

    Returns DataFrame with columns: date, rsi.
    Show as a mini chart below the price chart.
    """
    if not has_alpha_key():
        return pd.DataFrame()

    try:
        resp = requests.get(
            BASE,
            params={
                "function": "RSI",
                "symbol": symbol.upper(),
                "interval": interval,
                "time_period": period,
                "series_type": "close",
                "apikey": ALPHA_KEY.strip(),
            },
            timeout=15,
        )
        data = resp.json()

        ta_key = "Technical Analysis: RSI"
        if ta_key not in data:
            return pd.DataFrame()

        rows = []
        for date_str, values in data[ta_key].items():
            rows.append({
                "date": pd.to_datetime(date_str),
                "rsi": float(values.get("RSI", 0)),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()
