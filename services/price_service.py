# services/price_service.py
"""Live and historical stock prices via Yahoo Finance (supports NSE .NS, BSE .BO, US, global)."""

from datetime import datetime
import pandas as pd


def fetch_live_quote(yahoo_symbol: str) -> dict:
    """
    Fetch latest price snapshot and recent history for charting.
    Returns dict with price metrics and history DataFrame.
    """
    try:
        import yfinance as yf
    except ImportError:
        return {
            "ok": False,
            "error": "Install yfinance: pip install yfinance",
            "yahoo_symbol": yahoo_symbol,
        }

    try:
        ticker = yf.Ticker(yahoo_symbol)
        hist = ticker.history(period="1mo", interval="1d")
        if hist.empty:
            hist = ticker.history(period="5d", interval="1h")

        if hist.empty:
            return {
                "ok": False,
                "error": f"No price data found for '{yahoo_symbol}'. Check symbol and exchange suffix (.NS / .BO).",
                "yahoo_symbol": yahoo_symbol,
            }

        hist = hist.reset_index()
        date_col = "Date" if "Date" in hist.columns else hist.columns[0]
        hist["date"] = pd.to_datetime(hist[date_col]).dt.tz_localize(None)
        hist["close"] = hist["Close"].round(2)
        hist["open"] = hist["Open"].round(2)
        hist["high"] = hist["High"].round(2)
        hist["low"] = hist["Low"].round(2)
        hist["volume"] = hist["Volume"].fillna(0).astype(int)

        last = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else last
        price = float(last["close"])
        prev_price = float(prev["close"])
        change = price - prev_price
        change_pct = (change / prev_price * 100) if prev_price else 0.0

        name = yahoo_symbol
        tradingview_symbol = yahoo_symbol
        try:
            from services.ticker_service import resolve_tradingview_symbol
            info = ticker.info
            name = info.get("shortName") or info.get("longName") or yahoo_symbol
            currency = info.get("currency", "")
            tradingview_symbol = resolve_tradingview_symbol(yahoo_symbol)
        except Exception:
            currency = ""

        return {
            "ok": True,
            "yahoo_symbol": yahoo_symbol,
            "tradingview_symbol": tradingview_symbol,
            "name": name,
            "price": round(price, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "volume": int(last["volume"]),
            "currency": currency,
            "as_of": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "history": hist[["date", "open", "high", "low", "close", "volume"]].copy(),
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "yahoo_symbol": yahoo_symbol,
        }
