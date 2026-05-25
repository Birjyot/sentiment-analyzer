# services/ticker_service.py
"""Resolve tickers for global exchanges (US, NSE, BSE, and custom Yahoo Finance symbols)."""

EXCHANGE_OPTIONS = {
    "US (NYSE / NASDAQ)": {"suffix": "", "tv_prefix": "", "currency": "USD"},
    "NSE (India)": {"suffix": ".NS", "tv_prefix": "NSE", "currency": "INR"},
    "BSE (India)": {"suffix": ".BO", "tv_prefix": "BSE", "currency": "INR"},
    "London (LSE)": {"suffix": ".L", "tv_prefix": "LSE", "currency": "GBP"},
    "Tokyo (TSE)": {"suffix": ".T", "tv_prefix": "TSE", "currency": "JPY"},
    "Hong Kong": {"suffix": ".HK", "tv_prefix": "HKEX", "currency": "HKD"},
    "Custom (Yahoo symbol)": {"suffix": None, "tv_prefix": "", "currency": ""},
}

NSE_POPULAR = [
    ("Reliance Industries", "RELIANCE"),
    ("TCS", "TCS"),
    ("HDFC Bank", "HDFCBANK"),
    ("Infosys", "INFY"),
    ("ICICI Bank", "ICICIBANK"),
    ("SBI", "SBIN"),
    ("Bharti Airtel", "BHARTIARTL"),
    ("ITC", "ITC"),
    ("Wipro", "WIPRO"),
    ("Tata Motors", "TATAMOTORS"),
    ("Zomato", "ZOMATO"),
    ("Adani Enterprises", "ADANIENT"),
]

BSE_POPULAR = [
    ("Reliance Industries", "RELIANCE"),
    ("TCS", "TCS"),
    ("HDFC Bank", "HDFCBANK"),
    ("Infosys", "INFY"),
    ("SBI", "SBIN"),
]

US_POPULAR = [
    ("Apple", "AAPL"),
    ("Microsoft", "MSFT"),
    ("NVIDIA", "NVDA"),
    ("Tesla", "TSLA"),
    ("Amazon", "AMZN"),
    ("Google", "GOOGL"),
    ("Meta", "META"),
    ("S&P 500 ETF", "SPY"),
]


def normalize_ticker(symbol: str, exchange_label: str) -> dict:
    """
    Build Yahoo Finance symbol and metadata from user input.
    Examples: RELIANCE + NSE -> RELIANCE.NS, AAPL + US -> AAPL
    """
    raw = (symbol or "").strip().upper()
    if not raw:
        raise ValueError("Enter a stock symbol.")

    meta = EXCHANGE_OPTIONS.get(exchange_label, EXCHANGE_OPTIONS["US (NYSE / NASDAQ)"])

    if exchange_label == "Custom (Yahoo symbol)":
        yahoo_symbol = raw
    elif meta["suffix"] is None:
        yahoo_symbol = raw
    else:
        base = raw
        for suf in (".NS", ".BO", ".L", ".T", ".HK"):
            if base.endswith(suf):
                base = base[: -len(suf)]
                break
        yahoo_symbol = f"{base}{meta['suffix']}"

    tv_symbol = resolve_tradingview_symbol(yahoo_symbol, exchange_label)

    company_search = raw.split(".")[0].replace("-", " ")

    return {
        "yahoo_symbol": yahoo_symbol,
        "tradingview_symbol": tv_symbol,
        "display_ticker": yahoo_symbol,
        "exchange": exchange_label,
        "currency": meta["currency"] or "—",
        "news_query": company_search,
    }


def resolve_tradingview_symbol(yahoo_symbol: str, exchange_label: str = "") -> str:
    """
    Map Yahoo symbol to TradingView symbol (e.g. OLAELEC.NS -> NSE:OLAELEC).
    """
    sym = yahoo_symbol.upper().strip()
    base = sym.split(".")[0]

    if sym.endswith(".NS") or "NSE" in exchange_label:
        return f"NSE:{base}"
    if sym.endswith(".BO") or "BSE" in exchange_label:
        return f"BSE:{base}"
    if sym.endswith(".L") or "London" in exchange_label:
        return f"LSE:{base}"
    if sym.endswith(".T") or "Tokyo" in exchange_label:
        return f"TSE:{base}"
    if sym.endswith(".HK") or "Hong Kong" in exchange_label:
        return f"HKEX:{base}"

    try:
        import yfinance as yf
        info = yf.Ticker(sym).info
        exch = str(info.get("exchange", "") or "").upper()
        code = (info.get("symbol") or base).upper()
        if "NSE" in exch or exch in ("NSI", "XNSE"):
            return f"NSE:{code}"
        if "BSE" in exch or exch == "XBOM":
            return f"BSE:{code}"
        market = (info.get("market") or "").upper()
        if market in ("US", "NASDAQ", "NYSE"):
            return code
    except Exception:
        pass

    if exchange_label.startswith("US"):
        return base
    return sym


def tradingview_widget_html(tv_symbol: str, height: int = 450) -> str:
    """Embed TradingView chart — passes symbol via tv.js so it cannot fall back to Apple."""
    import hashlib
    sym = tv_symbol.replace("\\", "").replace('"', "")
    cid = "tv_" + hashlib.md5(sym.encode()).hexdigest()[:10]
    return f"""
<div style="height:{height}px;width:100%;">
  <div id="{cid}" style="height:100%;width:100%;"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
  (function() {{
    if (typeof TradingView === "undefined") return;
    new TradingView.widget({{
      "autosize": true,
      "symbol": "{sym}",
      "interval": "D",
      "timezone": "Asia/Kolkata",
      "theme": "dark",
      "style": "1",
      "locale": "en",
      "toolbar_bg": "#1e293b",
      "enable_publishing": false,
      "allow_symbol_change": true,
      "container_id": "{cid}"
    }});
  }})();
  </script>
</div>
"""
