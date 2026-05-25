# services/sentiment.py
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

FINANCE_BOOSTERS = {
    "bullish": 3.0,
    "bearish": -3.0,
    "surge": 2.5,
    "crash": -2.5,
    "soar": 2.5,
    "plunge": -2.5,
    "rally": 2.0,
    "selloff": -2.5,
    "beat": 2.0,
    "miss": -2.0,
    "outperform": 2.5,
    "underperform": -2.0,
    "record": 1.5,
    "collapse": -3.0,
    "boom": 2.5,
    "recession": -2.5,
    "downgrade": -2.0,
    "upgrade": 2.0,
    "default": -2.5,
    "blowout": 3.0,
}
for word, score in FINANCE_BOOSTERS.items():
    analyzer.lexicon[word] = score


def get_market_signal(compound: float) -> str:
    if compound >= 0.05:
        return "Bullish"
    if compound <= -0.05:
        return "Bearish"
    return "Neutral"


def get_signal_strength(compound: float) -> str:
    abs_c = abs(compound)
    if abs_c >= 0.6:
        return "Strong"
    if abs_c >= 0.3:
        return "Moderate"
    if abs_c >= 0.05:
        return "Weak"
    return "Flat"


def _format_sentiment_score(compound: float) -> str:
    return f"{compound:+.2f}"


def _market_health_score(avg_compound: float) -> int:
    normalized = (avg_compound + 1) / 2
    return int(round(max(0, min(100, normalized * 100))))


def _overall_signal(bullish_pct: float, bearish_pct: float, avg_compound: float) -> str:
    if bullish_pct >= 45 and avg_compound > 0.05 and bullish_pct > bearish_pct + 10:
        return "BULLISH 🟢"
    if bearish_pct >= 45 and avg_compound < -0.05 and bearish_pct > bullish_pct + 10:
        return "BEARISH 🔴"
    if avg_compound > 0.08 and bullish_pct > bearish_pct:
        return "BULLISH 🟢"
    if avg_compound < -0.08 and bearish_pct > bullish_pct:
        return "BEARISH 🔴"
    return "MIXED ⚪"


def analyze_single(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {
            "text": text,
            "signal": "Neutral",
            "signal_strength": "Flat",
            "compound": 0.0,
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 1.0,
            "confidence": 0.0,
            "sentiment_score": "+0.00",
        }

    scores = analyzer.polarity_scores(text)
    compound = round(scores["compound"], 4)
    signal = get_market_signal(compound)

    return {
        "text": text,
        "signal": signal,
        "signal_strength": get_signal_strength(compound),
        "compound": compound,
        "positive": round(scores["pos"], 4),
        "negative": round(scores["neg"], 4),
        "neutral": round(scores["neu"], 4),
        "confidence": round(abs(compound) * 100, 1),
        "sentiment_score": _format_sentiment_score(compound),
    }


def analyze_batch(texts: list) -> tuple:
    results = [analyze_single(t) for t in texts if (t or "").strip()]
    total = len(results)

    if total == 0:
        empty_summary = {
            "total": 0,
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "bullish_pct": 0.0,
            "bearish_pct": 0.0,
            "neutral_pct": 0.0,
            "avg_compound": 0.0,
            "overall_signal": "MIXED ⚪",
            "market_health_score": 50,
            "strong_signals_count": 0,
            "top_bullish": [],
            "top_bearish": [],
        }
        return [], empty_summary

    bullish_count = sum(1 for r in results if r["signal"] == "Bullish")
    bearish_count = sum(1 for r in results if r["signal"] == "Bearish")
    neutral_count = sum(1 for r in results if r["signal"] == "Neutral")
    avg_compound = round(sum(r["compound"] for r in results) / total, 4)
    bullish_pct = round(bullish_count / total * 100, 1)
    bearish_pct = round(bearish_count / total * 100, 1)
    neutral_pct = round(neutral_count / total * 100, 1)

    sorted_results = sorted(results, key=lambda x: x["compound"])
    top_bearish = [r["text"] for r in sorted_results[:3]]
    top_bullish = [r["text"] for r in sorted(results, key=lambda x: x["compound"], reverse=True)[:3]]

    summary = {
        "total": total,
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "neutral_count": neutral_count,
        "bullish_pct": bullish_pct,
        "bearish_pct": bearish_pct,
        "neutral_pct": neutral_pct,
        "avg_compound": avg_compound,
        "overall_signal": _overall_signal(bullish_pct, bearish_pct, avg_compound),
        "market_health_score": _market_health_score(avg_compound),
        "strong_signals_count": sum(1 for r in results if abs(r["compound"]) >= 0.5),
        "top_bullish": top_bullish,
        "top_bearish": top_bearish,
    }

    return results, summary


def analyze_by_ticker(results: list) -> dict:
    groups = {}
    for r in results:
        ticker = r.get("ticker", "UNKNOWN")
        if ticker not in groups:
            groups[ticker] = []
        groups[ticker].append(r)

    out = {}
    for ticker, items in groups.items():
        count = len(items)
        avg_compound = round(sum(x["compound"] for x in items) / count, 4) if count else 0.0
        signal = get_market_signal(avg_compound)
        out[ticker] = {
            "ticker": ticker,
            "count": count,
            "avg_compound": avg_compound,
            "signal": signal,
            "health_score": _market_health_score(avg_compound),
        }
    return out
