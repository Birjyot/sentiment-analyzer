import pandas as pd
import numpy as np


def load_vader():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer(), "vader"
    except ImportError:
        return None, "custom"


def score_headline(headline: str, analyzer, mode: str) -> dict:
    if mode == "vader":
        scores = analyzer.polarity_scores(headline)
        return {"compound": scores["compound"], "positive": scores["pos"],
                "negative": scores["neg"], "neutral": scores["neu"]}
    else:
        return _custom_score(headline)


def _custom_score(text: str) -> dict:
    text_lower = text.lower()
    positive_words = {
        "record": 0.6, "beats": 0.7, "surge": 0.8, "growth": 0.5,
        "profit": 0.5, "gain": 0.5, "rise": 0.4, "strong": 0.5,
        "upgrade": 0.7, "buy": 0.5, "positive": 0.5, "boost": 0.6,
        "expand": 0.4, "innovation": 0.5, "partnership": 0.4,
        "better": 0.4, "exceed": 0.6, "outperform": 0.7,
        "rally": 0.6, "recovery": 0.5, "optimistic": 0.6,
        "landmark": 0.5, "innovative": 0.5, "secures": 0.4,
        "raises": 0.4, "launches": 0.3,
    }
    negative_words = {
        "miss": -0.6, "fall": -0.5, "decline": -0.5, "loss": -0.6,
        "downgrade": -0.7, "sell": -0.4, "concern": -0.4, "risk": -0.4,
        "cut": -0.5, "layoff": -0.7, "lawsuit": -0.8, "fraud": -0.9,
        "warning": -0.5, "slowing": -0.4, "pressure": -0.4,
        "resign": -0.6, "scrutiny": -0.5, "controversy": -0.6,
        "disruption": -0.5, "restructuring": -0.4, "warns": -0.5,
        "regulatory": -0.3, "investigation": -0.6, "penalty": -0.6,
    }
    score = 0.0
    count = 0
    for word, val in positive_words.items():
        if word in text_lower:
            score += val
            count += 1
    for word, val in negative_words.items():
        if word in text_lower:
            score += val
            count += 1
    compound = 0.0 if count == 0 else float(np.clip(score / (count + 1), -1, 1))
    pos = max(0, compound)
    neg = abs(min(0, compound))
    neu = 1 - pos - neg
    return {"compound": round(compound, 4), "positive": round(pos, 4),
            "negative": round(neg, 4), "neutral": round(neu, 4)}


def analyze_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    analyzer, mode = load_vader()
    scores = news_df["headline"].apply(lambda h: score_headline(h, analyzer, mode))
    news_df = news_df.copy()
    news_df["compound_score"] = scores.apply(lambda x: x["compound"])
    news_df["positive"]       = scores.apply(lambda x: x["positive"])
    news_df["negative"]       = scores.apply(lambda x: x["negative"])
    news_df["neutral"]        = scores.apply(lambda x: x["neutral"])
    news_df["sentiment_label"] = news_df["compound_score"].apply(
        lambda s: "🟢 Positive" if s >= 0.05 else ("🔴 Negative" if s <= -0.05 else "🟡 Neutral")
    )
    return news_df.sort_values("date").reset_index(drop=True)


def aggregate_daily_sentiment(scored_df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        scored_df.groupby("date")
        .agg(
            avg_sentiment  = ("compound_score", "mean"),
            sentiment_std  = ("compound_score", "std"),
            headline_count = ("headline", "count"),
            positive_count = ("sentiment_label", lambda x: (x == "🟢 Positive").sum()),
            negative_count = ("sentiment_label", lambda x: (x == "🔴 Negative").sum()),
        )
        .reset_index()
    )
    daily["sentiment_std"]     = daily["sentiment_std"].fillna(0)
    daily["avg_sentiment"]     = daily["avg_sentiment"].round(4)
    daily["rolling_sentiment"] = daily["avg_sentiment"].rolling(3, min_periods=1).mean().round(4)
    return daily


def merge_and_correlate(sentiment_daily: pd.DataFrame, stock_df: pd.DataFrame) -> tuple:
    sentiment_daily = sentiment_daily.copy()
    stock_df        = stock_df.copy()

    sentiment_daily["date"] = pd.to_datetime(sentiment_daily["date"]).dt.tz_localize(None)

    try:
        stock_df["date"] = pd.to_datetime(stock_df["date"]).dt.tz_convert(None)
    except TypeError:
        stock_df["date"] = pd.to_datetime(stock_df["date"]).dt.tz_localize(None)

    sentiment_daily["date"] = sentiment_daily["date"].dt.normalize()
    stock_df["date"]        = stock_df["date"].dt.normalize()

    merged = pd.merge(sentiment_daily, stock_df, on="date", how="inner")

    if len(merged) < 3:
        return merged, 0.0, 0.0

    same_day_corr = merged["avg_sentiment"].corr(merged["pct_change"])
    merged["next_day_return"] = merged["pct_change"].shift(-1)
    lagged       = merged.dropna(subset=["next_day_return"])
    lagged_corr  = lagged["avg_sentiment"].corr(lagged["next_day_return"])

    return (
        merged,
        round(same_day_corr, 4) if not np.isnan(same_day_corr) else 0.0,
        round(lagged_corr,   4) if not np.isnan(lagged_corr)   else 0.0,
    )