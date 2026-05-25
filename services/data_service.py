# services/data_service.py
import os
import io
from typing import Optional
import pandas as pd

TEXT_COLUMNS = [
    "text", "headline", "title", "news", "content",
    "article", "description", "summary", "message",
]
TICKER_COLUMNS = ["ticker", "symbol", "stock", "company", "scrip"]
SOURCE_COLUMNS = ["source", "publisher", "outlet", "media"]


def _find_column(df: pd.DataFrame, candidates: list) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    text_col = _find_column(df, TEXT_COLUMNS)
    if not text_col:
        raise ValueError(
            "No headline text column found. Rename a column to: text, headline, title, "
            "news, content, article, description, summary, or message."
        )

    ticker_col = _find_column(df, TICKER_COLUMNS)
    source_col = _find_column(df, SOURCE_COLUMNS)

    rows = []
    for _, row in df.iterrows():
        text = str(row[text_col]).strip()
        if not text or text.lower() == "nan" or len(text) < 4:
            continue
        ticker = "UNKNOWN"
        if ticker_col:
            t = str(row[ticker_col]).strip().upper()
            if t and t.lower() != "nan":
                ticker = t
        source = "Unknown"
        if source_col:
            s = str(row[source_col]).strip()
            if s and s.lower() != "nan":
                source = s
        rows.append({"text": text, "ticker": ticker, "source": source})

    if not rows:
        raise ValueError("No valid headlines found. Each row needs at least 4 characters of text.")

    out = pd.DataFrame(rows)
    return out.head(500)


def parse_csv(uploaded_file) -> pd.DataFrame:
    if hasattr(uploaded_file, "getvalue"):
        raw = uploaded_file.getvalue()
    elif isinstance(uploaded_file, bytes):
        raw = uploaded_file
    else:
        raw = uploaded_file.read()

    df = pd.read_csv(io.BytesIO(raw))
    if df.empty:
        raise ValueError("CSV file is empty.")
    return _normalize_dataframe(df)


def get_texts_list(df: pd.DataFrame) -> list:
    return df["text"].tolist()


def load_sample_data() -> pd.DataFrame:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "sample_data.csv")
    df = pd.read_csv(path)
    return _normalize_dataframe(df)
