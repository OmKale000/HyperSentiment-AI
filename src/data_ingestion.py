"""
Data ingestion module.

Handles loading, cleaning, and merging of sentiment + trade datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from src.utils import (
    normalize_columns,
    safe_to_datetime,
    summarize_df,
    validate_required_columns,
)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_csv_or_excel(filepath: Path | str) -> pd.DataFrame:
    """Load a CSV or Excel file with automatic format detection.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not supported.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    ext = filepath.suffix.lower()
    logger.info("Loading {} ({:.1f} KB)", filepath.name, filepath.stat().st_size / 1024)

    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(filepath, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    logger.info("  → Loaded {} rows × {} cols", len(df), len(df.columns))
    return df


# ── Sentiment ─────────────────────────────────────────────────────────────────

def load_sentiment(filepath: Path | str) -> pd.DataFrame:
    """Load and clean the Bitcoin sentiment dataset.

    Expected raw columns (case-insensitive):
        Date, Classification, [Value]

    Returns
    -------
    pd.DataFrame
        Columns: date (datetime), classification (str), value (float, optional).
    """
    df = load_csv_or_excel(filepath)
    df = normalize_columns(df)

    # Accept common column name variants
    rename_map: dict[str, str] = {}
    for col in df.columns:
        if col in ("timestamp", "datetime", "day"):
            rename_map[col] = "date"
        if col in ("label", "class", "sentiment", "category"):
            rename_map[col] = "classification"
    df = df.rename(columns=rename_map)

    validate_required_columns(df, ["date", "classification"], source="sentiment")

    # Parse dates
    df["date"] = safe_to_datetime(df["date"])
    df = df.dropna(subset=["date"])

    # Standardize classification labels
    df["classification"] = (
        df["classification"]
        .astype(str)
        .str.strip()
        .str.title()
        .replace({"Neutral": "Fear"})  # treat neutral as cautious
    )
    valid_labels = {"Fear", "Greed"}
    df = df[df["classification"].isin(valid_labels)].copy()

    # Drop duplicates (keep first per date)
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="first")
    df = df.reset_index(drop=True)

    info = summarize_df(df, "sentiment")
    logger.info("Sentiment cleaned: {}", info)
    return df


# ── Trades ────────────────────────────────────────────────────────────────────

_TRADE_RENAME = {
    "exec_price": "execution_price",
    "executionprice": "execution_price",
    "price": "execution_price",
    "qty": "size",
    "quantity": "size",
    "amount": "size",
    "direction": "side",
    "timestamp": "time",
    "datetime": "time",
    "closed_pnl": "closedpnl",
    "pnl": "closedpnl",
    "realized_pnl": "closedpnl",
    "lev": "leverage",
}


def load_trades(filepath: Path | str) -> pd.DataFrame:
    """Load and clean the Hyperliquid trader dataset.

    Expected raw columns (flexible naming):
        account, symbol, execution_price, size, side, time,
        start_position, event, closedPnl, leverage

    Returns
    -------
    pd.DataFrame
        Cleaned and typed trade records.
    """
    df = load_csv_or_excel(filepath)
    df = normalize_columns(df)

    # Flexible renames
    for old, new in _TRADE_RENAME.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    essential = ["time", "closedpnl"]
    validate_required_columns(df, essential, source="trades")

    # Parse timestamps
    df["time"] = safe_to_datetime(df["time"])
    df = df.dropna(subset=["time"])

    # Derive a date column for merging
    df["date"] = df["time"].dt.normalize()

    # Numeric coercions
    for col in ["execution_price", "size", "closedpnl", "leverage", "start_position"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Side standardization
    if "side" in df.columns:
        df["side"] = df["side"].astype(str).str.strip().str.title()
        df["side"] = df["side"].replace({"Long": "Buy", "Short": "Sell"})

    # Event standardization
    if "event" in df.columns:
        df["event"] = df["event"].astype(str).str.strip().str.lower()

    # Drop fully-null rows and reset index
    df = df.dropna(how="all").reset_index(drop=True)

    info = summarize_df(df, "trades")
    logger.info("Trades cleaned: {}", info)
    return df


# ── Merge ─────────────────────────────────────────────────────────────────────

def merge_sentiment_trades(
    sentiment_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    how: str = "inner",
) -> pd.DataFrame:
    """Align trades with daily sentiment by date.

    Each trade gets the sentiment label for the day it occurred.

    Parameters
    ----------
    sentiment_df : pd.DataFrame
        Must have columns: date, classification.
    trades_df : pd.DataFrame
        Must have column: date.
    how : str
        Merge type (default 'inner' keeps only matched days).

    Returns
    -------
    pd.DataFrame
        Trades enriched with classification column.
    """
    sentiment_slim = sentiment_df[["date", "classification"]].copy()
    if "value" in sentiment_df.columns:
        sentiment_slim["sentiment_value"] = sentiment_df["value"]

    merged = trades_df.merge(sentiment_slim, on="date", how=how)

    n_before, n_after = len(trades_df), len(merged)
    logger.info(
        "Merge complete: {} trades → {} matched ({:.1f}% coverage)",
        n_before,
        n_after,
        100 * n_after / max(n_before, 1),
    )
    return merged


# ── Convenience ───────────────────────────────────────────────────────────────

def load_and_merge(
    sentiment_path: Optional[Path | str] = None,
    trades_path: Optional[Path | str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One-call loader: returns (sentiment_df, trades_df, merged_df).

    Uses config defaults when paths are None.
    """
    import config  # deferred to allow standalone testing

    sentiment_path = sentiment_path or config.SENTIMENT_FILE
    trades_path = trades_path or config.TRADES_FILE

    sentiment_df = load_sentiment(sentiment_path)
    trades_df = load_trades(trades_path)
    merged_df = merge_sentiment_trades(sentiment_df, trades_df)

    return sentiment_df, trades_df, merged_df
