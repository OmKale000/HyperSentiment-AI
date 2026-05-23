"""
Shared utilities: logging setup, date helpers, validation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
from loguru import logger

# Allow imports when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LOG_FILE, LOG_LEVEL


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(level: str = LOG_LEVEL) -> None:
    """Configure project-wide logging with loguru."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )
    logger.add(
        str(LOG_FILE),
        level="DEBUG",
        rotation="5 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}",
    )
    logger.info("Logging initialized (level={})", level)


# ── Data helpers ──────────────────────────────────────────────────────────────

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip, and snake_case all column names.

    Examples
    --------
    >>> normalize_columns(pd.DataFrame({"Foo Bar": [1], " Baz ": [2]})).columns.tolist()
    ['foo_bar', 'baz']
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def safe_to_datetime(
    series: pd.Series,
    formats: Sequence[str] | None = None,
) -> pd.Series:
    """Parse a Series to datetime, trying multiple formats.

    Falls back to ``pd.to_datetime(..., infer_datetime_format=True)``
    if explicit formats fail.
    """
    formats = formats or [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%Y/%m/%d",
    ]
    for fmt in formats:
        try:
            return pd.to_datetime(series, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.to_datetime(series, format="mixed", dayfirst=False)


def summarize_df(df: pd.DataFrame, name: str = "DataFrame") -> dict:
    """Return a quick summary dict for logging / display."""
    return {
        "name": name,
        "rows": len(df),
        "columns": len(df.columns),
        "nulls": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "dtypes": df.dtypes.value_counts().to_dict(),
    }


def validate_required_columns(
    df: pd.DataFrame,
    required: list[str],
    source: str = "dataset",
) -> None:
    """Raise ValueError if any required columns are missing."""
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {source}: {sorted(missing)}. "
            f"Available: {sorted(df.columns)}"
        )


def compute_win_rate(pnl_series: pd.Series) -> float:
    """Fraction of trades with positive PnL. Returns 0.0 for empty series."""
    if len(pnl_series) == 0:
        return 0.0
    return float((pnl_series > 0).sum() / len(pnl_series))
