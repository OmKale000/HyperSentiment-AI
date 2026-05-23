"""
Feature engineering module.

Creates sentiment-based features and aggregated trader metrics.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger

from src.utils import compute_win_rate


# ── Sentiment Features ────────────────────────────────────────────────────────

def add_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived sentiment columns to the merged dataset.

    New columns:
        - is_fear / is_greed : binary flags
        - sentiment_streak   : consecutive days in same regime
        - regime_shift        : 1 on the day sentiment flips
    """
    df = df.copy()
    df["is_fear"] = (df["classification"] == "Fear").astype(int)
    df["is_greed"] = (df["classification"] == "Greed").astype(int)

    # Regime shift detection (per-day)
    daily = df.drop_duplicates(subset=["date"]).sort_values("date").copy()
    daily["_prev"] = daily["classification"].shift(1)
    daily["regime_shift"] = (daily["classification"] != daily["_prev"]).astype(int)

    # Streak length (consecutive days in same regime)
    streaks: list[int] = []
    current_streak = 0
    for shift in daily["regime_shift"]:
        if shift:
            current_streak = 1
        else:
            current_streak += 1
        streaks.append(current_streak)
    daily["sentiment_streak"] = streaks

    # Merge back
    merge_cols = ["date", "regime_shift", "sentiment_streak"]
    df = df.merge(daily[merge_cols], on="date", how="left")

    logger.info("Added sentiment features: is_fear, is_greed, regime_shift, sentiment_streak")
    return df


def add_trade_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived trade-level features.

    New columns:
        - is_winner       : closedpnl > 0
        - is_long / is_short : side flags
        - notional_value   : execution_price × size
        - pnl_per_unit     : closedpnl / size (where size > 0)
        - log_size         : log1p(size)
    """
    df = df.copy()

    if "closedpnl" in df.columns:
        df["is_winner"] = (df["closedpnl"] > 0).astype(int)

    if "side" in df.columns:
        df["is_long"] = (df["side"] == "Buy").astype(int)
        df["is_short"] = (df["side"] == "Sell").astype(int)

    if "execution_price" in df.columns and "size" in df.columns:
        df["notional_value"] = df["execution_price"] * df["size"]

    if "closedpnl" in df.columns and "size" in df.columns:
        df["pnl_per_unit"] = np.where(
            df["size"] > 0, df["closedpnl"] / df["size"], 0.0
        )

    if "size" in df.columns:
        df["log_size"] = np.log1p(df["size"].clip(lower=0))

    logger.info("Added trade features: is_winner, is_long, notional_value, etc.")
    return df


# ── Aggregations ──────────────────────────────────────────────────────────────

def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trades to daily level with performance metrics."""
    agg = df.groupby("date").agg(
        classification=("classification", "first"),
        trade_count=("closedpnl", "count"),
        total_pnl=("closedpnl", "sum"),
        mean_pnl=("closedpnl", "mean"),
        median_pnl=("closedpnl", "median"),
        std_pnl=("closedpnl", "std"),
        win_rate=("is_winner", "mean"),
        mean_leverage=("leverage", "mean"),
        mean_size=("size", "mean"),
        total_notional=("notional_value", "sum"),
        long_ratio=("is_long", "mean"),
        unique_accounts=("account", "nunique"),
        unique_symbols=("symbol", "nunique"),
    ).reset_index()

    if "sentiment_value" in df.columns:
        sv = df.groupby("date")["sentiment_value"].first().reset_index()
        agg = agg.merge(sv, on="date", how="left")

    logger.info("Daily aggregation: {} days", len(agg))
    return agg


def aggregate_by_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Compare key metrics between Fear and Greed regimes."""
    agg = df.groupby("classification").agg(
        trade_count=("closedpnl", "count"),
        total_pnl=("closedpnl", "sum"),
        mean_pnl=("closedpnl", "mean"),
        median_pnl=("closedpnl", "median"),
        std_pnl=("closedpnl", "std"),
        win_rate=("is_winner", "mean"),
        mean_leverage=("leverage", "mean"),
        median_leverage=("leverage", "median"),
        mean_size=("size", "mean"),
        long_ratio=("is_long", "mean"),
        unique_accounts=("account", "nunique"),
    ).reset_index()

    logger.info("Regime aggregation:\n{}", agg.to_string(index=False))
    return agg


def aggregate_by_account_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Per-account performance split by sentiment regime."""
    agg = df.groupby(["account", "classification"]).agg(
        trade_count=("closedpnl", "count"),
        total_pnl=("closedpnl", "sum"),
        mean_pnl=("closedpnl", "mean"),
        win_rate=("is_winner", "mean"),
        mean_leverage=("leverage", "mean"),
        mean_size=("size", "mean"),
        long_ratio=("is_long", "mean"),
    ).reset_index()

    return agg


def aggregate_by_symbol_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol performance split by sentiment regime."""
    agg = df.groupby(["symbol", "classification"]).agg(
        trade_count=("closedpnl", "count"),
        total_pnl=("closedpnl", "sum"),
        mean_pnl=("closedpnl", "mean"),
        win_rate=("is_winner", "mean"),
        mean_leverage=("leverage", "mean"),
    ).reset_index()

    return agg


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in sequence."""
    df = add_sentiment_features(df)
    df = add_trade_features(df)
    return df
