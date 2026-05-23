"""
Tests for feature engineering functions.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_engineering import (
    add_sentiment_features,
    add_trade_features,
    aggregate_by_regime,
    aggregate_daily,
)


@pytest.fixture
def merged_df() -> pd.DataFrame:
    """Minimal merged dataset for testing features."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
        "classification": ["Fear", "Fear", "Greed", "Greed"],
        "account": ["0xa", "0xb", "0xa", "0xb"],
        "symbol": ["BTC", "ETH", "BTC", "ETH"],
        "execution_price": [42000, 2300, 43000, 2400],
        "size": [1.0, 2.0, 0.5, 3.0],
        "side": ["Buy", "Sell", "Buy", "Buy"],
        "closedpnl": [100, -50, 200, -30],
        "leverage": [10, 5, 3, 20],
        "start_position": [1.0, -2.0, 0.5, 3.0],
        "event": ["fill", "fill", "fill", "fill"],
    })


class TestSentimentFeatures:
    def test_adds_columns(self, merged_df):
        result = add_sentiment_features(merged_df)
        assert "is_fear" in result.columns
        assert "is_greed" in result.columns
        assert "regime_shift" in result.columns
        assert "sentiment_streak" in result.columns

    def test_binary_flags_correct(self, merged_df):
        result = add_sentiment_features(merged_df)
        assert result.loc[result["classification"] == "Fear", "is_fear"].all()
        assert result.loc[result["classification"] == "Greed", "is_greed"].all()


class TestTradeFeatures:
    def test_adds_columns(self, merged_df):
        result = add_trade_features(merged_df)
        assert "is_winner" in result.columns
        assert "notional_value" in result.columns
        assert "pnl_per_unit" in result.columns
        assert "log_size" in result.columns

    def test_winner_flag(self, merged_df):
        result = add_trade_features(merged_df)
        assert result.iloc[0]["is_winner"] == 1   # pnl = 100
        assert result.iloc[1]["is_winner"] == 0   # pnl = -50


class TestAggregation:
    def test_regime_aggregation(self, merged_df):
        enriched = add_trade_features(merged_df)
        enriched = add_sentiment_features(enriched)
        result = aggregate_by_regime(enriched)
        assert len(result) == 2
        assert set(result["classification"]) == {"Fear", "Greed"}

    def test_daily_aggregation(self, merged_df):
        enriched = add_trade_features(merged_df)
        enriched = add_sentiment_features(enriched)
        result = aggregate_daily(enriched)
        assert len(result) == 2  # 2 unique dates
        assert "trade_count" in result.columns
        assert "total_pnl" in result.columns
