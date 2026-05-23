"""
Tests for data ingestion and utility functions.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    compute_win_rate,
    normalize_columns,
    safe_to_datetime,
    validate_required_columns,
)
from src.data_ingestion import load_sentiment, load_trades, merge_sentiment_trades


# ── Utils tests ───────────────────────────────────────────────────────────────

class TestNormalizeColumns:
    def test_basic(self):
        df = pd.DataFrame({"Foo Bar": [1], " Baz ": [2], "HELLO-WORLD": [3]})
        result = normalize_columns(df)
        assert list(result.columns) == ["foo_bar", "baz", "hello_world"]

    def test_special_characters(self):
        df = pd.DataFrame({"Price ($)": [1], "Win %": [2]})
        result = normalize_columns(df)
        assert "price" in result.columns
        assert "win" in result.columns


class TestSafeToDatetime:
    def test_standard_format(self):
        s = pd.Series(["2024-01-01", "2024-06-15"])
        result = safe_to_datetime(s)
        assert pd.api.types.is_datetime64_any_dtype(result)
        assert result.iloc[0].year == 2024

    def test_slash_format(self):
        s = pd.Series(["01/15/2024", "06/20/2024"])
        result = safe_to_datetime(s)
        assert pd.api.types.is_datetime64_any_dtype(result)


class TestValidateColumns:
    def test_all_present(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        validate_required_columns(df, ["a", "b"])  # should not raise

    def test_missing(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_required_columns(df, ["a", "b", "z"])


class TestWinRate:
    def test_basic(self):
        assert compute_win_rate(pd.Series([10, -5, 20, -1])) == 0.5

    def test_all_winners(self):
        assert compute_win_rate(pd.Series([1, 2, 3])) == 1.0

    def test_empty(self):
        assert compute_win_rate(pd.Series([], dtype=float)) == 0.0


# ── Data ingestion tests ─────────────────────────────────────────────────────

@pytest.fixture
def sample_sentiment_csv(tmp_path: Path) -> Path:
    """Create a minimal sentiment CSV for testing."""
    df = pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "Classification": ["Fear", "Greed", "Fear"],
        "Value": [25, 72, 30],
    })
    path = tmp_path / "sentiment.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_trades_csv(tmp_path: Path) -> Path:
    """Create a minimal trades CSV for testing."""
    df = pd.DataFrame({
        "account": ["0xabc", "0xabc", "0xdef"],
        "symbol": ["BTC", "ETH", "BTC"],
        "execution_price": [42000, 2300, 41500],
        "size": [0.5, 2.0, 1.0],
        "side": ["Buy", "Sell", "Buy"],
        "time": [
            "2024-01-01 10:30:00",
            "2024-01-02 14:00:00",
            "2024-01-03 09:15:00",
        ],
        "start_position": [0.5, -2.0, 1.0],
        "event": ["fill", "fill", "fill"],
        "closedPnl": [150.0, -80.0, 200.0],
        "leverage": [10, 5, 20],
    })
    path = tmp_path / "trades.csv"
    df.to_csv(path, index=False)
    return path


class TestLoadSentiment:
    def test_loads_correctly(self, sample_sentiment_csv: Path):
        df = load_sentiment(sample_sentiment_csv)
        assert len(df) == 3
        assert "date" in df.columns
        assert "classification" in df.columns
        assert set(df["classification"].unique()) == {"Fear", "Greed"}

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_sentiment(Path("/nonexistent/file.csv"))


class TestLoadTrades:
    def test_loads_correctly(self, sample_trades_csv: Path):
        df = load_trades(sample_trades_csv)
        assert len(df) == 3
        assert "date" in df.columns
        assert df["closedpnl"].dtype in [float, "float64"]

    def test_side_standardization(self, sample_trades_csv: Path):
        df = load_trades(sample_trades_csv)
        assert set(df["side"].unique()).issubset({"Buy", "Sell"})


class TestMerge:
    def test_merge(self, sample_sentiment_csv: Path, sample_trades_csv: Path):
        sent = load_sentiment(sample_sentiment_csv)
        trades = load_trades(sample_trades_csv)
        merged = merge_sentiment_trades(sent, trades)
        assert "classification" in merged.columns
        assert len(merged) == 3
