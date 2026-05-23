"""
Generate realistic sample datasets for development and testing.

Creates:
  - bitcoin_sentiment.csv   (daily Fear/Greed classification)
  - hyperliquid_trades.csv  (historical trader activity)
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from project root or scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DATA_DIR, SENTIMENT_LABELS


def generate_sentiment_data(
    start: str = "2024-01-01",
    end: str = "2024-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate daily Bitcoin Fear & Greed Index data.

    Simulates realistic sentiment with regime clustering: fear tends to
    persist for multi-day stretches and so does greed, mimicking real
    market psychology.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")

    # Markov-style transitions: 85% chance of staying in same regime
    classifications: list[str] = []
    current = rng.choice(SENTIMENT_LABELS)
    for _ in dates:
        if rng.random() < 0.15:
            current = "Greed" if current == "Fear" else "Fear"
        classifications.append(current)

    # Generate a numeric index (0-100) consistent with classification
    values: list[int] = []
    for cls in classifications:
        if cls == "Fear":
            values.append(int(rng.integers(5, 45)))
        else:
            values.append(int(rng.integers(55, 95)))

    df = pd.DataFrame(
        {"Date": dates, "Value": values, "Classification": classifications}
    )
    return df


def generate_trade_data(
    sentiment_df: pd.DataFrame,
    n_accounts: int = 50,
    avg_trades_per_day: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic Hyperliquid trader records.

    Trader behavior is influenced by sentiment:
      - In Fear periods: smaller sizes, higher leverage, more shorts, worse PnL.
      - In Greed periods: larger sizes, moderate leverage, more longs, better PnL.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    accounts = [f"0x{rng.integers(0, 2**32):08x}" for _ in range(n_accounts)]
    symbols = ["BTC", "ETH", "SOL", "DOGE", "ARB", "AVAX", "MATIC", "LINK", "OP", "APT"]
    events = ["fill", "liquidation", "funding"]
    event_weights = [0.88, 0.04, 0.08]

    base_prices = {
        "BTC": 42000, "ETH": 2300, "SOL": 95, "DOGE": 0.08,
        "ARB": 1.2, "AVAX": 35, "MATIC": 0.9, "LINK": 14, "OP": 3.2, "APT": 9.5,
    }

    rows: list[dict] = []
    for _, sent_row in sentiment_df.iterrows():
        date = sent_row["Date"]
        regime = sent_row["Classification"]
        is_fear = regime == "Fear"

        n_trades = rng.poisson(avg_trades_per_day)
        for _ in range(n_trades):
            account = rng.choice(accounts)
            symbol = rng.choice(symbols)
            base_price = base_prices[symbol]

            # Price variation
            price = base_price * (1 + rng.normal(0, 0.03))

            # Sentiment-driven behavior
            if is_fear:
                side = rng.choice(["Buy", "Sell"], p=[0.35, 0.65])
                leverage = float(rng.choice([5, 10, 20, 25, 50], p=[0.15, 0.25, 0.25, 0.20, 0.15]))
                size = abs(rng.lognormal(2, 1.2)) * 0.7
                pnl_bias = -0.4
            else:
                side = rng.choice(["Buy", "Sell"], p=[0.65, 0.35])
                leverage = float(rng.choice([2, 3, 5, 10, 20], p=[0.20, 0.25, 0.30, 0.15, 0.10]))
                size = abs(rng.lognormal(2.5, 1.0))
                pnl_bias = 0.3

            # Event type
            event = rng.choice(events, p=event_weights)
            if event == "liquidation":
                closed_pnl = -abs(rng.lognormal(4, 1.5))
            elif event == "funding":
                closed_pnl = rng.normal(0, 5)
            else:
                closed_pnl = rng.normal(pnl_bias, 1) * size * 10

            # Random time within the day
            hour = int(rng.integers(0, 24))
            minute = int(rng.integers(0, 60))
            second = int(rng.integers(0, 60))
            timestamp = pd.Timestamp(date) + pd.Timedelta(
                hours=hour, minutes=minute, seconds=second
            )

            start_position = abs(rng.lognormal(3, 1)) * (1 if side == "Buy" else -1)

            rows.append(
                {
                    "account": account,
                    "symbol": symbol,
                    "execution_price": round(price, 4),
                    "size": round(size, 4),
                    "side": side,
                    "time": timestamp,
                    "start_position": round(start_position, 4),
                    "event": event,
                    "closedPnl": round(closed_pnl, 2),
                    "leverage": int(leverage),
                }
            )

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    """Generate and save sample datasets."""
    print("Generating Bitcoin sentiment data...")
    sentiment_df = generate_sentiment_data()
    sentiment_path = RAW_DATA_DIR / "bitcoin_sentiment.csv"
    sentiment_df.to_csv(sentiment_path, index=False)
    print(f"  -> Saved {len(sentiment_df)} rows to {sentiment_path}")

    print("Generating Hyperliquid trade data...")
    trades_df = generate_trade_data(sentiment_df)
    trades_path = RAW_DATA_DIR / "hyperliquid_trades.csv"
    trades_df.to_csv(trades_path, index=False)
    print(f"  -> Saved {len(trades_df)} rows to {trades_path}")

    print("\nSample sentiment data:")
    print(sentiment_df.head(10).to_string(index=False))
    print(f"\nSample trade data:")
    print(trades_df.head(10).to_string(index=False))
    print("\n[OK] Done!")


if __name__ == "__main__":
    main()
