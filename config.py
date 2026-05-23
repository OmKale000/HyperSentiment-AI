"""
Central configuration for HyperSentiment AI.
All paths, constants, and defaults are defined here for easy customization.
"""

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# ── Data paths (override via environment variables) ───────────────────────────
DATA_DIR: Path = Path(os.getenv("HYPERSENTIMENT_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

SENTIMENT_FILE: Path = Path(
    os.getenv("SENTIMENT_FILE", RAW_DATA_DIR / "bitcoin_sentiment.csv")
)
TRADES_FILE: Path = Path(
    os.getenv("TRADES_FILE", RAW_DATA_DIR / "hyperliquid_trades.csv")
)

# ── Ensure directories exist ─────────────────────────────────────────────────
for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ── Analysis constants ────────────────────────────────────────────────────────
SENTIMENT_LABELS: list[str] = ["Fear", "Greed"]
DEFAULT_DATE_COL: str = "date"
CLASSIFICATION_COL: str = "classification"

# ── Visualization defaults ────────────────────────────────────────────────────
CHART_STYLE: str = "seaborn-v0_8-darkgrid"
FEAR_COLOR: str = "#E74C3C"
GREED_COLOR: str = "#2ECC71"
NEUTRAL_COLOR: str = "#95A5A6"
PALETTE: dict[str, str] = {"Fear": FEAR_COLOR, "Greed": GREED_COLOR}
FIG_DPI: int = 150
FIG_SIZE: tuple[int, int] = (12, 6)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: Path = PROJECT_ROOT / "hypersentiment.log"

# ── Streamlit ─────────────────────────────────────────────────────────────────
APP_TITLE: str = "HyperSentiment AI"
APP_ICON: str = "🧠"
APP_LAYOUT: str = "wide"
