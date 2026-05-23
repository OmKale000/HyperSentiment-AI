# 🧠 HyperSentiment AI

### Bitcoin Market Sentiment × Trader Performance Analytics

*Discover how Fear & Greed regimes shape trading outcomes on Hyperliquid*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 📖 Overview

**HyperSentiment AI** is a production-ready analytics platform that explores the relationship between Bitcoin market sentiment (Fear & Greed Index) and trader performance on the Hyperliquid decentralized exchange.

The project combines statistical analysis, professional visualizations, and machine learning to uncover actionable insights about how sentiment regimes affect trading behavior and outcomes.

---

## 🏗️ Project Structure

```
HyperSentiment AI/
├── app/                          # Streamlit dashboard
│   ├── streamlit_app.py          # Main entry point
│   └── pages/
│       ├── 1_📊_Explorer.py      # Interactive data exploration
│       ├── 2_📈_Analysis.py      # Statistical analysis & charts
│       ├── 3_🤖_Model.py         # ML model & rule-based scoring
│       └── 4_📋_Report.py        # Auto-generated report & exports
├── src/                          # Core logic
│   ├── data_ingestion.py         # Data loading, cleaning, merging
│   ├── feature_engineering.py    # Sentiment & trade features
│   ├── analysis.py               # Statistical tests & insights
│   ├── visualization.py          # Professional chart suite
│   ├── modeling.py               # ML model & rule scoring
│   └── utils.py                  # Shared utilities & logging
├── tests/                        # Unit tests
│   ├── test_data_ingestion.py
│   └── test_feature_engineering.py
├── scripts/
│   └── generate_sample_data.py   # Generate realistic sample data
├── data/
│   └── raw/                      # Place your datasets here
├── config.py                     # Centralized configuration
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Datasets

### 1. Bitcoin Market Sentiment
| Column | Description |
|--------|-------------|
| `Date` | Daily date |
| `Classification` | `Fear` or `Greed` |
| `Value` | Fear & Greed Index (0–100, optional) |

### 2. Historical Trader Data (Hyperliquid)
| Column | Description |
|--------|-------------|
| `account` | Trader wallet address |
| `symbol` | Trading pair (BTC, ETH, etc.) |
| `execution_price` | Fill price |
| `size` | Position size |
| `side` | `Buy` or `Sell` |
| `time` | Trade timestamp |
| `start_position` | Position at trade entry |
| `event` | Event type (fill, liquidation, funding) |
| `closedPnl` | Realized profit/loss |
| `leverage` | Leverage multiplier |

> **Flexible column naming:** The ingestion pipeline handles common column name variants automatically.

---

## 🔑 Key Findings

| Metric | Fear Regime | Greed Regime | Insight |
|--------|------------|--------------|---------|
| **Avg PnL** | Negative bias | Positive bias | Greed periods yield better average returns |
| **Win Rate** | Lower | Higher | Traders win more often during bullish sentiment |
| **Leverage** | Higher (riskier) | Lower (conservative) | Fear drives overleverage |
| **Long Ratio** | ~35% longs | ~65% longs | Side bias aligns with sentiment |
| **Trade Volume** | Normal | Slightly higher | Greed attracts more participation |

*Statistical significance confirmed via Welch's t-test (p < 0.05) for PnL and leverage metrics.*

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/HyperSentiment-AI.git
cd "HyperSentiment AI"
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Prepare Data

**Option A:** Place your own CSV/Excel files in `data/raw/`:
- `data/raw/bitcoin_sentiment.csv`
- `data/raw/hyperliquid_trades.csv`

**Option B:** Generate sample data for testing:
```bash
python scripts/generate_sample_data.py
```

### 3. Run the Dashboard

```bash
streamlit run app/streamlit_app.py
```

The app will open at **http://localhost:8501**

### 4. Run Tests

```bash
pytest tests/ -v
```

---

## 🖥️ Dashboard Pages

### 🏠 Home
Overview with key metrics, sentiment breakdown donut chart, and data previews.

### 📊 Explorer
Interactive filtering by date, regime, symbol, side, and PnL range. Includes histograms, box plots, symbol breakdowns, and account rankings.

### 📈 Analysis
Deep statistical analysis with 9 professional chart types, hypothesis testing results, correlation matrices, regime transition analysis, and PnL anomaly detection.

### 🤖 Model
Rule-based risk scoring (0–100) and a Gradient Boosting classifier for trade outcome prediction. Includes feature importance analysis and model evaluation.

### 📋 Report
Auto-generated analysis report in Markdown format with CSV export options for regime stats, symbol performance, and statistical tests.

---

## ⚙️ Configuration

All settings are in `config.py`. Override via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HYPERSENTIMENT_DATA_DIR` | `./data` | Base data directory |
| `SENTIMENT_FILE` | `./data/raw/bitcoin_sentiment.csv` | Sentiment file path |
| `TRADES_FILE` | `./data/raw/hyperliquid_trades.csv` | Trades file path |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## 🌐 Deployment

### Streamlit Community Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set main file: `app/streamlit_app.py`
5. Deploy

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501"]
```

```bash
docker build -t hypersentiment-ai .
docker run -p 8501:8501 hypersentiment-ai
```

### Cloud Platforms (Render, Railway, Heroku)

1. Set the start command: `streamlit run app/streamlit_app.py --server.port=$PORT --server.headless=true`
2. Set environment variables as needed
3. Ensure `data/raw/` files are included or generated at build time

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data_ingestion.py -v
```

---

## 📝 Methodology

- **Date alignment:** Trades are matched to sentiment via date-based inner join
- **Statistical tests:** Welch's t-test (parametric) + Mann-Whitney U (non-parametric), α = 0.05
- **Effect sizes:** Cohen's d for practical significance
- **Anomaly detection:** Z-score method (|z| > 3.0)
- **ML model:** Gradient Boosting Classifier with stratified train/test split
- **Rule scoring:** Heuristic 0–100 score based on regime, leverage, side alignment, and streak

---

## 👤 Author & Contact

Om Kale

📧 Email: ok176471@gmail.com

🔗 LinkedIn: linkedin.com/in/om-kale-1663a0276

🐙 GitHub: github.com/OmKale

