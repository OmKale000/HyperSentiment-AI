"""
HyperSentiment AI — Streamlit Dashboard

Main entry point. Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.utils import setup_logging
from src.data_ingestion import load_and_merge
from src.feature_engineering import build_all_features

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout=config.APP_LAYOUT,
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1A1D23 0%, #252830 100%);
        border: 1px solid #2D3139;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        color: #B0B0B0 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #151920 100%);
    }

    /* Headers */
    h1 {
        background: linear-gradient(90deg, #00D4FF, #7B61FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
    }

    /* Divider */
    hr {
        border-color: #2D3139;
    }

    /* DataFrames */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached) ────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading and processing data...")
def load_data():
    """Load, clean, merge, and feature-engineer all datasets."""
    setup_logging()

    sentiment_df, trades_df, merged_df = load_and_merge()
    merged_df = build_all_features(merged_df)

    return sentiment_df, trades_df, merged_df


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🧠 HyperSentiment AI")
    st.caption("Bitcoin Sentiment × Trader Performance")
    st.divider()

    st.markdown("### 📂 Data Source")
    sentiment_path = st.text_input(
        "Sentiment file",
        value=str(config.SENTIMENT_FILE),
        help="Path to Bitcoin sentiment CSV/Excel file",
    )
    trades_path = st.text_input(
        "Trades file",
        value=str(config.TRADES_FILE),
        help="Path to Hyperliquid trades CSV/Excel file",
    )

    st.divider()
    st.markdown(
        "Built by **HyperSentiment AI**  \n"
        "📊 Sentiment-driven trading analytics"
    )

# ── Load data ─────────────────────────────────────────────────────────────────
try:
    sentiment_df, trades_df, merged_df = load_data()
    st.session_state["sentiment_df"] = sentiment_df
    st.session_state["trades_df"] = trades_df
    st.session_state["merged_df"] = merged_df
    data_loaded = True
except FileNotFoundError as e:
    st.error(f"❌ Data file not found: {e}")
    st.info(
        "💡 Generate sample data by running:\n\n"
        "```bash\npython scripts/generate_sample_data.py\n```"
    )
    data_loaded = False
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    data_loaded = False

# ── Home Page ─────────────────────────────────────────────────────────────────
if data_loaded:
    st.markdown("# 🧠 HyperSentiment AI")
    st.markdown("#### Analyzing Bitcoin Sentiment × Trader Performance on Hyperliquid")
    st.divider()

    # Key stats row
    col1, col2, col3, col4, col5 = st.columns(5)

    n_trades = len(merged_df)
    n_days = merged_df["date"].nunique()
    n_accounts = merged_df["account"].nunique()
    total_pnl = merged_df["closedpnl"].sum()
    overall_wr = merged_df["is_winner"].mean() if "is_winner" in merged_df.columns else 0

    fear_days = (sentiment_df["classification"] == "Fear").sum()
    greed_days = (sentiment_df["classification"] == "Greed").sum()

    col1.metric("Total Trades", f"{n_trades:,}")
    col2.metric("Trading Days", f"{n_days}")
    col3.metric("Active Accounts", f"{n_accounts}")
    col4.metric("Net PnL", f"${total_pnl:,.0f}", delta=f"{'▲' if total_pnl > 0 else '▼'}")
    col5.metric("Win Rate", f"{overall_wr:.1%}")

    st.divider()

    # Regime overview
    st.markdown("### 📊 Sentiment Regime Breakdown")
    rc1, rc2, rc3 = st.columns([1, 1, 2])

    rc1.metric("😨 Fear Days", f"{fear_days}", delta=f"{fear_days / len(sentiment_df):.0%} of period")
    rc2.metric("🤑 Greed Days", f"{greed_days}", delta=f"{greed_days / len(sentiment_df):.0%} of period")

    with rc3:
        import plotly.graph_objects as go

        fig = go.Figure(data=[go.Pie(
            labels=["Fear", "Greed"],
            values=[fear_days, greed_days],
            marker=dict(colors=[config.FEAR_COLOR, config.GREED_COLOR]),
            hole=0.55,
            textinfo="label+percent",
            textfont=dict(size=14, color="#FAFAFA"),
        )])
        fig.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=200,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Quick data preview
    st.markdown("### 🔍 Data Preview")
    tab1, tab2, tab3 = st.tabs(["📈 Merged Data", "😨/🤑 Sentiment", "💱 Raw Trades"])

    with tab1:
        st.dataframe(
            merged_df.head(100),
            use_container_width=True,
            height=350,
        )
        st.caption(f"Showing first 100 of {len(merged_df):,} records")

    with tab2:
        st.dataframe(sentiment_df.head(50), use_container_width=True, height=350)

    with tab3:
        st.dataframe(trades_df.head(50), use_container_width=True, height=350)

    st.markdown("---")
    st.markdown(
        "👈 Use the **sidebar navigation** to explore detailed analysis, "
        "visualizations, and model insights."
    )
else:
    st.markdown("# 🧠 HyperSentiment AI")
    st.markdown("---")
    st.warning("Please ensure data files are available and reload the page.")
