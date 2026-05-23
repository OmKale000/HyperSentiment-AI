"""
📊 Explorer — Interactive data exploration with filters and drill-downs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config

st.set_page_config(page_title="Explorer | HyperSentiment AI", page_icon="📊", layout="wide")

# ── Check data ────────────────────────────────────────────────────────────────
if "merged_df" not in st.session_state:
    st.warning("⚠️ Please load data from the Home page first.")
    st.stop()

merged_df: pd.DataFrame = st.session_state["merged_df"]
sentiment_df: pd.DataFrame = st.session_state["sentiment_df"]

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("# 📊 Data Explorer")
st.caption("Filter, slice, and explore the dataset interactively")
st.divider()

# ── Sidebar filters ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Filters")

    # Date range
    min_date = merged_df["date"].min().date()
    max_date = merged_df["date"].max().date()
    date_range = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # Sentiment
    regime_filter = st.multiselect(
        "Sentiment regime",
        options=["Fear", "Greed"],
        default=["Fear", "Greed"],
    )

    # Symbols
    all_symbols = sorted(merged_df["symbol"].unique()) if "symbol" in merged_df.columns else []
    symbol_filter = st.multiselect("Symbols", options=all_symbols, default=all_symbols)

    # Side
    side_filter = st.multiselect("Side", options=["Buy", "Sell"], default=["Buy", "Sell"])

    # Event type
    if "event" in merged_df.columns:
        all_events = sorted(merged_df["event"].unique())
        event_filter = st.multiselect("Event type", options=all_events, default=all_events)
    else:
        event_filter = None

    # PnL range
    pnl_min, pnl_max = float(merged_df["closedpnl"].min()), float(merged_df["closedpnl"].max())
    pnl_range = st.slider(
        "PnL range ($)",
        min_value=pnl_min,
        max_value=pnl_max,
        value=(pnl_min, pnl_max),
        format="$%.0f",
    )

# ── Apply filters ─────────────────────────────────────────────────────────────
filtered = merged_df.copy()

if len(date_range) == 2:
    filtered = filtered[
        (filtered["date"].dt.date >= date_range[0])
        & (filtered["date"].dt.date <= date_range[1])
    ]

filtered = filtered[filtered["classification"].isin(regime_filter)]

if "symbol" in filtered.columns:
    filtered = filtered[filtered["symbol"].isin(symbol_filter)]

if "side" in filtered.columns:
    filtered = filtered[filtered["side"].isin(side_filter)]

if event_filter is not None and "event" in filtered.columns:
    filtered = filtered[filtered["event"].isin(event_filter)]

filtered = filtered[
    (filtered["closedpnl"] >= pnl_range[0])
    & (filtered["closedpnl"] <= pnl_range[1])
]

# ── Stats row ─────────────────────────────────────────────────────────────────
st.markdown(f"### Filtered Results: **{len(filtered):,}** trades")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Trades", f"{len(filtered):,}")
c2.metric("Accounts", f"{filtered['account'].nunique() if 'account' in filtered.columns else 'N/A'}")
c3.metric("Net PnL", f"${filtered['closedpnl'].sum():,.0f}")
c4.metric("Avg PnL", f"${filtered['closedpnl'].mean():,.2f}")
c5.metric("Win Rate", f"{(filtered['closedpnl'] > 0).mean():.1%}")
c6.metric("Avg Leverage", f"{filtered['leverage'].mean():.1f}x" if "leverage" in filtered.columns else "N/A")

st.divider()

# ── Interactive charts ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 PnL Distribution", "🏦 By Symbol", "👤 Top Accounts", "📋 Data Table"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            filtered, x="closedpnl", color="classification",
            color_discrete_map=config.PALETTE,
            nbins=80, barmode="overlay", opacity=0.7,
            title="PnL Distribution by Regime",
            labels={"closedpnl": "Closed PnL ($)", "classification": "Regime"},
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            filtered, x="classification", y="closedpnl",
            color="classification", color_discrete_map=config.PALETTE,
            title="PnL Box Plot by Regime",
            labels={"closedpnl": "Closed PnL ($)", "classification": "Regime"},
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    if "symbol" in filtered.columns:
        symbol_stats = filtered.groupby(["symbol", "classification"]).agg(
            total_pnl=("closedpnl", "sum"),
            trade_count=("closedpnl", "count"),
            win_rate=("closedpnl", lambda x: (x > 0).mean()),
        ).reset_index()

        fig = px.bar(
            symbol_stats, x="symbol", y="total_pnl",
            color="classification", barmode="group",
            color_discrete_map=config.PALETTE,
            title="Total PnL by Symbol & Regime",
            labels={"total_pnl": "Total PnL ($)", "classification": "Regime"},
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            symbol_stats.pivot_table(
                index="symbol", columns="classification",
                values=["total_pnl", "trade_count", "win_rate"],
            ).round(4),
            use_container_width=True,
        )

with tab3:
    if "account" in filtered.columns:
        top_n = st.slider("Top N accounts", 5, 50, 15)
        account_stats = filtered.groupby("account").agg(
            total_pnl=("closedpnl", "sum"),
            trade_count=("closedpnl", "count"),
            win_rate=("closedpnl", lambda x: (x > 0).mean()),
            avg_leverage=("leverage", "mean"),
        ).reset_index().sort_values("total_pnl", ascending=False).head(top_n)

        fig = px.bar(
            account_stats, x="account", y="total_pnl",
            color="total_pnl",
            color_continuous_scale=["#E74C3C", "#FFD700", "#2ECC71"],
            title=f"Top {top_n} Accounts by Total PnL",
            labels={"total_pnl": "Total PnL ($)", "account": "Account"},
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(account_stats.round(4), use_container_width=True)

with tab4:
    st.dataframe(filtered, use_container_width=True, height=500)

    csv = filtered.to_csv(index=False)
    st.download_button(
        "📥 Download Filtered Data (CSV)",
        csv,
        file_name="hypersentiment_filtered.csv",
        mime="text/csv",
    )
