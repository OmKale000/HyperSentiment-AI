"""
📈 Analysis — Deep statistical analysis and professional visualizations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.feature_engineering import (
    aggregate_daily,
    aggregate_by_regime,
    aggregate_by_symbol_regime,
)
from src.analysis import (
    run_all_comparisons,
    sentiment_correlation_matrix,
    detect_pnl_anomalies,
    analyze_regime_transitions,
    generate_insights,
)
from src.visualization import (
    plot_sentiment_timeline,
    plot_pnl_by_regime,
    plot_daily_pnl_trend,
    plot_win_rate_comparison,
    plot_leverage_distribution,
    plot_symbol_performance,
    plot_trade_frequency,
    plot_correlation_heatmap,
    plot_cumulative_pnl,
)

st.set_page_config(page_title="Analysis | HyperSentiment AI", page_icon="📈", layout="wide")

# ── Data check ────────────────────────────────────────────────────────────────
if "merged_df" not in st.session_state:
    st.warning("⚠️ Please load data from the Home page first.")
    st.stop()

merged_df: pd.DataFrame = st.session_state["merged_df"]
sentiment_df: pd.DataFrame = st.session_state["sentiment_df"]

# ── Aggregations ──────────────────────────────────────────────────────────────
daily_df = aggregate_daily(merged_df)
regime_df = aggregate_by_regime(merged_df)
symbol_regime_df = aggregate_by_symbol_regime(merged_df)
comparisons = run_all_comparisons(merged_df)
corr_matrix = sentiment_correlation_matrix(daily_df)
transition_df = analyze_regime_transitions(daily_df)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("# 📈 Deep Analysis")
st.caption("Statistical tests, regime comparisons, and visual insights")
st.divider()

# ── Key Insights ──────────────────────────────────────────────────────────────
insights = generate_insights(regime_df, comparisons, daily_df)

with st.expander("🔑 **Key Insights** (auto-generated)", expanded=True):
    for i, insight in enumerate(insights, 1):
        st.markdown(f"**{i}.** {insight}")

st.divider()

# ── Regime Comparison Table ───────────────────────────────────────────────────
st.markdown("### 📋 Regime Comparison Summary")

display_regime = regime_df.copy()
format_map = {
    "trade_count": "{:,.0f}",
    "total_pnl": "${:,.2f}",
    "mean_pnl": "${:,.2f}",
    "median_pnl": "${:,.2f}",
    "std_pnl": "${:,.2f}",
    "win_rate": "{:.2%}",
    "mean_leverage": "{:.1f}x",
    "median_leverage": "{:.1f}x",
    "mean_size": "{:.2f}",
    "long_ratio": "{:.2%}",
}

for col, fmt in format_map.items():
    if col in display_regime.columns:
        display_regime[col] = display_regime[col].apply(lambda x: fmt.format(x))

st.dataframe(display_regime, use_container_width=True, hide_index=True)

st.divider()

# ── Statistical Tests ─────────────────────────────────────────────────────────
st.markdown("### 🧪 Statistical Significance Tests")
st.caption("Welch's t-test and Mann-Whitney U test comparing Fear vs Greed")

if not comparisons.empty:
    display_comp = comparisons.copy()
    display_comp["significance"] = display_comp["significant"].map(
        {True: "✅ Significant", False: "❌ Not Significant"}
    )
    st.dataframe(display_comp, use_container_width=True, hide_index=True)
else:
    st.info("Insufficient data for statistical tests.")

st.divider()

# ── Visualizations ────────────────────────────────────────────────────────────
st.markdown("### 📊 Visualizations")

viz_tabs = st.tabs([
    "🌡️ Sentiment Timeline",
    "💰 PnL by Regime",
    "📈 Daily PnL Trend",
    "📊 Key Metrics",
    "🔧 Leverage",
    "🏛️ Symbols",
    "📉 Trade Frequency",
    "🔗 Correlations",
    "📈 Cumulative PnL",
])

with viz_tabs[0]:
    fig = plot_sentiment_timeline(sentiment_df)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with viz_tabs[1]:
    fig = plot_pnl_by_regime(merged_df)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with viz_tabs[2]:
    fig = plot_daily_pnl_trend(daily_df)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with viz_tabs[3]:
    fig = plot_win_rate_comparison(regime_df)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with viz_tabs[4]:
    fig = plot_leverage_distribution(merged_df)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with viz_tabs[5]:
    fig = plot_symbol_performance(symbol_regime_df)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with viz_tabs[6]:
    fig = plot_trade_frequency(daily_df)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with viz_tabs[7]:
    fig = plot_correlation_heatmap(corr_matrix)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with viz_tabs[8]:
    fig = plot_cumulative_pnl(daily_df)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.divider()

# ── Regime Transition Analysis ────────────────────────────────────────────────
st.markdown("### 🔄 Regime Transition Analysis")
st.caption("How do trading metrics change on days when sentiment flips?")

if not transition_df.empty:
    st.dataframe(transition_df, use_container_width=True, hide_index=True)
else:
    st.info("No regime transition data available.")

# ── Anomalies ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown("### ⚠️ PnL Anomalies (Z-score > 3)")

anomalies = detect_pnl_anomalies(merged_df, z_threshold=3.0)
if len(anomalies) > 0:
    st.metric("Anomalous Trades", f"{len(anomalies):,}")
    display_cols = [c for c in ["account", "symbol", "side", "closedpnl", "pnl_zscore", "leverage", "classification", "date"] if c in anomalies.columns]
    st.dataframe(
        anomalies[display_cols].sort_values("pnl_zscore", key=abs, ascending=False).head(50),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.success("No extreme PnL anomalies detected.")
