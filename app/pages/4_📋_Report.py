"""
📋 Report — Auto-generated analysis report for export.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import aggregate_by_regime, aggregate_by_symbol_regime
from src.analysis import run_all_comparisons, generate_insights

st.set_page_config(page_title="Report | HyperSentiment AI", page_icon="📋", layout="wide")

# ── Data check ────────────────────────────────────────────────────────────────
if "merged_df" not in st.session_state:
    st.warning("⚠️ Please load data from the Home page first.")
    st.stop()

merged_df: pd.DataFrame = st.session_state["merged_df"]
sentiment_df: pd.DataFrame = st.session_state["sentiment_df"]

# ── Computations ──────────────────────────────────────────────────────────────
regime_df = aggregate_by_regime(merged_df)
symbol_df = aggregate_by_symbol_regime(merged_df)
comparisons = run_all_comparisons(merged_df)
insights = generate_insights(regime_df, comparisons)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("# 📋 Analysis Report")
st.caption("Auto-generated summary — downloadable as Markdown")
st.divider()

# ── Build report ──────────────────────────────────────────────────────────────
report_lines: list[str] = []

report_lines.append("# HyperSentiment AI — Analysis Report")
report_lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
report_lines.append(f"\n**Data period:** {merged_df['date'].min().strftime('%Y-%m-%d')} "
                     f"to {merged_df['date'].max().strftime('%Y-%m-%d')}")
report_lines.append(f"\n**Total trades analyzed:** {len(merged_df):,}")
report_lines.append(f"\n**Active accounts:** {merged_df['account'].nunique()}")

# Overview
report_lines.append("\n---\n")
report_lines.append("## 1. Executive Summary\n")
for i, insight in enumerate(insights, 1):
    report_lines.append(f"{i}. {insight}")

# Regime stats
report_lines.append("\n---\n")
report_lines.append("## 2. Regime Comparison\n")
report_lines.append(regime_df.to_markdown(index=False))

# Statistical tests
report_lines.append("\n---\n")
report_lines.append("## 3. Statistical Significance\n")
if not comparisons.empty:
    report_lines.append(comparisons.to_markdown(index=False))
else:
    report_lines.append("_Insufficient data for statistical testing._")

# Symbol breakdown
report_lines.append("\n---\n")
report_lines.append("## 4. Symbol-Level Performance\n")
report_lines.append(symbol_df.to_markdown(index=False))

# Methodology
report_lines.append("\n---\n")
report_lines.append("## 5. Methodology\n")
report_lines.append("""
- **Sentiment alignment:** Trades are matched to daily sentiment using date-based inner join.
- **Statistical tests:** Welch's t-test (parametric) and Mann-Whitney U test (non-parametric) with α=0.05.
- **Effect size:** Cohen's d measures practical significance of differences.
- **Anomaly detection:** Z-score method with threshold of 3.0 standard deviations.
- **Win rate:** Fraction of trades with closedPnL > 0.
""")

# Disclaimer
report_lines.append("\n---\n")
report_lines.append("## 6. Disclaimer\n")
report_lines.append(
    "This analysis is for educational and research purposes only. "
    "Past performance does not guarantee future results. "
    "Do not use these findings as the sole basis for trading decisions."
)

full_report = "\n".join(report_lines)

# ── Display report ────────────────────────────────────────────────────────────
st.markdown(full_report)

st.divider()

# ── Download ──────────────────────────────────────────────────────────────────
st.download_button(
    "📥 Download Report (Markdown)",
    data=full_report,
    file_name=f"hypersentiment_report_{datetime.now().strftime('%Y%m%d')}.md",
    mime="text/markdown",
    type="primary",
)

# Also offer CSV exports
st.markdown("### 📦 Data Exports")
ec1, ec2, ec3 = st.columns(3)

with ec1:
    st.download_button(
        "📥 Regime Stats (CSV)",
        data=regime_df.to_csv(index=False),
        file_name="regime_comparison.csv",
        mime="text/csv",
    )

with ec2:
    st.download_button(
        "📥 Symbol Stats (CSV)",
        data=symbol_df.to_csv(index=False),
        file_name="symbol_performance.csv",
        mime="text/csv",
    )

with ec3:
    if not comparisons.empty:
        st.download_button(
            "📥 Statistical Tests (CSV)",
            data=comparisons.to_csv(index=False),
            file_name="statistical_tests.csv",
            mime="text/csv",
        )
