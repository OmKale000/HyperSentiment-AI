"""
🤖 Model — ML model training, evaluation, and rule-based scoring.
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
from src.modeling import (
    apply_rule_scores,
    train_model,
    get_model_summary,
    FEATURE_COLS,
)

st.set_page_config(page_title="Model | HyperSentiment AI", page_icon="🤖", layout="wide")

# ── Data check ────────────────────────────────────────────────────────────────
if "merged_df" not in st.session_state:
    st.warning("⚠️ Please load data from the Home page first.")
    st.stop()

merged_df: pd.DataFrame = st.session_state["merged_df"]

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("# 🤖 Predictive Modeling")
st.caption("Rule-based scoring and ML-based trade outcome prediction")
st.divider()

# ── Rule-Based Scoring ────────────────────────────────────────────────────────
st.markdown("### 📏 Rule-Based Risk Score")
st.markdown(
    "A heuristic score (0–100) estimating trade success likelihood based on "
    "sentiment regime, leverage, side alignment, and streak length."
)

scored_df = apply_rule_scores(merged_df)

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        scored_df, x="risk_score", color="classification",
        color_discrete_map=config.PALETTE,
        nbins=50, barmode="overlay", opacity=0.7,
        title="Risk Score Distribution by Regime",
        labels={"risk_score": "Risk Score (0-100)", "classification": "Regime"},
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Score vs actual PnL
    sample = scored_df.sample(min(2000, len(scored_df)), random_state=42)
    fig = px.scatter(
        sample, x="risk_score", y="closedpnl",
        color="classification",
        color_discrete_map=config.PALETTE,
        opacity=0.4, size_max=6,
        title="Risk Score vs Actual PnL",
        labels={"risk_score": "Risk Score", "closedpnl": "Closed PnL ($)"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#FFD700", opacity=0.5)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

# Score bucket analysis
st.markdown("#### Score Bucket Performance")
scored_df["score_bucket"] = pd.cut(
    scored_df["risk_score"],
    bins=[0, 25, 50, 75, 100],
    labels=["0-25 (High Risk)", "25-50 (Moderate)", "50-75 (Favorable)", "75-100 (Strong)"],
)
bucket_stats = scored_df.groupby("score_bucket", observed=True).agg(
    trades=("closedpnl", "count"),
    mean_pnl=("closedpnl", "mean"),
    win_rate=("is_winner", "mean"),
    avg_leverage=("leverage", "mean"),
).reset_index()

st.dataframe(bucket_stats.round(4), use_container_width=True, hide_index=True)

st.divider()

# ── ML Model ──────────────────────────────────────────────────────────────────
st.markdown("### 🧠 Gradient Boosting Classifier")
st.markdown(
    "A lightweight ML model predicting whether a trade will be profitable, "
    "using sentiment and behavioral features."
)

# Check feature availability
available_features = [f for f in FEATURE_COLS if f in merged_df.columns]
missing_features = [f for f in FEATURE_COLS if f not in merged_df.columns]

if missing_features:
    st.warning(f"Missing features: {missing_features}. Model will use available features only.")

if "is_winner" not in merged_df.columns:
    st.error("Target variable 'is_winner' not found. Cannot train model.")
    st.stop()

if len(available_features) < 2:
    st.error("Insufficient features for model training.")
    st.stop()

# Train button
if st.button("🚀 Train Model", type="primary"):
    with st.spinner("Training Gradient Boosting model..."):
        try:
            result = train_model(merged_df)
            st.session_state["model_result"] = result
            st.success("✅ Model trained successfully!")
        except Exception as e:
            st.error(f"❌ Model training failed: {e}")

# Display results if model exists
if "model_result" in st.session_state:
    result = st.session_state["model_result"]
    metrics = result["metrics"]
    importance = result["feature_importance"]

    # Metrics
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    mc2.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
    mc3.metric("Train Size", f"{metrics['train_size']:,}")
    mc4.metric("Test Size", f"{metrics['test_size']:,}")

    # Feature importance chart
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            importance, x="importance", y="feature",
            orientation="h",
            title="Feature Importance",
            color="importance",
            color_continuous_scale=["#2D3139", "#00D4FF", "#7B61FF"],
            labels={"importance": "Importance", "feature": "Feature"},
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=350,
            yaxis=dict(autorange="reversed"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Classification report
        report = metrics["classification_report"]
        report_df = pd.DataFrame(report).transpose()
        st.markdown("#### Classification Report")
        st.dataframe(report_df.round(3), use_container_width=True)

    # Model summary
    with st.expander("📄 Full Model Summary"):
        st.markdown(get_model_summary(result))

else:
    st.info("👆 Click **Train Model** to build and evaluate the classifier.")
