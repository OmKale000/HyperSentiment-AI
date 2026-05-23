"""
Statistical analysis module.

Provides hypothesis tests, correlation analysis, and insight extraction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


# ── Hypothesis Testing ────────────────────────────────────────────────────────

def compare_regimes(
    df: pd.DataFrame,
    metric: str = "closedpnl",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compare a metric between Fear and Greed regimes using Welch's t-test
    and Mann-Whitney U test.

    Returns
    -------
    dict with keys: metric, fear_mean, greed_mean, t_stat, t_pvalue,
    u_stat, u_pvalue, significant, effect_size_cohens_d
    """
    fear = df.loc[df["classification"] == "Fear", metric].dropna()
    greed = df.loc[df["classification"] == "Greed", metric].dropna()

    if len(fear) < 5 or len(greed) < 5:
        logger.warning("Insufficient data for regime comparison on '{}'", metric)
        return {"metric": metric, "error": "insufficient data"}

    t_stat, t_pval = stats.ttest_ind(fear, greed, equal_var=False)
    u_stat, u_pval = stats.mannwhitneyu(fear, greed, alternative="two-sided")

    # Cohen's d
    pooled_std = np.sqrt(
        ((len(fear) - 1) * fear.std() ** 2 + (len(greed) - 1) * greed.std() ** 2)
        / (len(fear) + len(greed) - 2)
    )
    cohens_d = (greed.mean() - fear.mean()) / pooled_std if pooled_std > 0 else 0.0

    result = {
        "metric": metric,
        "fear_mean": round(float(fear.mean()), 4),
        "greed_mean": round(float(greed.mean()), 4),
        "fear_median": round(float(fear.median()), 4),
        "greed_median": round(float(greed.median()), 4),
        "t_statistic": round(float(t_stat), 4),
        "t_pvalue": round(float(t_pval), 6),
        "u_statistic": round(float(u_stat), 2),
        "u_pvalue": round(float(u_pval), 6),
        "significant": bool(t_pval < alpha),
        "cohens_d": round(float(cohens_d), 4),
    }

    sig_str = "✅ SIGNIFICANT" if result["significant"] else "❌ not significant"
    logger.info(
        "Regime comparison [{}]: Fear μ={:.2f}, Greed μ={:.2f} — {} (p={:.4f}, d={:.2f})",
        metric, result["fear_mean"], result["greed_mean"],
        sig_str, result["t_pvalue"], result["cohens_d"],
    )
    return result


def run_all_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """Run regime comparisons on all relevant numeric metrics."""
    metrics = ["closedpnl", "size", "leverage", "notional_value", "pnl_per_unit"]
    metrics = [m for m in metrics if m in df.columns]

    results = [compare_regimes(df, m) for m in metrics]
    return pd.DataFrame(results)


# ── Correlation Analysis ──────────────────────────────────────────────────────

def sentiment_correlation_matrix(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlations between sentiment value and daily trading metrics."""
    numeric_cols = [
        "sentiment_value", "trade_count", "total_pnl", "mean_pnl",
        "win_rate", "mean_leverage", "mean_size", "long_ratio",
    ]
    available = [c for c in numeric_cols if c in daily_df.columns]

    if len(available) < 2:
        logger.warning("Not enough numeric columns for correlation matrix")
        return pd.DataFrame()

    corr = daily_df[available].corr()
    logger.info("Correlation matrix computed ({} features)", len(available))
    return corr


# ── Anomaly Detection ─────────────────────────────────────────────────────────

def detect_pnl_anomalies(
    df: pd.DataFrame,
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    """Flag trades with extreme PnL (Z-score > threshold)."""
    if "closedpnl" not in df.columns:
        return pd.DataFrame()

    pnl = df["closedpnl"]
    mean, std = pnl.mean(), pnl.std()
    if std == 0:
        return pd.DataFrame()

    df = df.copy()
    df["pnl_zscore"] = (pnl - mean) / std
    anomalies = df[df["pnl_zscore"].abs() > z_threshold].copy()

    logger.info(
        "Anomaly detection: {} trades flagged (z > {:.1f})",
        len(anomalies), z_threshold,
    )
    return anomalies


# ── Regime Shift Analysis ────────────────────────────────────────────────────

def analyze_regime_transitions(
    daily_df: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze trader behavior around sentiment regime transitions.

    Compares metrics on regime-shift days vs. stable days.
    """
    if "regime_shift" not in daily_df.columns:
        logger.warning("regime_shift column not found; skipping transition analysis")
        return pd.DataFrame()

    shift_days = daily_df[daily_df["regime_shift"] == 1]
    stable_days = daily_df[daily_df["regime_shift"] == 0]

    metrics = ["total_pnl", "trade_count", "win_rate", "mean_leverage"]
    metrics = [m for m in metrics if m in daily_df.columns]

    rows: list[dict] = []
    for m in metrics:
        rows.append({
            "metric": m,
            "shift_day_mean": round(float(shift_days[m].mean()), 4) if len(shift_days) else None,
            "stable_day_mean": round(float(stable_days[m].mean()), 4) if len(stable_days) else None,
            "shift_days_count": len(shift_days),
            "stable_days_count": len(stable_days),
        })

    return pd.DataFrame(rows)


# ── Summary Insights ──────────────────────────────────────────────────────────

def generate_insights(
    regime_stats: pd.DataFrame,
    comparisons: pd.DataFrame,
    daily_df: pd.DataFrame | None = None,
) -> list[str]:
    """Generate plain-English bullet-point insights from the analysis."""
    insights: list[str] = []

    if regime_stats.empty or comparisons.empty:
        return ["Insufficient data to generate insights."]

    # Regime stats insights
    fear = regime_stats[regime_stats["classification"] == "Fear"]
    greed = regime_stats[regime_stats["classification"] == "Greed"]

    if not fear.empty and not greed.empty:
        fear_pnl = fear["mean_pnl"].values[0]
        greed_pnl = greed["mean_pnl"].values[0]
        insights.append(
            f"Average trade PnL during Greed is ${greed_pnl:,.2f} vs ${fear_pnl:,.2f} "
            f"during Fear — a {'positive' if greed_pnl > fear_pnl else 'negative'} "
            f"sentiment premium of ${abs(greed_pnl - fear_pnl):,.2f}."
        )

        fear_wr = fear["win_rate"].values[0]
        greed_wr = greed["win_rate"].values[0]
        insights.append(
            f"Win rate is {greed_wr:.1%} in Greed vs {fear_wr:.1%} in Fear "
            f"({'+' if greed_wr > fear_wr else ''}{(greed_wr - fear_wr)*100:.1f}pp difference)."
        )

        fear_lev = fear["mean_leverage"].values[0]
        greed_lev = greed["mean_leverage"].values[0]
        insights.append(
            f"Average leverage is {fear_lev:.1f}x in Fear vs {greed_lev:.1f}x in Greed — "
            f"traders use {'higher' if fear_lev > greed_lev else 'lower'} leverage during fearful markets."
        )

        fear_lr = fear["long_ratio"].values[0]
        greed_lr = greed["long_ratio"].values[0]
        insights.append(
            f"Long ratio is {greed_lr:.1%} in Greed vs {fear_lr:.1%} in Fear, "
            f"confirming {'directional alignment' if greed_lr > fear_lr else 'contrarian behavior'} "
            f"with sentiment."
        )

    # Significance insights
    sig_metrics = comparisons[comparisons.get("significant", pd.Series(dtype=bool)) == True]
    if not sig_metrics.empty:
        names = sig_metrics["metric"].tolist()
        insights.append(
            f"Statistically significant differences (p < 0.05) found in: {', '.join(names)}."
        )
    else:
        insights.append(
            "No statistically significant differences found at p < 0.05 level."
        )

    return insights
