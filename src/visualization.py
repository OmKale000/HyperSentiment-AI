"""
Visualization module.

Professional-grade charts for sentiment vs. trader performance analysis.
All functions return matplotlib Figure objects for flexible usage.
"""

from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

import config


# ── Style setup ───────────────────────────────────────────────────────────────

def _apply_style() -> None:
    """Apply consistent chart styling."""
    try:
        plt.style.use(config.CHART_STYLE)
    except OSError:
        plt.style.use("seaborn-v0_8-darkgrid")

    plt.rcParams.update({
        "figure.facecolor": "#0E1117",
        "axes.facecolor": "#1A1D23",
        "axes.edgecolor": "#2D3139",
        "axes.labelcolor": "#FAFAFA",
        "text.color": "#FAFAFA",
        "xtick.color": "#B0B0B0",
        "ytick.color": "#B0B0B0",
        "grid.color": "#2D3139",
        "grid.alpha": 0.5,
        "legend.facecolor": "#1A1D23",
        "legend.edgecolor": "#2D3139",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


_apply_style()

PALETTE = config.PALETTE
FEAR_COLOR = config.FEAR_COLOR
GREED_COLOR = config.GREED_COLOR


# ── Chart Functions ───────────────────────────────────────────────────────────

def plot_sentiment_timeline(
    sentiment_df: pd.DataFrame,
    figsize: tuple = (14, 4),
) -> plt.Figure:
    """Color-coded sentiment timeline showing Fear/Greed regimes."""
    fig, ax = plt.subplots(figsize=figsize, dpi=config.FIG_DPI)

    colors = sentiment_df["classification"].map(PALETTE)
    if "value" in sentiment_df.columns:
        ax.bar(sentiment_df["date"], sentiment_df["value"], color=colors, alpha=0.8, width=1.0)
        ax.set_ylabel("Fear & Greed Index")
        ax.axhline(50, color="#FFD700", linestyle="--", alpha=0.5, label="Neutral (50)")
    else:
        # Binary strip
        y = sentiment_df["classification"].map({"Fear": 0, "Greed": 1})
        ax.bar(sentiment_df["date"], y + 0.5, color=colors, alpha=0.8, width=1.0)
        ax.set_ylabel("Regime")
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["Fear", "Greed"])

    ax.set_title("Bitcoin Market Sentiment Over Time", fontweight="bold", pad=12)
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=FEAR_COLOR, label="Fear"),
        Patch(facecolor=GREED_COLOR, label="Greed"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    fig.tight_layout()
    return fig


def plot_pnl_by_regime(
    df: pd.DataFrame,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Box + strip plot comparing PnL distributions across regimes."""
    fig, ax = plt.subplots(figsize=figsize, dpi=config.FIG_DPI)

    # Clip extreme outliers for visualization
    q99 = df["closedpnl"].quantile(0.99)
    q01 = df["closedpnl"].quantile(0.01)
    plot_df = df[(df["closedpnl"] >= q01) & (df["closedpnl"] <= q99)].copy()

    sns.boxplot(
        data=plot_df, x="classification", y="closedpnl",
        palette=PALETTE, ax=ax, width=0.5, showfliers=False,
        boxprops=dict(alpha=0.7),
    )
    sns.stripplot(
        data=plot_df.sample(min(1000, len(plot_df)), random_state=42),
        x="classification", y="closedpnl",
        palette=PALETTE, ax=ax, alpha=0.15, size=3, jitter=0.3,
    )

    ax.axhline(0, color="#FFD700", linestyle="--", alpha=0.6, linewidth=1)
    ax.set_title("Trade PnL Distribution: Fear vs Greed", fontweight="bold", pad=12)
    ax.set_xlabel("Sentiment Regime")
    ax.set_ylabel("Closed PnL ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    return fig


def plot_daily_pnl_trend(
    daily_df: pd.DataFrame,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Daily total PnL with sentiment regime background shading."""
    fig, ax = plt.subplots(figsize=figsize, dpi=config.FIG_DPI)

    # Background regime shading
    for _, row in daily_df.iterrows():
        color = PALETTE.get(row["classification"], "#333")
        ax.axvspan(
            row["date"] - pd.Timedelta(hours=12),
            row["date"] + pd.Timedelta(hours=12),
            alpha=0.08, color=color, linewidth=0,
        )

    # PnL line
    ax.plot(daily_df["date"], daily_df["total_pnl"], color="#00D4FF", linewidth=1.5, alpha=0.9)
    ax.fill_between(
        daily_df["date"], daily_df["total_pnl"], 0,
        where=daily_df["total_pnl"] >= 0, alpha=0.2, color=GREED_COLOR,
    )
    ax.fill_between(
        daily_df["date"], daily_df["total_pnl"], 0,
        where=daily_df["total_pnl"] < 0, alpha=0.2, color=FEAR_COLOR,
    )

    ax.axhline(0, color="#FFD700", linestyle="--", alpha=0.4)
    ax.set_title("Daily Aggregate PnL with Sentiment Regime", fontweight="bold", pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Total PnL ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_win_rate_comparison(
    regime_df: pd.DataFrame,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """Grouped bar chart comparing win rates and other metrics by regime."""
    fig, axes = plt.subplots(1, 3, figsize=(figsize[0] * 1.5, figsize[1]), dpi=config.FIG_DPI)

    metrics = [
        ("win_rate", "Win Rate", "{:.1%}"),
        ("mean_leverage", "Avg Leverage", "{:.1f}x"),
        ("long_ratio", "Long Ratio", "{:.1%}"),
    ]

    for ax, (col, title, fmt) in zip(axes, metrics):
        if col not in regime_df.columns:
            ax.set_visible(False)
            continue

        colors = [PALETTE.get(c, "#888") for c in regime_df["classification"]]
        bars = ax.bar(regime_df["classification"], regime_df[col], color=colors, alpha=0.85, width=0.5)

        for bar, val in zip(bars, regime_df[col]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                fmt.format(val), ha="center", va="bottom", fontweight="bold",
                color="#FAFAFA", fontsize=11,
            )

        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("")
        ax.set_xlabel("")

    fig.suptitle("Key Metrics by Sentiment Regime", fontweight="bold", fontsize=15, y=1.02)
    fig.tight_layout()
    return fig


def plot_leverage_distribution(
    df: pd.DataFrame,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Leverage distribution split by sentiment regime."""
    fig, ax = plt.subplots(figsize=figsize, dpi=config.FIG_DPI)

    for regime, color in PALETTE.items():
        data = df.loc[df["classification"] == regime, "leverage"].dropna()
        if len(data) > 0:
            sns.kdeplot(data, ax=ax, color=color, label=regime, fill=True, alpha=0.3, linewidth=2)

    ax.set_title("Leverage Distribution by Sentiment Regime", fontweight="bold", pad=12)
    ax.set_xlabel("Leverage (x)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_symbol_performance(
    symbol_df: pd.DataFrame,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Heatmap of mean PnL by symbol and sentiment regime."""
    pivot = symbol_df.pivot_table(
        index="symbol", columns="classification", values="mean_pnl", aggfunc="mean"
    )
    # Sort by overall PnL
    pivot["_total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("_total", ascending=False).drop(columns="_total")

    fig, ax = plt.subplots(figsize=figsize, dpi=config.FIG_DPI)
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
        ax=ax, linewidths=0.5, linecolor="#2D3139",
        cbar_kws={"label": "Mean PnL ($)"},
        annot_kws={"fontsize": 10, "fontweight": "bold"},
    )
    ax.set_title("Mean PnL by Symbol & Sentiment Regime", fontweight="bold", pad=12)
    ax.set_ylabel("Symbol")
    ax.set_xlabel("Sentiment Regime")
    fig.tight_layout()
    return fig


def plot_trade_frequency(
    daily_df: pd.DataFrame,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Trade frequency comparison by regime."""
    fig, ax = plt.subplots(figsize=figsize, dpi=config.FIG_DPI)

    for regime, color in PALETTE.items():
        data = daily_df.loc[daily_df["classification"] == regime, "trade_count"]
        sns.kdeplot(data, ax=ax, color=color, label=regime, fill=True, alpha=0.3, linewidth=2)

    ax.set_title("Daily Trade Frequency by Sentiment Regime", fontweight="bold", pad=12)
    ax.set_xlabel("Number of Trades per Day")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Correlation heatmap of daily trading metrics vs sentiment."""
    if corr_matrix.empty:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    fig, ax = plt.subplots(figsize=figsize, dpi=config.FIG_DPI)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, ax=ax,
        linewidths=0.5, linecolor="#2D3139",
        vmin=-1, vmax=1,
        annot_kws={"fontsize": 9},
    )
    ax.set_title("Correlation Matrix: Sentiment & Trading Metrics", fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def plot_cumulative_pnl(
    daily_df: pd.DataFrame,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Cumulative PnL curve with regime-colored segments."""
    fig, ax = plt.subplots(figsize=figsize, dpi=config.FIG_DPI)

    df = daily_df.sort_values("date").copy()
    df["cum_pnl"] = df["total_pnl"].cumsum()

    # Plot line with regime coloring
    prev_date = None
    prev_pnl = None
    for _, row in df.iterrows():
        if prev_date is not None:
            color = PALETTE.get(row["classification"], "#888")
            ax.plot(
                [prev_date, row["date"]], [prev_pnl, row["cum_pnl"]],
                color=color, linewidth=2, alpha=0.85,
            )
        prev_date = row["date"]
        prev_pnl = row["cum_pnl"]

    ax.fill_between(df["date"], df["cum_pnl"], 0, alpha=0.1, color="#00D4FF")
    ax.axhline(0, color="#FFD700", linestyle="--", alpha=0.4)

    ax.set_title("Cumulative PnL Over Time", fontweight="bold", pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(color=FEAR_COLOR, label="Fear"), Patch(color=GREED_COLOR, label="Greed")],
        loc="upper left",
    )

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
