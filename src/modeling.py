"""
Optional modeling module.

Provides a simple, interpretable model to predict trade outcome (win/loss)
based on sentiment features and trader behavior. Includes a rule-based
scoring approach alongside a lightweight ML model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Rule-Based Scoring ────────────────────────────────────────────────────────

def rule_based_score(row: pd.Series) -> float:
    """Score a trade's likelihood of being profitable (0–100).

    Rules are derived from exploratory analysis:
      - Greed regime → higher base score
      - Moderate leverage → bonus
      - Side aligned with sentiment → bonus
      - Small position size (relative) → bonus
    """
    score = 50.0  # neutral baseline

    # Sentiment regime
    if row.get("classification") == "Greed":
        score += 12
    else:
        score -= 8

    # Leverage penalty for extremes
    leverage = row.get("leverage", 5)
    if leverage <= 5:
        score += 8
    elif leverage <= 10:
        score += 3
    elif leverage >= 25:
        score -= 10
    elif leverage >= 50:
        score -= 18

    # Side alignment with sentiment
    side = row.get("side", "")
    classification = row.get("classification", "")
    if (side == "Buy" and classification == "Greed") or \
       (side == "Sell" and classification == "Fear"):
        score += 7
    else:
        score -= 5

    # Sentiment streak (longer streaks = more predictable)
    streak = row.get("sentiment_streak", 1)
    if streak >= 5:
        score += 5
    elif streak <= 2:
        score -= 3

    return max(0, min(100, score))


def apply_rule_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Apply rule-based scoring to all trades."""
    df = df.copy()
    df["risk_score"] = df.apply(rule_based_score, axis=1)
    logger.info(
        "Rule scores applied: mean={:.1f}, std={:.1f}",
        df["risk_score"].mean(), df["risk_score"].std(),
    )
    return df


# ── ML Model ─────────────────────────────────────────────────────────────────

FEATURE_COLS: list[str] = [
    "is_fear", "is_greed", "leverage", "log_size",
    "is_long", "is_short", "sentiment_streak",
]


def prepare_model_data(
    df: pd.DataFrame,
    target: str = "is_winner",
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Prepare features and target for modeling.

    Returns (X, y, used_features).
    """
    features = features or FEATURE_COLS
    available = [f for f in features if f in df.columns]

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")
    if len(available) < 2:
        raise ValueError(f"Insufficient features available: {available}")

    model_df = df[available + [target]].dropna()
    X = model_df[available]
    y = model_df[target]

    logger.info(
        "Model data: {} samples, {} features, {:.1%} positive rate",
        len(X), len(available), y.mean(),
    )
    return X, y, available


def train_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train a Gradient Boosting classifier to predict trade outcome.

    Returns
    -------
    dict with keys: model, scaler, features, metrics, feature_importance
    """
    X, y, features = prepare_model_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Gradient Boosting — good balance of interpretability and performance
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=random_state,
        subsample=0.8,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluation
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Feature importance
    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    logger.info("Model trained — Accuracy: {:.3f}, AUC: {:.3f}", accuracy, roc_auc)

    return {
        "model": model,
        "scaler": scaler,
        "features": features,
        "metrics": {
            "accuracy": round(accuracy, 4),
            "roc_auc": round(roc_auc, 4),
            "classification_report": report,
            "train_size": len(X_train),
            "test_size": len(X_test),
        },
        "feature_importance": importance,
    }


def get_model_summary(result: dict) -> str:
    """Generate a markdown-formatted model summary."""
    m = result["metrics"]
    imp = result["feature_importance"]

    lines = [
        "## Model Summary",
        "",
        f"**Algorithm:** Gradient Boosting Classifier",
        f"**Train / Test Split:** {m['train_size']} / {m['test_size']}",
        f"**Accuracy:** {m['accuracy']:.1%}",
        f"**ROC AUC:** {m['roc_auc']:.3f}",
        "",
        "### Feature Importance",
        "",
    ]
    for _, row in imp.iterrows():
        bar = "█" * int(row["importance"] * 50)
        lines.append(f"- **{row['feature']}**: {row['importance']:.3f} {bar}")

    lines.extend([
        "",
        "### Limitations",
        "",
        "- This is a simplified model for exploratory purposes.",
        "- It uses only sentiment and basic trade features, not order-book or price dynamics.",
        "- Past regime patterns may not predict future outcomes.",
        "- The model should not be used for live trading decisions without extensive validation.",
    ])

    return "\n".join(lines)
