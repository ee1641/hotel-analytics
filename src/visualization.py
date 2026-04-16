"""
Visualization functions -- all save PNG files to output_dir.
Uses Agg backend (non-interactive) for script/server execution.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


def _ensure_dir(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)


def plot_revenue_over_time(daily_df: pd.DataFrame, output_dir: str = "outputs"):
    """Daily revenue line chart."""
    _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily_df.index, daily_df["daily_revenue"], color="steelblue", linewidth=1)
    ax.set_title("Daily Revenue Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue (EUR)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/revenue_over_time.png", dpi=150)
    plt.close(fig)


def plot_forecast_vs_actual(
    actual: pd.Series,
    forecast: pd.Series,
    title: str,
    output_dir: str = "outputs",
    lower: pd.Series = None,
    upper: pd.Series = None,
):
    """
    Overlay actual and forecast lines on a shared axis.
    If lower/upper are provided, draws a shaded confidence band around the forecast.
    """
    _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(actual.index, actual.values, label="Actual", color="steelblue")
    ax.plot(forecast.index, forecast.values, label="Forecast", color="tomato", linestyle="--")
    if lower is not None and upper is not None:
        ax.fill_between(
            forecast.index, lower.values, upper.values,
            alpha=0.2, color="tomato", label="Confidence Band",
        )
    ax.set_title(f"{title}: Actual vs Forecast")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    safe = title.replace(" ", "_")
    fig.savefig(f"{output_dir}/{safe}_forecast.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(
    model,
    feature_names: list,
    output_dir: str = "outputs",
    top_n: int = 20,
):
    """Horizontal bar chart of top-N feature importances (XGBoost or RandomForest)."""
    _ensure_dir(output_dir)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_feats = [feature_names[i] for i in indices]
    top_imps = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_imps, y=top_feats, ax=ax, hue=top_feats, palette="viridis", legend=False)
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/feature_importance.png", dpi=150)
    plt.close(fig)


def plot_cancellation_roc(
    models: dict,
    X_test,
    y_test,
    output_dir: str = "outputs",
):
    """ROC curves for all cancellation models overlaid on one axes."""
    _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(8, 6))
    _skip = {"stacked_meta", "optimal_threshold", "target_encode_maps", "global_cancel_mean"}
    for name, model in models.items():
        if name in _skip or not hasattr(model, "predict_proba"):
            continue
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_title("Cancellation Prediction -- ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/cancellation_roc.png", dpi=150)
    plt.close(fig)
