"""
Evaluation metrics for time-series forecasting and classification models.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute MAE, RMSE, and MAPE.
    MAPE excludes zero-valued true entries to avoid division by zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mask = y_true != 0
    mape = (
        float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        if mask.any() else float("inf")
    )
    return {"mae": mae, "rmse": rmse, "mape": mape}


def ts_cross_val_scores(
    series: pd.Series,
    n_splits: int = 5,
    forecast_horizon: int = 30,
) -> "pd.DataFrame":
    """
    Expanding-window time-series cross-validation using XGBForecaster.

    Splits series into n_splits folds. Each fold trains on an expanding
    training window and evaluates on the next forecast_horizon days.

    Returns a DataFrame with columns: [fold, train_end, mae, rmse, mape].
    """
    from src.forecasting import train_xgb_forecaster, forecast_xgb  # deferred to avoid circular

    n = len(series)
    min_train = n // (n_splits + 1)
    fold_size = (n - min_train) // n_splits

    rows = []
    for fold in range(n_splits):
        train_end = min_train + fold * fold_size
        if train_end + forecast_horizon > n:
            break
        train = series.iloc[:train_end]
        test = series.iloc[train_end: train_end + forecast_horizon]
        try:
            from src.model_specs import FORECAST_LAGS
            model, feat_cols = train_xgb_forecaster(train, lags=FORECAST_LAGS)
            preds = forecast_xgb(model, feat_cols, train, periods=len(test))
            preds.index = test.index
            m = regression_metrics(test.values, preds.values)
            rows.append({
                "fold": fold + 1,
                "train_end": train.index[-1].date(),
                "mae": round(m["mae"], 2),
                "rmse": round(m["rmse"], 2),
                "mape": round(m["mape"], 2),
            })
        except Exception:
            pass

    return pd.DataFrame(rows)


def forecast_error_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    index: pd.DatetimeIndex,
) -> dict:
    """
    Per-weekday and per-month MAPE breakdown.

    Returns a dict with:
      by_weekday : DataFrame indexed by day name (Mon-Sun) with column 'mape'
      by_month   : DataFrame indexed by month number (1-12) with column 'mape'
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    index = pd.DatetimeIndex(index)

    mask = y_true != 0
    ape = np.where(mask, np.abs((y_true - y_pred) / y_true) * 100, np.nan)

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_weekday = (
        pd.DataFrame({"ape": ape, "dow": index.dayofweek})
        .groupby("dow")["ape"]
        .mean()
        .rename(index=dict(enumerate(day_names)))
        .rename("mape")
        .to_frame()
    )

    by_month = (
        pd.DataFrame({"ape": ape, "month": index.month})
        .groupby("month")["ape"]
        .mean()
        .rename("mape")
        .to_frame()
    )

    return {"by_weekday": by_weekday, "by_month": by_month}


def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    Compute Accuracy, Precision, Recall, F1, and ROC-AUC.
    """
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, y_prob)),
    }
