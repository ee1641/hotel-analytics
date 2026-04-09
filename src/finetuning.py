"""
Model fine-tuning pipeline.
Allows retraining on new hotel data with optional warm-starting for XGBoost.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import drop_leaky_features, encode_categoricals
from src.cancellation import (
    train_cancellation_models_balanced,
    compute_scale_pos_weight,
    predict_cancellation_proba,
)
from src.evaluation import classification_report_dict, regression_metrics


def finetune_cancellation_model(
    new_df: pd.DataFrame,
    base_model=None,
) -> tuple:
    """
    Train cancellation models on new_df.
    If base_model (an XGBClassifier) is provided, warm-start using xgb_model_init.

    Returns:
        (models_dict, metrics_dict)
        models_dict : {"logistic": ..., "random_forest": ..., "xgboost": ...}
        metrics_dict: {"logistic": {...}, "random_forest": {...}, "xgboost": {...}}
    """
    df_clean = drop_leaky_features(new_df)
    X, y, _ = encode_categoricals(df_clean)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X.values, y.values, test_size=0.2, random_state=42, stratify=y.values
    )
    spw = compute_scale_pos_weight(y_tr)

    if base_model is not None:
        # XGBoost warm-start: continue training from the base model
        from xgboost import XGBClassifier
        xgb_init = XGBClassifier(
            n_estimators=50,
            max_depth=6,
            learning_rate=0.05,
            eval_metric="logloss",
            verbosity=0,
            random_state=42,
            scale_pos_weight=spw,
            xgb_model=base_model,
        )
        xgb_init.fit(X_tr, y_tr)
        # Replace xgboost with warm-started version after balanced training
        models = train_cancellation_models_balanced(X_tr, y_tr, scale_pos_weight=spw)
        models["xgboost"] = xgb_init
    else:
        models = train_cancellation_models_balanced(X_tr, y_tr, scale_pos_weight=spw)

    metrics = {}
    for name, model in models.items():
        y_pred = model.predict(X_te)
        y_prob = predict_cancellation_proba(model, X_te)
        metrics[name] = classification_report_dict(y_te, y_pred, y_prob)

    return models, metrics


def finetune_forecast_model(
    new_daily_series: pd.Series,
    model_type: str = "prophet",
) -> tuple:
    """
    Retrain a forecasting model on a new daily time series.

    Args:
        new_daily_series : pd.Series with DatetimeIndex
        model_type       : "prophet" or "xgboost"

    Returns:
        (model, metrics_dict) where metrics are computed on last 20% as test set.
    """
    split = int(len(new_daily_series) * 0.8)
    train = new_daily_series.iloc[:split]
    test  = new_daily_series.iloc[split:]

    if model_type == "prophet":
        from src.forecasting import train_prophet_forecaster, forecast_prophet
        model = train_prophet_forecaster(train)
        pred_df = forecast_prophet(model, periods=len(test))
        preds = pred_df["yhat"].values
    else:
        from src.forecasting import train_xgb_forecaster, forecast_xgb
        model, feat_cols = train_xgb_forecaster(train)
        preds_series = forecast_xgb(model, feat_cols, train, periods=len(test))
        preds = preds_series.values
        model = (model, feat_cols)

    metrics = regression_metrics(test.values, preds)
    return model, metrics


def evaluate_before_deploy(
    new_model,
    test_df: pd.DataFrame,
    metric_key: str = "roc_auc",
    min_threshold: float = 0.70,
) -> dict:
    """
    Evaluate a cancellation model on test_df.
    Returns a metrics dict. Caller decides whether to deploy
    based on whether metrics[metric_key] exceeds min_threshold.

    test_df must contain the target column 'is_canceled'.
    """
    df_clean = drop_leaky_features(test_df)
    X, y, _ = encode_categoricals(df_clean)

    y_pred = new_model.predict(X.values)
    y_prob = predict_cancellation_proba(new_model, X.values)
    metrics = classification_report_dict(y.values, y_pred, y_prob)
    metrics["meets_threshold"] = metrics.get(metric_key, 0) >= min_threshold
    return metrics
