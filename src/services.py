"""
Shared data loading and model training logic.
Importable by both the Streamlit dashboard (app.py) and the FastAPI backend (api/).
No Streamlit dependencies here.
"""
import logging
import os
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.preprocessing import (
    load_data, clean_data, engineer_features, encode_categoricals,
    add_temporal_features, drop_leaky_features, _CATEGORICAL_COLS,
)
from src.time_series import build_arrival_date, aggregate_daily
from src.forecasting import (
    train_xgb_forecaster, forecast_xgb,
    naive_baseline_forecast,
    train_prophet_forecaster, forecast_prophet,
)
from src.cancellation import (
    train_cancellation_models_balanced, compute_scale_pos_weight,
    get_shap_feature_importance, select_top_features,
)
from src.evaluation import regression_metrics, ts_cross_val_scores
from src import model_cache
from src.ingestion import load_cancellation_prediction_csv
from src.cancellation_evaluation import evaluate_cross_dataset

DATA_PATH = "hotel_bookings.csv"
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurable constants
# Adjust these to match your hotel's region, tax regime, and operations.
# ---------------------------------------------------------------------------
APP_CONFIG = {
    "tax_rate": 0.10,                # VAT / local tax rate applied to revenue
    "cost_ratio": 0.60,              # Operational costs as fraction of revenue
    "guests_per_staff_default": 10,  # Default guests per staff member
    "timezone": "Europe/Lisbon",     # Hotel timezone (IANA tz string)
    "currency_symbol": "€",
}


def load_and_prepare_data(data_path: str = DATA_PATH, *, dataset_context=None):
    """
    Full preprocessing pipeline (no Streamlit caching — callers add caching as needed).

    Returns: (df, daily_df)
      df       : booking-level DataFrame with all engineered columns
      daily_df : daily-aggregated DataFrame (revenue, bookings, cancellations)
    """
    if dataset_context is not None:
        data_path = dataset_context.data_path
    df = engineer_features(clean_data(load_data(data_path)))
    df = build_arrival_date(df)
    df = add_temporal_features(df)
    daily_df = aggregate_daily(df)
    return df, daily_df


def get_or_train_models(data_path: str = DATA_PATH, *, dataset_context=None):
    """
    Load cached ML artifacts from disk or train them fresh.
    No Streamlit progress indicators — uses logging instead.

    Returns a dict with keys:
      xgb_revenue, prophet_revenue, xgb_demand, prophet_demand,
      cancel_models, feature_columns, label_encoders,
      feature_defaults, selected_features, eval_metrics
    """
    _models_dir = model_cache.MODELS_DIR
    if dataset_context is not None:
        data_path = dataset_context.data_path
        _models_dir = Path(dataset_context.models_dir)

    df, daily_df = load_and_prepare_data(data_path)
    results = {}

    rev_series = daily_df["daily_revenue"]
    dem_series = daily_df["daily_bookings"]

    # ------------------------------------------------------------------
    # Revenue forecasting (production models — trained on full data)
    # ------------------------------------------------------------------
    if model_cache.artifact_exists("xgb_revenue", models_dir=_models_dir):
        results["xgb_revenue"] = model_cache.load_artifact("xgb_revenue", models_dir=_models_dir)
    else:
        log.info("Training XGBoost revenue forecaster...")
        xgb_m, xgb_cols = train_xgb_forecaster(rev_series, lags=[1, 7, 14, 30])
        results["xgb_revenue"] = (xgb_m, xgb_cols)
        model_cache.save_artifact("xgb_revenue", (xgb_m, xgb_cols), models_dir=_models_dir)

    if model_cache.artifact_exists("prophet_revenue", models_dir=_models_dir):
        results["prophet_revenue"] = model_cache.load_artifact("prophet_revenue", models_dir=_models_dir)
    else:
        log.info("Training Prophet revenue forecaster...")
        prophet_rev = train_prophet_forecaster(rev_series)
        results["prophet_revenue"] = prophet_rev
        model_cache.save_artifact("prophet_revenue", prophet_rev, models_dir=_models_dir)

    # ------------------------------------------------------------------
    # Demand forecasting (production models — trained on full data)
    # ------------------------------------------------------------------
    if model_cache.artifact_exists("xgb_demand", models_dir=_models_dir):
        results["xgb_demand"] = model_cache.load_artifact("xgb_demand", models_dir=_models_dir)
    else:
        log.info("Training XGBoost demand forecaster...")
        xgb_d, xgb_d_cols = train_xgb_forecaster(dem_series, lags=[1, 7, 14, 30])
        results["xgb_demand"] = (xgb_d, xgb_d_cols)
        model_cache.save_artifact("xgb_demand", (xgb_d, xgb_d_cols), models_dir=_models_dir)

    if model_cache.artifact_exists("prophet_demand", models_dir=_models_dir):
        results["prophet_demand"] = model_cache.load_artifact("prophet_demand", models_dir=_models_dir)
    else:
        log.info("Training Prophet demand forecaster...")
        prophet_dem = train_prophet_forecaster(dem_series)
        results["prophet_demand"] = prophet_dem
        model_cache.save_artifact("prophet_demand", prophet_dem, models_dir=_models_dir)

    # ------------------------------------------------------------------
    # Cancellation classification
    # ------------------------------------------------------------------
    if model_cache.artifact_exists("cancel_models", models_dir=_models_dir):
        results["cancel_models"] = model_cache.load_artifact("cancel_models", models_dir=_models_dir)
        results["feature_columns"] = model_cache.load_artifact("feature_columns", models_dir=_models_dir)
        results["label_encoders"] = model_cache.load_artifact("label_encoders", models_dir=_models_dir)
        results["feature_defaults"] = model_cache.load_artifact("feature_defaults", models_dir=_models_dir)
        results["selected_features"] = model_cache.load_artifact("selected_features", models_dir=_models_dir) or []
    else:
        log.info("Training cancellation classifiers...")
        df_cancel = drop_leaky_features(df)
        label_encoders = {}
        for col in _CATEGORICAL_COLS:
            if col in df_cancel.columns:
                le = LabelEncoder()
                df_cancel = df_cancel.copy()
                df_cancel[col] = le.fit_transform(df_cancel[col].astype(str))
                label_encoders[col] = le

        X, y_cancel, _ = encode_categoricals(drop_leaky_features(df))
        feature_columns = list(X.columns)

        feature_defaults = {}
        for col in feature_columns:
            col_data = X[col].dropna()
            if col_data.dtype == float or col_data.nunique() > 20:
                feature_defaults[col] = float(col_data.median())
            else:
                feature_defaults[col] = float(col_data.mode().iloc[0])

        X_tr, _, y_tr, _ = train_test_split(
            X, y_cancel, test_size=0.2, random_state=42, stratify=y_cancel
        )
        spw = compute_scale_pos_weight(y_tr.values)
        cancel_models = train_cancellation_models_balanced(
            X_tr.values, y_tr.values, scale_pos_weight=spw
        )

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        X_tr2, X_te, _, _ = train_test_split(
            X, y_cancel, test_size=0.2, random_state=42, stratify=y_cancel
        )
        shap_df = get_shap_feature_importance(
            cancel_models["xgboost"], X_te.values, feature_columns
        )
        selected_features = select_top_features(shap_df, threshold=0.01)
        os.makedirs("outputs", exist_ok=True)
        top_shap = shap_df.nlargest(15, "mean_abs_shap")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top_shap["feature"], top_shap["mean_abs_shap"])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Feature Importance (SHAP)")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig("outputs/shap_summary.png", dpi=120, bbox_inches="tight")
        plt.close()

        results["cancel_models"] = cancel_models
        results["feature_columns"] = feature_columns
        results["label_encoders"] = label_encoders
        results["feature_defaults"] = feature_defaults
        results["selected_features"] = selected_features
        model_cache.save_artifact("cancel_models", cancel_models, models_dir=_models_dir)
        model_cache.save_artifact("feature_columns", feature_columns, models_dir=_models_dir)
        model_cache.save_artifact("label_encoders", label_encoders, models_dir=_models_dir)
        model_cache.save_artifact("feature_defaults", feature_defaults, models_dir=_models_dir)
        model_cache.save_artifact("selected_features", selected_features, models_dir=_models_dir)

    # ------------------------------------------------------------------
    # Evaluation metrics — proper temporal holdout (1C fix)
    #
    # Evaluation models are trained on first (N-30) days of data.
    # Metrics are reported on the true last 30 days (never seen in training).
    # Production models above remain trained on all data for live forecasting.
    # ------------------------------------------------------------------
    if model_cache.artifact_exists("eval_metrics", models_dir=_models_dir):
        results["eval_metrics"] = model_cache.load_artifact("eval_metrics", models_dir=_models_dir)
    else:
        log.info("Computing evaluation metrics with temporal holdout (last 30 days unseen)...")
        TEST_DAYS = 30

        # Revenue holdout — eval model trained on all but last 30 days
        rev_eval_train = rev_series.iloc[:-TEST_DAYS]
        rev_eval_test = rev_series.iloc[-TEST_DAYS:]

        rev_eval_xgb_m, rev_eval_xgb_c = train_xgb_forecaster(
            rev_eval_train, lags=[1, 7, 14, 30]
        )
        rev_xgb_preds = forecast_xgb(rev_eval_xgb_m, rev_eval_xgb_c, rev_eval_train, TEST_DAYS)
        rev_xgb_preds.index = rev_eval_test.index

        rev_naive = naive_baseline_forecast(rev_eval_train, TEST_DAYS)
        rev_naive.index = rev_eval_test.index

        rev_eval_prophet = train_prophet_forecaster(rev_eval_train)
        rev_prophet_df = forecast_prophet(rev_eval_prophet, TEST_DAYS)

        # Demand holdout — eval model trained on all but last 30 days
        dem_eval_train = dem_series.iloc[:-TEST_DAYS]
        dem_eval_test = dem_series.iloc[-TEST_DAYS:]

        dem_eval_xgb_m, dem_eval_xgb_c = train_xgb_forecaster(
            dem_eval_train, lags=[1, 7, 14, 30]
        )
        dem_xgb_preds = forecast_xgb(dem_eval_xgb_m, dem_eval_xgb_c, dem_eval_train, TEST_DAYS)
        dem_xgb_preds.index = dem_eval_test.index

        dem_naive = naive_baseline_forecast(dem_eval_train, TEST_DAYS)
        dem_naive.index = dem_eval_test.index

        dem_eval_prophet = train_prophet_forecaster(dem_eval_train)
        dem_prophet_df = forecast_prophet(dem_eval_prophet, TEST_DAYS)

        eval_metrics = {
            "revenue_xgb":     regression_metrics(rev_eval_test.values, rev_xgb_preds.values),
            "revenue_prophet": regression_metrics(rev_eval_test.values, rev_prophet_df["yhat"].values),
            "revenue_naive":   regression_metrics(rev_eval_test.values, rev_naive.values),
            "demand_xgb":      regression_metrics(dem_eval_test.values, dem_xgb_preds.values),
            "demand_prophet":  regression_metrics(dem_eval_test.values, dem_prophet_df["yhat"].values),
            "demand_naive":    regression_metrics(dem_eval_test.values, dem_naive.values),
            "eval_note": (
                f"Holdout: last {TEST_DAYS} days (unseen during training). "
                "Production models are retrained on the full dataset for live forecasts."
            ),
        }
        cv_scores = ts_cross_val_scores(rev_series, n_splits=4, forecast_horizon=30)
        eval_metrics["cv_scores"] = cv_scores

        try:
            second_df = load_cancellation_prediction_csv("cancellation_Prediction.csv")
            cross_results = evaluate_cross_dataset(
                results["cancel_models"],
                results["feature_columns"],
                results["label_encoders"],
                results["feature_defaults"],
                second_df,
            )
            eval_metrics["cross_dataset"] = cross_results
        except Exception:
            eval_metrics["cross_dataset"] = {}

        results["eval_metrics"] = eval_metrics
        model_cache.save_artifact("eval_metrics", eval_metrics, models_dir=_models_dir)

    return results
