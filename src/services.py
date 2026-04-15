"""
Shared data loading and model training logic.
Importable by both the Streamlit dashboard and the FastAPI backend.
"""
import logging
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src import model_cache
from src.cancellation import (
    compute_scale_pos_weight,
    get_shap_feature_importance,
    select_top_features,
    train_cancellation_models_balanced,
    train_cancellation_survival_model,
    tune_xgb_cancellation,
)
from src.cancellation_evaluation import evaluate_cross_dataset
from src.config import APP_CONFIG
from src.evaluation import regression_metrics, ts_cross_val_scores
from src.forecasting import (
    forecast_prophet,
    forecast_xgb,
    forecast_xgb_direct,
    naive_baseline_forecast,
    train_prophet_forecaster,
    train_xgb_direct_multistep,
    train_xgb_forecaster,
    train_xgb_forecaster_with_intervals,
)
from src.ingestion import load_cancellation_prediction_csv
from src.model_specs import (
    CANCELLATION_MODEL_VERSION,
    DIRECT_FORECAST_MODEL_VERSION,
    EVAL_METRICS_VERSION,
    FORECAST_LAGS,
    FORECAST_MODEL_VERSION,
    INTERVAL_MODEL_VERSION,
    PROPHET_MODEL_VERSION,
    SURVIVAL_MODEL_VERSION,
)
from src.preprocessing import (
    _CATEGORICAL_COLS,
    add_temporal_features,
    clean_data,
    drop_leaky_features,
    encode_categoricals,
    engineer_features,
    load_data,
)
from src.time_series import aggregate_daily, build_arrival_date

DATA_PATH = "hotel_bookings.csv"
log = logging.getLogger(__name__)


def _artifact_current(name: str, version: str, models_dir: Path) -> bool:
    return model_cache.artifact_matches_version(name, version, models_dir=models_dir)


def _save_versioned(name: str, obj, version: str, models_dir: Path) -> None:
    model_cache.save_artifact(name, obj, metadata={"version": version}, models_dir=models_dir)


def load_and_prepare_data(data_path: str = DATA_PATH, *, dataset_context=None):
    """
    Full preprocessing pipeline.

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
    """
    models_dir = model_cache.MODELS_DIR
    if dataset_context is not None:
        data_path = dataset_context.data_path
        models_dir = Path(dataset_context.models_dir)

    df, daily_df = load_and_prepare_data(data_path)
    results = {}

    rev_series = daily_df["daily_revenue"]
    dem_series = daily_df["daily_bookings"]

    # ------------------------------------------------------------------
    # Revenue forecasting
    # ------------------------------------------------------------------
    if _artifact_current("xgb_revenue", FORECAST_MODEL_VERSION, models_dir):
        results["xgb_revenue"] = model_cache.load_artifact("xgb_revenue", models_dir=models_dir)
    else:
        xgb_m, xgb_cols = train_xgb_forecaster(rev_series, lags=FORECAST_LAGS)
        results["xgb_revenue"] = (xgb_m, xgb_cols)
        _save_versioned("xgb_revenue", (xgb_m, xgb_cols), FORECAST_MODEL_VERSION, models_dir)

    if _artifact_current("xgb_revenue_direct", DIRECT_FORECAST_MODEL_VERSION, models_dir):
        results["xgb_revenue_direct"] = model_cache.load_artifact("xgb_revenue_direct", models_dir=models_dir)
    else:
        direct_rev = train_xgb_direct_multistep(rev_series, lags=FORECAST_LAGS)
        results["xgb_revenue_direct"] = direct_rev
        _save_versioned("xgb_revenue_direct", direct_rev, DIRECT_FORECAST_MODEL_VERSION, models_dir)

    if _artifact_current("xgb_revenue_intervals", INTERVAL_MODEL_VERSION, models_dir):
        results["xgb_revenue_intervals"] = model_cache.load_artifact("xgb_revenue_intervals", models_dir=models_dir)
    else:
        interval_models = train_xgb_forecaster_with_intervals(rev_series, lags=FORECAST_LAGS)
        results["xgb_revenue_intervals"] = interval_models
        _save_versioned("xgb_revenue_intervals", interval_models, INTERVAL_MODEL_VERSION, models_dir)

    if _artifact_current("prophet_revenue", PROPHET_MODEL_VERSION, models_dir):
        results["prophet_revenue"] = model_cache.load_artifact("prophet_revenue", models_dir=models_dir)
    else:
        prophet_rev = train_prophet_forecaster(rev_series)
        results["prophet_revenue"] = prophet_rev
        _save_versioned("prophet_revenue", prophet_rev, PROPHET_MODEL_VERSION, models_dir)

    # ------------------------------------------------------------------
    # Demand forecasting
    # ------------------------------------------------------------------
    if _artifact_current("xgb_demand", FORECAST_MODEL_VERSION, models_dir):
        results["xgb_demand"] = model_cache.load_artifact("xgb_demand", models_dir=models_dir)
    else:
        xgb_d, xgb_d_cols = train_xgb_forecaster(dem_series, lags=FORECAST_LAGS)
        results["xgb_demand"] = (xgb_d, xgb_d_cols)
        _save_versioned("xgb_demand", (xgb_d, xgb_d_cols), FORECAST_MODEL_VERSION, models_dir)

    if _artifact_current("xgb_demand_direct", DIRECT_FORECAST_MODEL_VERSION, models_dir):
        results["xgb_demand_direct"] = model_cache.load_artifact("xgb_demand_direct", models_dir=models_dir)
    else:
        direct_dem = train_xgb_direct_multistep(dem_series, lags=FORECAST_LAGS)
        results["xgb_demand_direct"] = direct_dem
        _save_versioned("xgb_demand_direct", direct_dem, DIRECT_FORECAST_MODEL_VERSION, models_dir)

    if _artifact_current("xgb_demand_intervals", INTERVAL_MODEL_VERSION, models_dir):
        results["xgb_demand_intervals"] = model_cache.load_artifact("xgb_demand_intervals", models_dir=models_dir)
    else:
        interval_models = train_xgb_forecaster_with_intervals(dem_series, lags=FORECAST_LAGS)
        results["xgb_demand_intervals"] = interval_models
        _save_versioned("xgb_demand_intervals", interval_models, INTERVAL_MODEL_VERSION, models_dir)

    if _artifact_current("prophet_demand", PROPHET_MODEL_VERSION, models_dir):
        results["prophet_demand"] = model_cache.load_artifact("prophet_demand", models_dir=models_dir)
    else:
        prophet_dem = train_prophet_forecaster(dem_series)
        results["prophet_demand"] = prophet_dem
        _save_versioned("prophet_demand", prophet_dem, PROPHET_MODEL_VERSION, models_dir)

    # ------------------------------------------------------------------
    # Cancellation classification
    # ------------------------------------------------------------------
    if _artifact_current("cancel_models", CANCELLATION_MODEL_VERSION, models_dir):
        results["cancel_models"] = model_cache.load_artifact("cancel_models", models_dir=models_dir)
        results["feature_columns"] = model_cache.load_artifact("feature_columns", models_dir=models_dir)
        results["label_encoders"] = model_cache.load_artifact("label_encoders", models_dir=models_dir)
        results["feature_defaults"] = model_cache.load_artifact("feature_defaults", models_dir=models_dir)
        results["selected_features"] = model_cache.load_artifact("selected_features", models_dir=models_dir) or []
    else:
        # Sort by arrival_date for time-based split — prevents future booking
        # info leaking into the cancellation model's training data.
        df_sorted = df.sort_values("arrival_date")
        df_cancel = drop_leaky_features(df_sorted)
        label_encoders = {}
        for col in _CATEGORICAL_COLS:
            if col in df_cancel.columns:
                le = LabelEncoder()
                df_cancel = df_cancel.copy()
                df_cancel[col] = le.fit_transform(df_cancel[col].astype(str))
                label_encoders[col] = le

        X, y_cancel, _ = encode_categoricals(drop_leaky_features(df_sorted))
        feature_columns = list(X.columns)

        feature_defaults = {}
        for col in feature_columns:
            col_data = X[col].dropna()
            if col_data.dtype == float or col_data.nunique() > 20:
                feature_defaults[col] = float(col_data.median())
            else:
                feature_defaults[col] = float(col_data.mode().iloc[0])

        # Time-based 80/20 split (last 20% of sorted dates as test)
        n_test = max(1, int(len(X) * 0.2))
        X_tr = X.iloc[:-n_test]
        y_tr = y_cancel.iloc[:-n_test]
        X_te = X.iloc[-n_test:]

        spw = compute_scale_pos_weight(y_tr.values)
        best_xgb_params = tune_xgb_cancellation(X_tr.values, y_tr.values, scale_pos_weight=spw)
        cancel_models = train_cancellation_models_balanced(
            X_tr.values,
            y_tr.values,
            scale_pos_weight=spw,
            xgb_params=best_xgb_params,
        )

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        shap_df = get_shap_feature_importance(
            cancel_models["xgboost"],
            X_te.values,
            feature_columns,
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
        _save_versioned("cancel_models", cancel_models, CANCELLATION_MODEL_VERSION, models_dir)
        model_cache.save_artifact("feature_columns", feature_columns, models_dir=models_dir)
        model_cache.save_artifact("label_encoders", label_encoders, models_dir=models_dir)
        model_cache.save_artifact("feature_defaults", feature_defaults, models_dir=models_dir)
        model_cache.save_artifact("selected_features", selected_features, models_dir=models_dir)

    # ------------------------------------------------------------------
    # Cancellation survival
    # ------------------------------------------------------------------
    if _artifact_current("cancel_survival_model", SURVIVAL_MODEL_VERSION, models_dir):
        results["cancel_survival_model"] = model_cache.load_artifact("cancel_survival_model", models_dir=models_dir)
    else:
        survival_bundle = train_cancellation_survival_model(df)
        results["cancel_survival_model"] = survival_bundle
        _save_versioned("cancel_survival_model", survival_bundle, SURVIVAL_MODEL_VERSION, models_dir)

    # ------------------------------------------------------------------
    # Evaluation metrics
    # ------------------------------------------------------------------
    if _artifact_current("eval_metrics", EVAL_METRICS_VERSION, models_dir):
        results["eval_metrics"] = model_cache.load_artifact("eval_metrics", models_dir=models_dir)
    else:
        test_days = 30

        rev_eval_train = rev_series.iloc[:-test_days]
        rev_eval_test = rev_series.iloc[-test_days:]
        rev_eval_xgb_m, rev_eval_xgb_c = train_xgb_forecaster(rev_eval_train, lags=FORECAST_LAGS)
        rev_xgb_preds = forecast_xgb(rev_eval_xgb_m, rev_eval_xgb_c, rev_eval_train, test_days)
        rev_xgb_preds.index = rev_eval_test.index
        rev_naive = naive_baseline_forecast(rev_eval_train, test_days)
        rev_naive.index = rev_eval_test.index
        rev_eval_prophet = train_prophet_forecaster(rev_eval_train)
        rev_prophet_df = forecast_prophet(rev_eval_prophet, test_days)

        dem_eval_train = dem_series.iloc[:-test_days]
        dem_eval_test = dem_series.iloc[-test_days:]
        dem_eval_xgb_m, dem_eval_xgb_c = train_xgb_forecaster(dem_eval_train, lags=FORECAST_LAGS)
        dem_xgb_preds = forecast_xgb(dem_eval_xgb_m, dem_eval_xgb_c, dem_eval_train, test_days)
        dem_xgb_preds.index = dem_eval_test.index
        dem_naive = naive_baseline_forecast(dem_eval_train, test_days)
        dem_naive.index = dem_eval_test.index
        dem_eval_prophet = train_prophet_forecaster(dem_eval_train)
        dem_prophet_df = forecast_prophet(dem_eval_prophet, test_days)

        eval_metrics = {
            "revenue_xgb": regression_metrics(rev_eval_test.values, rev_xgb_preds.values),
            "revenue_prophet": regression_metrics(rev_eval_test.values, rev_prophet_df["yhat"].values),
            "revenue_naive": regression_metrics(rev_eval_test.values, rev_naive.values),
            "demand_xgb": regression_metrics(dem_eval_test.values, dem_xgb_preds.values),
            "demand_prophet": regression_metrics(dem_eval_test.values, dem_prophet_df["yhat"].values),
            "demand_naive": regression_metrics(dem_eval_test.values, dem_naive.values),
            "eval_note": (
                f"Holdout: last {test_days} days (unseen during training). "
                "Production models are retrained on the full dataset for live forecasts."
            ),
            "cv_scores": ts_cross_val_scores(rev_series, n_splits=4, forecast_horizon=30),
        }

        try:
            second_df = load_cancellation_prediction_csv("cancellation_Prediction.csv")
            eval_metrics["cross_dataset"] = evaluate_cross_dataset(
                results["cancel_models"],
                results["feature_columns"],
                results["label_encoders"],
                results["feature_defaults"],
                second_df,
            )
        except Exception as exc:
            log.warning("Cross-dataset evaluation failed: %s", exc)
            eval_metrics["cross_dataset"] = {}

        results["eval_metrics"] = eval_metrics
        _save_versioned("eval_metrics", eval_metrics, EVAL_METRICS_VERSION, models_dir)

    return results
