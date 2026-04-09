"""
Hotel Analytics Dashboard — Streamlit entry point.
Run with: streamlit run app.py

Pages are in the pages/ directory (auto-discovered by Streamlit).
This file defines the shared cached loaders used by all pages.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.preprocessing import (
    load_data, clean_data, engineer_features, encode_categoricals,
    add_temporal_features, drop_leaky_features, _CATEGORICAL_COLS,
)
from src.time_series import build_arrival_date, aggregate_daily
from src.forecasting import (
    train_xgb_forecaster, forecast_xgb,
    train_sarima_forecaster, forecast_sarima,
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

import os
DATA_PATH = "hotel_bookings.csv"
APP_MODE = os.getenv("APP_MODE", "demo")


# ---------------------------------------------------------------------------
# Shared cached loaders — imported by every page
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading and preprocessing data...")
def load_and_prepare_data(data_path: str = DATA_PATH):
    """
    Full preprocessing pipeline:
      load_data → clean_data → engineer_features → build_arrival_date
      → add_temporal_features → aggregate_daily

    Returns: (df, daily_df)
      df       : booking-level DataFrame with all engineered columns
      daily_df : daily-aggregated DataFrame (revenue, bookings, cancellations)
    """
    df = engineer_features(clean_data(load_data(data_path)))
    df = build_arrival_date(df)
    df = add_temporal_features(df)
    daily_df = aggregate_daily(df)
    return df, daily_df


@st.cache_resource(show_spinner="Initializing models (first run trains everything — ~5 min)...")
def get_or_train_models(data_path: str = DATA_PATH):
    """
    Granular per-artifact check:  load from disk if present, otherwise train and save.
    SARIMA training (~2 min each) is skipped on subsequent runs via joblib persistence.

    Returns a dict with keys:
      xgb_revenue, sarima_revenue, xgb_demand, sarima_demand,
      cancel_models, feature_columns, label_encoders,
      feature_defaults, eval_metrics
    """
    df, daily_df = load_and_prepare_data(data_path)
    results = {}

    # ------------------------------------------------------------------ #
    # Revenue forecasting
    # ------------------------------------------------------------------ #
    rev_series = daily_df["daily_revenue"]

    if model_cache.artifact_exists("xgb_revenue"):
        results["xgb_revenue"] = model_cache.load_artifact("xgb_revenue")
    else:
        with st.status("Training XGBoost revenue forecaster..."):
            xgb_m, xgb_cols = train_xgb_forecaster(rev_series, lags=[1, 7, 14, 30])
        results["xgb_revenue"] = (xgb_m, xgb_cols)
        model_cache.save_artifact("xgb_revenue", (xgb_m, xgb_cols))

    if model_cache.artifact_exists("prophet_revenue"):
        results["prophet_revenue"] = model_cache.load_artifact("prophet_revenue")
    else:
        with st.status("Training Prophet revenue forecaster..."):
            prophet_rev = train_prophet_forecaster(rev_series)
        results["prophet_revenue"] = prophet_rev
        model_cache.save_artifact("prophet_revenue", prophet_rev)

    # ------------------------------------------------------------------ #
    # Demand forecasting
    # ------------------------------------------------------------------ #
    dem_series = daily_df["daily_bookings"]

    if model_cache.artifact_exists("xgb_demand"):
        results["xgb_demand"] = model_cache.load_artifact("xgb_demand")
    else:
        with st.status("Training XGBoost demand forecaster..."):
            xgb_d, xgb_d_cols = train_xgb_forecaster(dem_series, lags=[1, 7, 14, 30])
        results["xgb_demand"] = (xgb_d, xgb_d_cols)
        model_cache.save_artifact("xgb_demand", (xgb_d, xgb_d_cols))

    if model_cache.artifact_exists("prophet_demand"):
        results["prophet_demand"] = model_cache.load_artifact("prophet_demand")
    else:
        with st.status("Training Prophet demand forecaster..."):
            prophet_dem = train_prophet_forecaster(dem_series)
        results["prophet_demand"] = prophet_dem
        model_cache.save_artifact("prophet_demand", prophet_dem)

    # ------------------------------------------------------------------ #
    # Cancellation classification
    # ------------------------------------------------------------------ #
    if model_cache.artifact_exists("cancel_models"):
        results["cancel_models"] = model_cache.load_artifact("cancel_models")
        results["feature_columns"] = model_cache.load_artifact("feature_columns")
        results["label_encoders"] = model_cache.load_artifact("label_encoders")
        results["feature_defaults"] = model_cache.load_artifact("feature_defaults")
        results["selected_features"] = model_cache.load_artifact("selected_features") or []
    else:
        with st.status("Training cancellation classifiers..."):
            # Drop leaky features before any cancellation encoding/training
            df_cancel = drop_leaky_features(df)

            # Fit label encoders externally on the leaky-free df so we can save them for inference
            label_encoders = {}
            for col in _CATEGORICAL_COLS:
                if col in df_cancel.columns:
                    le = LabelEncoder()
                    df_cancel = df_cancel.copy()
                    df_cancel[col] = le.fit_transform(df_cancel[col].astype(str))
                    label_encoders[col] = le

            X, y_cancel, _ = encode_categoricals(drop_leaky_features(df))
            feature_columns = list(X.columns)

            # Compute per-column median/mode defaults for the predictor form
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

        # SHAP feature importance using XGBoost model on test split
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import os
        X_tr, X_te, y_tr2, _ = train_test_split(
            X, y_cancel, test_size=0.2, random_state=42, stratify=y_cancel
        )
        shap_df = get_shap_feature_importance(
            cancel_models["xgboost"], X_te.values, feature_columns
        )
        selected_features = select_top_features(shap_df, threshold=0.01)
        os.makedirs("outputs", exist_ok=True)
        import shap as shap_lib
        fig, ax = plt.subplots(figsize=(8, 6))
        shap_lib.bar_plot(shap_df["mean_abs_shap"].values,
                          feature_names=shap_df["feature"].tolist(),
                          max_display=15, ax=ax, show=False)
        plt.tight_layout()
        plt.savefig("outputs/shap_summary.png", dpi=120, bbox_inches="tight")
        plt.close()

        results["cancel_models"] = cancel_models
        results["feature_columns"] = feature_columns
        results["label_encoders"] = label_encoders
        results["feature_defaults"] = feature_defaults
        results["selected_features"] = selected_features
        model_cache.save_artifact("cancel_models", cancel_models)
        model_cache.save_artifact("feature_columns", feature_columns)
        model_cache.save_artifact("label_encoders", label_encoders)
        model_cache.save_artifact("feature_defaults", feature_defaults)
        model_cache.save_artifact("selected_features", selected_features)

    # ------------------------------------------------------------------ #
    # Pre-computed evaluation metrics (test on last 20% of time series)
    # ------------------------------------------------------------------ #
    if model_cache.artifact_exists("eval_metrics"):
        results["eval_metrics"] = model_cache.load_artifact("eval_metrics")
    else:
        with st.status("Computing evaluation metrics..."):
            split = int(len(rev_series) * 0.8)
            rev_train, rev_test = rev_series.iloc[:split], rev_series.iloc[split:]
            dem_train, dem_test = dem_series.iloc[:split], dem_series.iloc[split:]

            xgb_rev_m, xgb_rev_c = results["xgb_revenue"]
            xgb_dem_m, xgb_dem_c = results["xgb_demand"]

            rev_xgb_preds = forecast_xgb(xgb_rev_m, xgb_rev_c, rev_train, len(rev_test))
            rev_xgb_preds.index = rev_test.index
            rev_naive = naive_baseline_forecast(rev_train, len(rev_test))
            rev_naive.index = rev_test.index

            dem_xgb_preds = forecast_xgb(xgb_dem_m, xgb_dem_c, dem_train, len(dem_test))
            dem_xgb_preds.index = dem_test.index
            dem_naive = naive_baseline_forecast(dem_train, len(dem_test))
            dem_naive.index = dem_test.index

            # Prophet evaluation on test period
            rev_prophet_df = forecast_prophet(results["prophet_revenue"], len(rev_test))
            dem_prophet_df = forecast_prophet(results["prophet_demand"], len(dem_test))

            eval_metrics = {
                "revenue_xgb":    regression_metrics(rev_test.values, rev_xgb_preds.values),
                "revenue_prophet": regression_metrics(rev_test.values, rev_prophet_df["yhat"].values),
                "revenue_naive":  regression_metrics(rev_test.values, rev_naive.values),
                "demand_xgb":    regression_metrics(dem_test.values, dem_xgb_preds.values),
                "demand_prophet": regression_metrics(dem_test.values, dem_prophet_df["yhat"].values),
                "demand_naive":  regression_metrics(dem_test.values, dem_naive.values),
            }
            cv_scores = ts_cross_val_scores(rev_series, n_splits=4, forecast_horizon=30)
            eval_metrics["cv_scores"] = cv_scores

        # Cross-dataset generalization test on cancellation_Prediction.csv
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
        model_cache.save_artifact("eval_metrics", eval_metrics)

    return results


# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Hotel Analytics",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

if APP_MODE == "demo":
    st.info("📊 You're viewing demo data from a sample hotel. Upload your own data to see your hotel's insights.")

st.title("🏨 Hotel Analytics Dashboard")
st.markdown(
    """
    An interactive ML-powered analytics platform built on hotel booking data (119k+ records, 2015–2017).

    **Use the sidebar to navigate between pages:**

    | Page | Description |
    |------|-------------|
    | **Overview** | KPIs, revenue & bookings charts, hotel/date filters |
    | **Revenue Forecast** | XGBoost vs SARIMA vs naive baseline, cross-validation |
    | **Demand & Staffing** | 30-day demand forecast + staffing recommendations |
    | **Cancellation Predictor** | Input a booking → get cancellation probability |
    | **Profit Simulator** | Adjust tax/cost rates → see 30-day profit breakdown |
    """
)

st.info("Models load from disk on startup. First run trains everything (~5 min). Subsequent runs start in ~2s.")

# Warm up the cache so pages load instantly
with st.spinner("Loading data and models..."):
    load_and_prepare_data()
    get_or_train_models()

st.success("Ready! Select a page from the sidebar.")
