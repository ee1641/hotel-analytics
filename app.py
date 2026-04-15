"""
Hotel Analytics Dashboard - Streamlit entry point.
Run with: streamlit run app.py
"""
import logging
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
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
from src.cancellation_evaluation import _align_to_feature_columns, evaluate_cross_dataset
from src.data_context import (
    DatasetContext,
    resolve_hotel_dataset_context,
    resolve_streamlit_dataset_context,
)
from src.evaluation import regression_metrics, ts_cross_val_scores
from src.forecasting import (
    forecast_prophet,
    forecast_xgb,
    naive_baseline_forecast,
    train_prophet_forecaster,
    train_xgb_forecaster,
    train_xgb_forecaster_with_intervals,
)
from src.ingestion import load_cancellation_prediction_csv
from src.model_specs import (
    EVAL_METRICS_VERSION,
    FORECAST_LAGS,
    FORECAST_MODEL_VERSION,
    INTERVAL_MODEL_VERSION,
    MIN_FORECAST_HISTORY,
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
APP_MODE = os.getenv("APP_MODE", "demo")


def _validated_forecast_series(
    daily_df: pd.DataFrame,
    column: str,
    dataset_id: str,
    label: str,
    min_points: int = MIN_FORECAST_HISTORY,
) -> pd.Series:
    series = pd.to_numeric(daily_df[column], errors="coerce")
    series = series[series.index.notna()].dropna()
    if len(series) < min_points:
        raise ValueError(
            f"Dataset '{dataset_id}' has only {len(series)} valid daily {label} rows after preprocessing. "
            f"Forecasting requires at least {min_points} daily observations."
        )
    return series


def _build_expected_revenue_views(
    df: pd.DataFrame,
    cancellation_model,
    feature_columns: list,
    label_encoders: dict,
    feature_defaults: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored_df = df.copy()
    X_scored = _align_to_feature_columns(
        drop_leaky_features(scored_df),
        feature_columns,
        label_encoders,
        feature_defaults,
    )
    if hasattr(cancellation_model, "predict_proba"):
        cancel_probability = cancellation_model.predict_proba(X_scored)[:, 1]
    else:
        cancel_probability = cancellation_model.predict(X_scored).astype(float)
    scored_df["cancel_probability"] = cancel_probability
    scored_df["expected_realized_revenue"] = (
        scored_df["booked_revenue"] * (1 - scored_df["cancel_probability"])
    ).clip(lower=0)
    return scored_df, aggregate_daily(scored_df)


def resolve_active_dataset_context(
    dataset_context: DatasetContext | dict | None = None,
    *,
    hotel_id: str | None = None,
) -> DatasetContext:
    if dataset_context is not None:
        if isinstance(dataset_context, DatasetContext):
            return dataset_context
        return DatasetContext(**dataset_context)
    if hotel_id is not None:
        return resolve_hotel_dataset_context(hotel_id)
    return resolve_streamlit_dataset_context(st.session_state)


@st.cache_data(show_spinner=False)
def _load_and_prepare_data_from_path(data_path: str, dataset_id: str):
    df = engineer_features(clean_data(load_data(data_path)))
    df = build_arrival_date(df)
    df = add_temporal_features(df)
    daily_df = aggregate_daily(df)
    return df, daily_df


def load_and_prepare_data(
    data_path: str = DATA_PATH,
    *,
    dataset_context: DatasetContext | dict | None = None,
    hotel_id: str | None = None,
):
    context = resolve_active_dataset_context(dataset_context, hotel_id=hotel_id)
    return _load_and_prepare_data_from_path(context.data_path, context.dataset_id)


def _artifact_current(name: str, version: str, models_dir: Path) -> bool:
    if not model_cache.artifact_exists(name, models_dir=models_dir):
        return False
    _, metadata = model_cache.load_artifact_with_metadata(name, models_dir=models_dir)
    return metadata.get("version") == version


def _save_versioned(name: str, obj, version: str, models_dir: Path) -> None:
    model_cache.save_artifact(name, obj, metadata={"version": version}, models_dir=models_dir)


@st.cache_resource(show_spinner=False)
def _get_or_train_models_for_context(data_path: str, models_dir: str, dataset_id: str):
    """
    Granular per-artifact check: load from disk if present, otherwise train and save.
    """
    models_path = Path(models_dir)
    df, daily_df = _load_and_prepare_data_from_path(data_path, dataset_id)
    results = {}

    # ------------------------------------------------------------------
    # Cancellation classification
    # ------------------------------------------------------------------
    if model_cache.artifact_exists("cancel_models", models_dir=models_path):
        results["cancel_models"] = model_cache.load_artifact("cancel_models", models_dir=models_path)
        results["feature_columns"] = model_cache.load_artifact("feature_columns", models_dir=models_path)
        results["label_encoders"] = model_cache.load_artifact("label_encoders", models_dir=models_path)
        results["feature_defaults"] = model_cache.load_artifact("feature_defaults", models_dir=models_path)
        results["selected_features"] = model_cache.load_artifact("selected_features", models_dir=models_path) or []
    else:
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
            X,
            y_cancel,
            test_size=0.2,
            random_state=42,
            stratify=y_cancel,
        )
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

        _, X_te, _, _ = train_test_split(
            X,
            y_cancel,
            test_size=0.2,
            random_state=42,
            stratify=y_cancel,
        )
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
        model_cache.save_artifact("cancel_models", cancel_models, models_dir=models_path)
        model_cache.save_artifact("feature_columns", feature_columns, models_dir=models_path)
        model_cache.save_artifact("label_encoders", label_encoders, models_dir=models_path)
        model_cache.save_artifact("feature_defaults", feature_defaults, models_dir=models_path)
        model_cache.save_artifact("selected_features", selected_features, models_dir=models_path)

    df_with_expected_revenue, revenue_daily_df = _build_expected_revenue_views(
        df,
        results["cancel_models"]["xgboost"],
        results["feature_columns"],
        results["label_encoders"],
        results["feature_defaults"],
    )
    results["df_with_expected_revenue"] = df_with_expected_revenue
    results["daily_revenue_views"] = revenue_daily_df

    revenue_model_specs = [
        (
            "xgb_revenue",
            "xgb_revenue_intervals",
            "prophet_revenue",
            "daily_booked_revenue",
            "booked revenue",
        ),
        (
            "xgb_realized_revenue",
            "xgb_realized_revenue_intervals",
            "prophet_realized_revenue",
            "daily_realized_revenue",
            "realized revenue",
        ),
        (
            "xgb_expected_revenue",
            "xgb_expected_revenue_intervals",
            "prophet_expected_revenue",
            "daily_expected_realized_revenue",
            "expected realized revenue",
        ),
    ]

    revenue_eval_specs = [
        (
            "revenue_booked_xgb",
            "revenue_booked_prophet",
            "revenue_booked_naive",
            "daily_booked_revenue",
        ),
        (
            "revenue_realized_xgb",
            "revenue_realized_prophet",
            "revenue_realized_naive",
            "daily_realized_revenue",
        ),
        (
            "revenue_expected_xgb",
            "revenue_expected_prophet",
            "revenue_expected_naive",
            "daily_expected_realized_revenue",
        ),
    ]

    revenue_series_lookup = {}
    for xgb_key, interval_key, prophet_key, column, label in revenue_model_specs:
        series = _validated_forecast_series(revenue_daily_df, column, dataset_id, label)
        revenue_series_lookup[column] = series

        if _artifact_current(xgb_key, FORECAST_MODEL_VERSION, models_dir=models_path):
            results[xgb_key] = model_cache.load_artifact(xgb_key, models_dir=models_path)
        else:
            xgb_m, xgb_cols = train_xgb_forecaster(series, lags=FORECAST_LAGS)
            results[xgb_key] = (xgb_m, xgb_cols)
            _save_versioned(xgb_key, (xgb_m, xgb_cols), FORECAST_MODEL_VERSION, models_path)

        if _artifact_current(interval_key, INTERVAL_MODEL_VERSION, models_dir=models_path):
            results[interval_key] = model_cache.load_artifact(interval_key, models_dir=models_path)
        else:
            interval_models = train_xgb_forecaster_with_intervals(series, lags=FORECAST_LAGS)
            results[interval_key] = interval_models
            _save_versioned(interval_key, interval_models, INTERVAL_MODEL_VERSION, models_path)

        if _artifact_current(prophet_key, PROPHET_MODEL_VERSION, models_dir=models_path):
            results[prophet_key] = model_cache.load_artifact(prophet_key, models_dir=models_path)
        else:
            prophet_model = train_prophet_forecaster(series)
            results[prophet_key] = prophet_model
            _save_versioned(prophet_key, prophet_model, PROPHET_MODEL_VERSION, models_path)

    dem_series = _validated_forecast_series(revenue_daily_df, "daily_bookings", dataset_id, "booking")

    if _artifact_current("xgb_demand", FORECAST_MODEL_VERSION, models_dir=models_path):
        results["xgb_demand"] = model_cache.load_artifact("xgb_demand", models_dir=models_path)
    else:
        xgb_d, xgb_d_cols = train_xgb_forecaster(dem_series, lags=FORECAST_LAGS)
        results["xgb_demand"] = (xgb_d, xgb_d_cols)
        _save_versioned("xgb_demand", (xgb_d, xgb_d_cols), FORECAST_MODEL_VERSION, models_path)

    if _artifact_current("xgb_demand_intervals", INTERVAL_MODEL_VERSION, models_dir=models_path):
        results["xgb_demand_intervals"] = model_cache.load_artifact("xgb_demand_intervals", models_dir=models_path)
    else:
        interval_models = train_xgb_forecaster_with_intervals(dem_series, lags=FORECAST_LAGS)
        results["xgb_demand_intervals"] = interval_models
        _save_versioned("xgb_demand_intervals", interval_models, INTERVAL_MODEL_VERSION, models_path)

    if _artifact_current("prophet_demand", PROPHET_MODEL_VERSION, models_dir=models_path):
        results["prophet_demand"] = model_cache.load_artifact("prophet_demand", models_dir=models_path)
    else:
        prophet_dem = train_prophet_forecaster(dem_series)
        results["prophet_demand"] = prophet_dem
        _save_versioned("prophet_demand", prophet_dem, PROPHET_MODEL_VERSION, models_path)

    if _artifact_current("cancel_survival_model", SURVIVAL_MODEL_VERSION, models_dir=models_path):
        results["cancel_survival_model"] = model_cache.load_artifact("cancel_survival_model", models_dir=models_path)
    else:
        survival_bundle = train_cancellation_survival_model(df)
        results["cancel_survival_model"] = survival_bundle
        _save_versioned("cancel_survival_model", survival_bundle, SURVIVAL_MODEL_VERSION, models_path)

    required_eval_keys = {
        "demand_xgb",
        "demand_prophet",
        "demand_naive",
        "revenue_booked_xgb",
        "revenue_booked_prophet",
        "revenue_booked_naive",
        "revenue_realized_xgb",
        "revenue_realized_prophet",
        "revenue_realized_naive",
        "revenue_expected_xgb",
        "revenue_expected_prophet",
        "revenue_expected_naive",
        "cv_scores",
    }
    cached_eval_metrics = None
    if _artifact_current("eval_metrics", EVAL_METRICS_VERSION, models_dir=models_path):
        cached_eval_metrics = model_cache.load_artifact("eval_metrics", models_dir=models_path)

    if isinstance(cached_eval_metrics, dict) and required_eval_keys.issubset(cached_eval_metrics.keys()):
        results["eval_metrics"] = cached_eval_metrics
    else:
        test_days = 30

        dem_train = dem_series.iloc[:-test_days]
        dem_test = dem_series.iloc[-test_days:]
        xgb_dem_m, xgb_dem_c = train_xgb_forecaster(dem_train, lags=FORECAST_LAGS)
        dem_xgb_preds = forecast_xgb(xgb_dem_m, xgb_dem_c, dem_train, len(dem_test))
        dem_xgb_preds.index = dem_test.index
        dem_naive = naive_baseline_forecast(dem_train, len(dem_test))
        dem_naive.index = dem_test.index
        dem_prophet_df = forecast_prophet(train_prophet_forecaster(dem_train), len(dem_test))

        eval_metrics = {
            "demand_xgb": regression_metrics(dem_test.values, dem_xgb_preds.values),
            "demand_prophet": regression_metrics(dem_test.values, dem_prophet_df["yhat"].values),
            "demand_naive": regression_metrics(dem_test.values, dem_naive.values),
        }

        for metric_xgb, metric_prophet, metric_naive, column in revenue_eval_specs:
            revenue_series = revenue_series_lookup[column]
            rev_train = revenue_series.iloc[:-test_days]
            rev_test = revenue_series.iloc[-test_days:]

            xgb_rev_m, xgb_rev_c = train_xgb_forecaster(rev_train, lags=FORECAST_LAGS)
            rev_xgb_preds = forecast_xgb(xgb_rev_m, xgb_rev_c, rev_train, len(rev_test))
            rev_xgb_preds.index = rev_test.index
            rev_naive = naive_baseline_forecast(rev_train, len(rev_test))
            rev_naive.index = rev_test.index
            rev_prophet_df = forecast_prophet(train_prophet_forecaster(rev_train), len(rev_test))

            eval_metrics[metric_xgb] = regression_metrics(rev_test.values, rev_xgb_preds.values)
            eval_metrics[metric_prophet] = regression_metrics(rev_test.values, rev_prophet_df["yhat"].values)
            eval_metrics[metric_naive] = regression_metrics(rev_test.values, rev_naive.values)

        eval_metrics["cv_scores"] = ts_cross_val_scores(
            revenue_series_lookup["daily_booked_revenue"],
            n_splits=4,
            forecast_horizon=30,
        )

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
        except Exception as exc:
            logging.warning("Cross-dataset evaluation failed: %s", exc)
            eval_metrics["cross_dataset"] = {}

        results["eval_metrics"] = eval_metrics
        _save_versioned("eval_metrics", eval_metrics, EVAL_METRICS_VERSION, models_path)

    return results


def get_or_train_models(
    data_path: str = DATA_PATH,
    *,
    dataset_context: DatasetContext | dict | None = None,
    hotel_id: str | None = None,
):
    context = resolve_active_dataset_context(dataset_context, hotel_id=hotel_id)
    return _get_or_train_models_for_context(context.data_path, context.models_dir, context.dataset_id)


def render_home_page() -> None:
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
        An interactive hotel revenue-management workspace built on booking data.

        **Use the sidebar to navigate between pages:**

        | Page | Description |
        |------|-------------|
        | **Overview** | KPIs, revenue & bookings charts, hotel/date filters |
        | **Revenue Forecast** | Demand/revenue forecast, pace, pricing, and intervals |
        | **Demand & Staffing** | 30-day demand forecast, staffing, and segment mix |
        | **Cancellation Predictor** | Cancellation probability plus survival profile |
        | **Profit Simulator** | Adjust tax/cost rates -> see 30-day profit breakdown |
        """
    )

    load_and_prepare_data()
    get_or_train_models()


if __name__ == "__main__":
    render_home_page()
