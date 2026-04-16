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
from sklearn.preprocessing import LabelEncoder

from src import model_cache
from src.revenue_management import build_otb_feature_matrix  # available for analytics; not injected into training
from src.weather import build_weather_features
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
    blend_forecasts,
    find_ensemble_weights,
    forecast_prophet,
    forecast_xgb,
    naive_baseline_forecast,
    train_prophet_forecaster,
    train_xgb_forecaster,
    train_xgb_forecaster_with_intervals,
    tune_xgb_forecaster,
)
from src.ingestion import load_cancellation_prediction_csv
from src.model_specs import (
    CANCELLATION_MODEL_VERSION,
    ENSEMBLE_WEIGHTS_VERSION,
    EVAL_METRICS_VERSION,
    FORECAST_LAGS,
    FORECAST_MODEL_VERSION,
    FORECAST_TUNE_N_ITER,
    FORECAST_TUNE_N_SPLITS,
    INTERVAL_MODEL_VERSION,
    MIN_FORECAST_HISTORY,
    PROPHET_MODEL_VERSION,
    SEGMENTED_INTERVAL_VERSION,
    SEGMENTED_MODEL_VERSION,
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
from src.time_series import aggregate_daily, aggregate_daily_by_hotel_type, build_arrival_date

DATA_PATH = "hotel_bookings.csv"
APP_MODE = os.getenv("APP_MODE", "demo")


def _hotel_slug(hotel_type: str) -> str:
    """'Resort Hotel' → 'resort_hotel' (used as cache-key suffix)."""
    return hotel_type.lower().replace(" ", "_")


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


@st.cache_resource(show_spinner="Training models… this may take a few minutes on first run.")
def _get_or_train_models_for_context(data_path: str, models_dir: str, dataset_id: str):
    """
    Granular per-artifact check: load from disk if present, otherwise train and save.
    """
    models_path = Path(models_dir)
    df, daily_df = _load_and_prepare_data_from_path(data_path, dataset_id)
    results = {}
    print("[1/7] Cancellation models…", flush=True)

    # ------------------------------------------------------------------
    # Cancellation classification
    # ------------------------------------------------------------------
    if _artifact_current("cancel_models", CANCELLATION_MODEL_VERSION, models_dir=models_path):
        results["cancel_models"] = model_cache.load_artifact("cancel_models", models_dir=models_path)
        results["feature_columns"] = model_cache.load_artifact("feature_columns", models_dir=models_path)
        results["label_encoders"] = model_cache.load_artifact("label_encoders", models_dir=models_path)
        results["feature_defaults"] = model_cache.load_artifact("feature_defaults", models_dir=models_path)
        results["selected_features"] = model_cache.load_artifact("selected_features", models_dir=models_path) or []
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
        print("[1/7] Cancellation models done.", flush=True)
        _save_versioned("cancel_models", cancel_models, CANCELLATION_MODEL_VERSION, models_path)
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

    # ------------------------------------------------------------------
    # OTB-30d feature matrix — otb_30d only, snapshot at arrival-30 days.
    # At inference, otb_today for dates 30 days out is the exact same
    # quantity (bookings made 30+ days before arrival), so training and
    # inference distributions are aligned.
    # ------------------------------------------------------------------
    try:
        otb_matrix_combined = build_otb_feature_matrix(df, lead_times=[30, 60])[["otb_30d", "otb_60d"]]
    except Exception as exc:
        logging.warning("OTB-30d matrix failed (training without OTB): %s", exc)
        otb_matrix_combined = pd.DataFrame()

    # ------------------------------------------------------------------
    # Weather feature matrix (Track 3) — Open-Meteo historical archive.
    # Fetched once and cached in data/weather_portugal.csv.
    # ------------------------------------------------------------------
    try:
        weather_matrix = build_weather_features(daily_df.index)
    except Exception as exc:
        logging.warning("Weather features unavailable (training without): %s", exc)
        weather_matrix = pd.DataFrame()

    def _combined_extra(series_index: pd.Index) -> pd.DataFrame | None:
        """Merge OTB + weather extra features aligned to a series index."""
        frames = []
        if not otb_matrix_combined.empty:
            frames.append(otb_matrix_combined.reindex(series_index).fillna(0))
        if not weather_matrix.empty:
            frames.append(weather_matrix.reindex(series_index).fillna(0))
        if not frames:
            return None
        if len(frames) == 1:
            return frames[0]
        return pd.concat(frames, axis=1)

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

    # Cache of tuned hyperparams keyed by series column name — tune once, reuse for
    # both the main model and the quantile-interval model of the same series.
    _series_best_params: dict = {}

    print("[2/7] Revenue & demand forecast models (3 series)…", flush=True)
    revenue_series_lookup = {}
    for xgb_key, interval_key, prophet_key, column, label in revenue_model_specs:
        series = _validated_forecast_series(revenue_daily_df, column, dataset_id, label)
        revenue_series_lookup[column] = series
        _extra = _combined_extra(series.index)

        xgb_cached      = _artifact_current(xgb_key,      FORECAST_MODEL_VERSION,  models_dir=models_path)
        interval_cached = _artifact_current(interval_key,  INTERVAL_MODEL_VERSION,  models_dir=models_path)

        if xgb_cached:
            results[xgb_key] = model_cache.load_artifact(xgb_key, models_dir=models_path)
        if interval_cached:
            results[interval_key] = model_cache.load_artifact(interval_key, models_dir=models_path)

        if not xgb_cached or not interval_cached:
            # Tune once; reuse params for both main model and intervals model.
            best_p = tune_xgb_forecaster(series, lags=FORECAST_LAGS,
                                         n_iter=FORECAST_TUNE_N_ITER, n_splits=FORECAST_TUNE_N_SPLITS)
            _series_best_params[column] = best_p
            if not xgb_cached:
                xgb_m, xgb_cols = train_xgb_forecaster(series, lags=FORECAST_LAGS,
                                                        extra_features=_extra, **best_p)
                results[xgb_key] = (xgb_m, xgb_cols)
                _save_versioned(xgb_key, (xgb_m, xgb_cols), FORECAST_MODEL_VERSION, models_path)
            if not interval_cached:
                interval_models = train_xgb_forecaster_with_intervals(series, lags=FORECAST_LAGS,
                                                                       extra_features=_extra, **best_p)
                results[interval_key] = interval_models
                _save_versioned(interval_key, interval_models, INTERVAL_MODEL_VERSION, models_path)
        else:
            # Both cached — no tuning needed; store a sentinel so eval can fall back gracefully.
            _series_best_params.setdefault(column, None)

        if _artifact_current(prophet_key, PROPHET_MODEL_VERSION, models_dir=models_path):
            results[prophet_key] = model_cache.load_artifact(prophet_key, models_dir=models_path)
        else:
            prophet_model = train_prophet_forecaster(series)
            results[prophet_key] = prophet_model
            _save_versioned(prophet_key, prophet_model, PROPHET_MODEL_VERSION, models_path)

    print("[3/7] Demand forecast model…", flush=True)
    dem_series = _validated_forecast_series(revenue_daily_df, "daily_bookings", dataset_id, "booking")
    _extra_dem = _combined_extra(dem_series.index)

    dem_xgb_cached      = _artifact_current("xgb_demand",           FORECAST_MODEL_VERSION,  models_dir=models_path)
    dem_interval_cached = _artifact_current("xgb_demand_intervals",  INTERVAL_MODEL_VERSION,  models_dir=models_path)

    if dem_xgb_cached:
        results["xgb_demand"] = model_cache.load_artifact("xgb_demand", models_dir=models_path)
    if dem_interval_cached:
        results["xgb_demand_intervals"] = model_cache.load_artifact("xgb_demand_intervals", models_dir=models_path)

    if not dem_xgb_cached or not dem_interval_cached:
        best_p_dem = tune_xgb_forecaster(dem_series, lags=FORECAST_LAGS,
                                         n_iter=FORECAST_TUNE_N_ITER, n_splits=FORECAST_TUNE_N_SPLITS)
        _series_best_params["daily_bookings"] = best_p_dem
        if not dem_xgb_cached:
            xgb_d, xgb_d_cols = train_xgb_forecaster(dem_series, lags=FORECAST_LAGS,
                                                       extra_features=_extra_dem, **best_p_dem)
            results["xgb_demand"] = (xgb_d, xgb_d_cols)
            _save_versioned("xgb_demand", (xgb_d, xgb_d_cols), FORECAST_MODEL_VERSION, models_path)
        if not dem_interval_cached:
            interval_models = train_xgb_forecaster_with_intervals(dem_series, lags=FORECAST_LAGS,
                                                                   extra_features=_extra_dem, **best_p_dem)
            results["xgb_demand_intervals"] = interval_models
            _save_versioned("xgb_demand_intervals", interval_models, INTERVAL_MODEL_VERSION, models_path)
    else:
        _series_best_params.setdefault("daily_bookings", None)

    if _artifact_current("prophet_demand", PROPHET_MODEL_VERSION, models_dir=models_path):
        results["prophet_demand"] = model_cache.load_artifact("prophet_demand", models_dir=models_path)
    else:
        prophet_dem = train_prophet_forecaster(dem_series)
        results["prophet_demand"] = prophet_dem
        _save_versioned("prophet_demand", prophet_dem, PROPHET_MODEL_VERSION, models_path)

    print("[4/7] Ensemble weights…", flush=True)
    # ------------------------------------------------------------------
    # Ensemble weights: find optimal XGBoost weight in XGBoost+Prophet blend
    # Optimised on a 60-day validation window before the holdout test set.
    # Reuses tuned hyperparams from the full series — no extra tuning pass.
    # ------------------------------------------------------------------
    if _artifact_current("ensemble_weights", ENSEMBLE_WEIGHTS_VERSION, models_dir=models_path):
        results["ensemble_weights"] = model_cache.load_artifact("ensemble_weights", models_dir=models_path)
    else:
        try:
            val_days = 60
            test_days_ew = 30
            dem_val = dem_series.iloc[-(test_days_ew + val_days):-test_days_ew]
            dem_val_train = dem_series.iloc[:-(test_days_ew + val_days)]
            _val_extra = _combined_extra(dem_val_train.index)
            _vp_dem = _series_best_params.get("daily_bookings") or {}
            xgb_val_m, xgb_val_c = train_xgb_forecaster(dem_val_train, lags=FORECAST_LAGS,
                                                          extra_features=_val_extra, **_vp_dem)
            prophet_val_m = train_prophet_forecaster(dem_val_train)
            xgb_val_preds  = forecast_xgb(xgb_val_m, xgb_val_c, dem_val_train, val_days)
            prophet_val_df = forecast_prophet(prophet_val_m, val_days)
            w_dem = find_ensemble_weights(xgb_val_preds.values, prophet_val_df["yhat"].values,
                                          dem_val.values)

            # Revenue ensemble weight (booked revenue)
            rev_series = revenue_series_lookup["daily_booked_revenue"]
            rev_val = rev_series.iloc[-(test_days_ew + val_days):-test_days_ew]
            rev_val_train = rev_series.iloc[:-(test_days_ew + val_days)]
            _val_extra_rev = _combined_extra(rev_val_train.index)
            _vp_rev = _series_best_params.get("daily_booked_revenue") or {}
            xgb_rev_val_m, xgb_rev_val_c = train_xgb_forecaster(rev_val_train, lags=FORECAST_LAGS,
                                                                   extra_features=_val_extra_rev, **_vp_rev)
            prophet_rev_val_m = train_prophet_forecaster(rev_val_train)
            xgb_rev_preds     = forecast_xgb(xgb_rev_val_m, xgb_rev_val_c, rev_val_train, val_days)
            prophet_rev_df    = forecast_prophet(prophet_rev_val_m, val_days)
            w_rev = find_ensemble_weights(xgb_rev_preds.values, prophet_rev_df["yhat"].values,
                                          rev_val.values)

            ensemble_weights = {"demand": w_dem, "revenue": w_rev}
            results["ensemble_weights"] = ensemble_weights
            _save_versioned("ensemble_weights", ensemble_weights, ENSEMBLE_WEIGHTS_VERSION, models_path)
            logging.info("Ensemble weights — demand XGB: %.2f, revenue XGB: %.2f", w_dem, w_rev)
        except Exception as exc:
            logging.warning("Ensemble weight computation failed: %s", exc)
            results["ensemble_weights"] = {"demand": 1.0, "revenue": 1.0}

    print("[5/7] Survival model…", flush=True)
    if _artifact_current("cancel_survival_model", SURVIVAL_MODEL_VERSION, models_dir=models_path):
        results["cancel_survival_model"] = model_cache.load_artifact("cancel_survival_model", models_dir=models_path)
    else:
        survival_bundle = train_cancellation_survival_model(df)
        results["cancel_survival_model"] = survival_bundle
        _save_versioned("cancel_survival_model", survival_bundle, SURVIVAL_MODEL_VERSION, models_path)

    print("[6/7] Per-hotel segmented models…", flush=True)
    # ------------------------------------------------------------------
    # Per-hotel-type segmented models (Track 1 + Track 2)
    # ------------------------------------------------------------------
    hotel_daily_map = aggregate_daily_by_hotel_type(df)
    results["hotel_types_available"] = list(hotel_daily_map.keys())

    # Tuned hyperparams keyed by (slug, column) — tune once per series, reuse
    # for both the main segmented model and its quantile-interval companion.
    _hotel_best_params: dict = {}

    for hotel_type, hotel_daily_df in hotel_daily_map.items():
        slug = _hotel_slug(hotel_type)
        hotel_df_scored = df_with_expected_revenue[df_with_expected_revenue["hotel"] == hotel_type]
        h_rev_daily = aggregate_daily(hotel_df_scored)
        try:
            otb_matrix_hotel = build_otb_feature_matrix(
                df[df["hotel"] == hotel_type], lead_times=[30, 60]
            )[["otb_30d", "otb_60d"]]
        except Exception as exc:
            logging.warning("OTB-30d matrix failed for '%s': %s", hotel_type, exc)
            otb_matrix_hotel = pd.DataFrame()

        def _hotel_extra(series_index):
            frames = []
            if not otb_matrix_hotel.empty:
                frames.append(otb_matrix_hotel.reindex(series_index).fillna(0))
            if not weather_matrix.empty:
                frames.append(weather_matrix.reindex(series_index).fillna(0))
            if not frames:
                return None
            return frames[0] if len(frames) == 1 else pd.concat(frames, axis=1)

        hotel_rev_specs = [
            (f"xgb_revenue_{slug}", f"xgb_revenue_intervals_{slug}",
             "daily_booked_revenue", "booked revenue"),
            (f"xgb_realized_revenue_{slug}", f"xgb_realized_revenue_intervals_{slug}",
             "daily_realized_revenue", "realized revenue"),
            (f"xgb_expected_revenue_{slug}", f"xgb_expected_revenue_intervals_{slug}",
             "daily_expected_realized_revenue", "expected realized revenue"),
        ]
        for xgb_key, interval_key, column, label in hotel_rev_specs:
            try:
                h_series = _validated_forecast_series(h_rev_daily, column,
                                                      f"{dataset_id}_{slug}", label)
            except ValueError:
                continue
            _h_extra = _hotel_extra(h_series.index)

            h_xgb_cached      = _artifact_current(xgb_key,      SEGMENTED_MODEL_VERSION,   models_dir=models_path)
            h_interval_cached = _artifact_current(interval_key,  SEGMENTED_INTERVAL_VERSION, models_dir=models_path)

            if h_xgb_cached:
                results[xgb_key] = model_cache.load_artifact(xgb_key, models_dir=models_path)
            if h_interval_cached:
                results[interval_key] = model_cache.load_artifact(interval_key, models_dir=models_path)

            if not h_xgb_cached or not h_interval_cached:
                best_p = tune_xgb_forecaster(h_series, lags=FORECAST_LAGS,
                                             n_iter=FORECAST_TUNE_N_ITER, n_splits=FORECAST_TUNE_N_SPLITS)
                _hotel_best_params[(slug, column)] = best_p
                if not h_xgb_cached:
                    xgb_m, xgb_cols = train_xgb_forecaster(h_series, lags=FORECAST_LAGS,
                                                            extra_features=_h_extra, **best_p)
                    results[xgb_key] = (xgb_m, xgb_cols)
                    _save_versioned(xgb_key, (xgb_m, xgb_cols), SEGMENTED_MODEL_VERSION, models_path)
                if not h_interval_cached:
                    imodels = train_xgb_forecaster_with_intervals(h_series, lags=FORECAST_LAGS,
                                                                  extra_features=_h_extra, **best_p)
                    results[interval_key] = imodels
                    _save_versioned(interval_key, imodels, SEGMENTED_INTERVAL_VERSION, models_path)
            else:
                _hotel_best_params.setdefault((slug, column), None)

        dem_key = f"xgb_demand_{slug}"
        dem_int_key = f"xgb_demand_intervals_{slug}"
        try:
            h_dem = _validated_forecast_series(hotel_daily_df, "daily_bookings",
                                               f"{dataset_id}_{slug}", "booking")
        except ValueError:
            continue
        _h_extra_dem = _hotel_extra(h_dem.index)

        h_dem_cached     = _artifact_current(dem_key,     SEGMENTED_MODEL_VERSION,    models_dir=models_path)
        h_dem_int_cached = _artifact_current(dem_int_key, SEGMENTED_INTERVAL_VERSION, models_dir=models_path)

        if h_dem_cached:
            results[dem_key] = model_cache.load_artifact(dem_key, models_dir=models_path)
        if h_dem_int_cached:
            results[dem_int_key] = model_cache.load_artifact(dem_int_key, models_dir=models_path)

        if not h_dem_cached or not h_dem_int_cached:
            best_p_hd = tune_xgb_forecaster(h_dem, lags=FORECAST_LAGS,
                                             n_iter=FORECAST_TUNE_N_ITER, n_splits=FORECAST_TUNE_N_SPLITS)
            _hotel_best_params[(slug, "daily_bookings")] = best_p_hd
            if not h_dem_cached:
                xgb_d, xgb_d_cols = train_xgb_forecaster(h_dem, lags=FORECAST_LAGS,
                                                           extra_features=_h_extra_dem, **best_p_hd)
                results[dem_key] = (xgb_d, xgb_d_cols)
                _save_versioned(dem_key, (xgb_d, xgb_d_cols), SEGMENTED_MODEL_VERSION, models_path)
            if not h_dem_int_cached:
                imodels = train_xgb_forecaster_with_intervals(h_dem, lags=FORECAST_LAGS,
                                                               extra_features=_h_extra_dem, **best_p_hd)
                results[dem_int_key] = imodels
                _save_versioned(dem_int_key, imodels, SEGMENTED_INTERVAL_VERSION, models_path)
        else:
            _hotel_best_params.setdefault((slug, "daily_bookings"), None)

    print("[7/7] Evaluation metrics…", flush=True)
    required_eval_keys = {
        "demand_xgb",
        "demand_xgb_baseline",
        "demand_prophet",
        "demand_naive",
        "revenue_booked_xgb",
        "revenue_booked_xgb_baseline",
        "revenue_booked_prophet",
        "revenue_booked_naive",
        "revenue_realized_xgb",
        "revenue_realized_xgb_baseline",
        "revenue_realized_prophet",
        "revenue_realized_naive",
        "revenue_expected_xgb",
        "revenue_expected_xgb_baseline",
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

        # OTB + weather features aligned to training split (no leakage — only uses train dates)
        _otb_eval = _combined_extra(dem_train.index)

        # Baseline (fixed default hyperparameters, no OTB)
        xgb_base_m, xgb_base_c = train_xgb_forecaster(dem_train, lags=FORECAST_LAGS)
        dem_base_preds = forecast_xgb(xgb_base_m, xgb_base_c, dem_train, len(dem_test))
        dem_base_preds.index = dem_test.index

        dem_best_p = _series_best_params.get("daily_bookings") or {}
        xgb_dem_m, xgb_dem_c = train_xgb_forecaster(dem_train, lags=FORECAST_LAGS,
                                                      extra_features=_otb_eval, **dem_best_p)
        dem_xgb_preds = forecast_xgb(xgb_dem_m, xgb_dem_c, dem_train, len(dem_test))
        dem_xgb_preds.index = dem_test.index

        dem_naive = naive_baseline_forecast(dem_train, len(dem_test))
        dem_naive.index = dem_test.index
        dem_prophet_df = forecast_prophet(train_prophet_forecaster(dem_train), len(dem_test))

        # Ensemble blend for demand
        _dem_w = find_ensemble_weights(dem_xgb_preds.values, dem_prophet_df["yhat"].values, dem_test.values)
        dem_ensemble = blend_forecasts(dem_xgb_preds, dem_prophet_df, _dem_w)

        eval_metrics = {
            "demand_xgb_baseline": regression_metrics(dem_test.values, dem_base_preds.values),
            "demand_xgb": regression_metrics(dem_test.values, dem_xgb_preds.values),
            "demand_prophet": regression_metrics(dem_test.values, dem_prophet_df["yhat"].values),
            "demand_naive": regression_metrics(dem_test.values, dem_naive.values),
            "demand_ensemble": regression_metrics(dem_test.values, dem_ensemble.values),
            "demand_ensemble_xgb_weight": _dem_w,
        }

        for metric_xgb, metric_prophet, metric_naive, column in revenue_eval_specs:
            revenue_series = revenue_series_lookup[column]
            rev_train = revenue_series.iloc[:-test_days]
            rev_test = revenue_series.iloc[-test_days:]

            # OTB + weather features aligned to this revenue series' training split
            _otb_rev_eval = _combined_extra(rev_train.index)

            # Baseline (no OTB)
            xgb_rev_base, xgb_rev_base_c = train_xgb_forecaster(rev_train, lags=FORECAST_LAGS)
            rev_base_preds = forecast_xgb(xgb_rev_base, xgb_rev_base_c, rev_train, len(rev_test))
            rev_base_preds.index = rev_test.index

            rev_best_p = _series_best_params.get(column) or {}
            xgb_rev_m, xgb_rev_c = train_xgb_forecaster(rev_train, lags=FORECAST_LAGS,
                                                          extra_features=_otb_rev_eval, **rev_best_p)
            rev_xgb_preds = forecast_xgb(xgb_rev_m, xgb_rev_c, rev_train, len(rev_test))
            rev_xgb_preds.index = rev_test.index

            rev_naive = naive_baseline_forecast(rev_train, len(rev_test))
            rev_naive.index = rev_test.index
            rev_prophet_df = forecast_prophet(train_prophet_forecaster(rev_train), len(rev_test))

            _rev_w = find_ensemble_weights(rev_xgb_preds.values, rev_prophet_df["yhat"].values, rev_test.values)
            rev_ensemble = blend_forecasts(rev_xgb_preds, rev_prophet_df, _rev_w)
            eval_metrics[f"{metric_xgb}_baseline"] = regression_metrics(rev_test.values, rev_base_preds.values)
            eval_metrics[metric_xgb] = regression_metrics(rev_test.values, rev_xgb_preds.values)
            eval_metrics[metric_prophet] = regression_metrics(rev_test.values, rev_prophet_df["yhat"].values)
            eval_metrics[metric_naive] = regression_metrics(rev_test.values, rev_naive.values)
            eval_metrics[f"{metric_xgb}_ensemble"] = regression_metrics(rev_test.values, rev_ensemble.values)

        # Per-hotel-type eval metrics
        _hotel_rev_col_map = {
            "demand":           ("daily_bookings",               "demand_xgb"),
            "revenue_booked":   ("daily_booked_revenue",         "revenue_booked_xgb"),
            "revenue_realized": ("daily_realized_revenue",       "revenue_realized_xgb"),
            "revenue_expected": ("daily_expected_realized_revenue", "revenue_expected_xgb"),
        }
        for hotel_type, hotel_daily_df in hotel_daily_map.items():
            slug = _hotel_slug(hotel_type)
            hotel_df_scored2 = df_with_expected_revenue[df_with_expected_revenue["hotel"] == hotel_type]
            h_rev_daily2 = aggregate_daily(hotel_df_scored2)
            for metric_prefix, (column, _) in _hotel_rev_col_map.items():
                src_df = hotel_daily_df if column == "daily_bookings" else h_rev_daily2
                try:
                    h_s = _validated_forecast_series(src_df, column, f"{dataset_id}_{slug}", metric_prefix)
                except ValueError:
                    continue
                if len(h_s) <= test_days:
                    continue
                h_train = h_s.iloc[:-test_days]
                h_test = h_s.iloc[-test_days:]
                h_best_p = _hotel_best_params.get((slug, column)) or {}
                h_m, h_c = train_xgb_forecaster(h_train, lags=FORECAST_LAGS,
                                                extra_features=_combined_extra(h_train.index), **h_best_p)
                h_preds = forecast_xgb(h_m, h_c, h_train, len(h_test))
                h_preds.index = h_test.index
                eval_metrics[f"{metric_prefix}_xgb_{slug}"] = regression_metrics(h_test.values, h_preds.values)

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

    print("All models ready.", flush=True)
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
