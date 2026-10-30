"""
Shared data loading and model training logic.
Importable by both the Streamlit dashboard and the FastAPI backend.
"""
import logging
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src import model_cache
from src.cancellation import (
    TARGET_ENCODED_COLS,
    apply_probability_calibrator,
    compute_scale_pos_weight,
    find_optimal_threshold,
    get_shap_feature_importance,
    predict_stacked_proba,
    select_top_features,
    train_cancellation_models_balanced,
    train_cancellation_survival_model,
    train_probability_calibrator,
    train_stacked_cancellation_model,
    tune_catboost_cancellation,
    tune_lgb_cancellation,
    tune_xgb_cancellation,
)
from src.cancellation_evaluation import evaluate_cross_dataset
from src.config import APP_CONFIG
from src.evaluation import (
    calibration_metrics,
    empirical_interval_coverage,
    forecast_error_breakdown,
    interval_coverage_by_horizon,
    regression_metrics,
    ts_cross_val_scores,
)
from src.forecasting import (
    blend_forecasts,
    find_ensemble_weights,
    forecast_prophet,
    forecast_xgb,
    forecast_xgb_direct,
    naive_baseline_forecast,
    train_prophet_forecaster,
    train_xgb_direct_multistep,
    train_xgb_forecaster,
    train_xgb_forecaster_with_intervals,
)
from src.conformal import forecast_conformal, split_conformal_calibrate
from src.ingestion import load_cancellation_prediction_csv
from src.trends import build_trends_features, TRENDS_FEATURE_COLS
from src.revenue_management import build_otb_feature_matrix
from src.model_specs import (
    CANCELLATION_MODEL_VERSION,
    CONFORMAL_INTERVAL_VERSION,
    DIRECT_FORECAST_MODEL_VERSION,
    ENSEMBLE_WEIGHTS_VERSION,
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
    compute_group_rates,
    drop_leaky_features,
    encode_categoricals,
    engineer_features,
    expanding_target_encode,
    load_data,
    target_encode_column,
)
from src.time_series import aggregate_daily, aggregate_daily_by_hotel_type, build_arrival_date

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


# Exogenous demand-feature columns that are genuinely known for FUTURE dates
# at forecast time: trends are published externally; OTB/pace describe bookings
# already on the books for future arrival dates. cancel_rate_lag7 is a lagged
# actual (only known 7 days ahead) and is deliberately excluded.
DEMAND_FUTURE_FEATURE_COLS = list(TRENDS_FEATURE_COLS) + [
    "otb_30d", "otb_60d", "otb_30d_stly", "otb_60d_stly",
    "pace_ratio_30d", "pace_ratio_60d",
]


def build_demand_extra_features(df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Demand exogenous features — cancel rate lag, Google Trends, OTB (30d/60d),
    STLY OTB, and pace ratios — aligned to ``daily_df.index``.

    Only otb_30d and otb_60d are included: for a 30-day-ahead forecast these
    are the only OTB snapshots that are available at prediction time without
    leaking future booking information. Reused by training, the eval block,
    and the rolling backtest CLI (scripts/backtest.py).
    """
    _cancel_rate_raw = daily_df["daily_cancellation_rate"]
    _cancel_rate_feat = pd.DataFrame(
        {"cancel_rate_lag7": _cancel_rate_raw.shift(7).fillna(0.0)},
        index=daily_df.index,
    )
    try:
        _trends_feat = build_trends_features(daily_df.index)
    except Exception as exc:
        log.warning("Google Trends unavailable, skipping trends features: %s", exc)
        _trends_feat = pd.DataFrame(0.0, index=daily_df.index, columns=TRENDS_FEATURE_COLS)

    try:
        _otb_raw = build_otb_feature_matrix(df, lead_times=[30, 60])
        _otb_feat = _otb_raw[["otb_30d", "otb_60d"]].reindex(daily_df.index, fill_value=0)
    except Exception as exc:
        log.warning("OTB feature build failed, skipping: %s", exc)
        _otb_raw = pd.DataFrame(0.0, index=daily_df.index, columns=["otb_30d", "otb_60d"])
        _otb_feat = _otb_raw.copy()

    # STLY OTB: shift the OTB index forward 364 days so each row maps to
    # "this year's equivalent arrival date", giving same-time-last-year counts
    # without a second full loop. Pace ratio = current / STLY (how far ahead/behind).
    _otb_stly = _otb_raw[["otb_30d", "otb_60d"]].copy()
    _otb_stly.index = _otb_stly.index + pd.Timedelta(days=364)
    _otb_stly.columns = ["otb_30d_stly", "otb_60d_stly"]
    _otb_stly = _otb_stly.reindex(daily_df.index, fill_value=0)

    _pace_feat = pd.DataFrame(index=daily_df.index)
    _pace_feat["pace_ratio_30d"] = _otb_feat["otb_30d"] / _otb_stly["otb_30d_stly"].clip(lower=1)
    _pace_feat["pace_ratio_60d"] = _otb_feat["otb_60d"] / _otb_stly["otb_60d_stly"].clip(lower=1)

    return pd.concat([_cancel_rate_feat, _trends_feat, _otb_feat, _otb_stly, _pace_feat], axis=1)


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

    demand_extra = build_demand_extra_features(df, daily_df)

    # ------------------------------------------------------------------
    # Revenue forecasting
    #
    # NOTE: extras like adr_lag1, OTB-STLY, and pace ratios were previously
    # passed via extra_features here, but no inference caller (the FastAPI
    # route nor the Streamlit fallback) supplies future_features for them.
    # That created a train/serve skew where every horizon row defaulted those
    # columns to 0.0 at inference. Train without extras until a future_features
    # supplier exists at every call site.
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

    if _artifact_current("xgb_revenue_conformal", CONFORMAL_INTERVAL_VERSION, models_dir):
        results["xgb_revenue_conformal"] = model_cache.load_artifact("xgb_revenue_conformal", models_dir=models_dir)
    else:
        rev_conformal = split_conformal_calibrate(
            rev_series, lags=FORECAST_LAGS, holdout_days=60, alpha=0.10,
        )
        results["xgb_revenue_conformal"] = rev_conformal
        _save_versioned("xgb_revenue_conformal", rev_conformal, CONFORMAL_INTERVAL_VERSION, models_dir)

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
        xgb_d, xgb_d_cols = train_xgb_forecaster(
            dem_series, lags=FORECAST_LAGS, extra_features=demand_extra
        )
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

    if _artifact_current("xgb_demand_conformal", CONFORMAL_INTERVAL_VERSION, models_dir):
        results["xgb_demand_conformal"] = model_cache.load_artifact("xgb_demand_conformal", models_dir=models_dir)
    else:
        # Demand model uses extra features (cancel rate lag, trends, OTB, pace).
        # Calibrate with those so train and serve match — though the conformal
        # holdout doesn't see future-only signals because the holdout period is
        # past data with known feature values.
        dem_conformal = split_conformal_calibrate(
            dem_series, lags=FORECAST_LAGS, holdout_days=60, alpha=0.10,
            extra_features=demand_extra,
        )
        results["xgb_demand_conformal"] = dem_conformal
        _save_versioned("xgb_demand_conformal", dem_conformal, CONFORMAL_INTERVAL_VERSION, models_dir)

    if _artifact_current("prophet_demand", PROPHET_MODEL_VERSION, models_dir):
        results["prophet_demand"] = model_cache.load_artifact("prophet_demand", models_dir=models_dir)
    else:
        prophet_dem = train_prophet_forecaster(dem_series)
        results["prophet_demand"] = prophet_dem
        _save_versioned("prophet_demand", prophet_dem, PROPHET_MODEL_VERSION, models_dir)

    # ------------------------------------------------------------------
    # Ensemble weights (XGB share of the XGB+Prophet blend), fit on a 60-day
    # validation window before a 30-day holdout — never the holdout itself.
    # Cached so API/serving contexts can blend without refitting a weight.
    # ------------------------------------------------------------------
    if _artifact_current("ensemble_weights", ENSEMBLE_WEIGHTS_VERSION, models_dir):
        results["ensemble_weights"] = model_cache.load_artifact("ensemble_weights", models_dir=models_dir)
    else:
        ensemble_weights = {"demand": 1.0, "revenue": 1.0}
        try:
            val_days, test_days_ew = 60, 30
            for label, series_ew, extra_ew in [
                ("demand", dem_series, demand_extra),
                ("revenue", rev_series, None),
            ]:
                if len(series_ew) <= val_days + test_days_ew + MIN_FORECAST_HISTORY:
                    continue
                val = series_ew.iloc[-(test_days_ew + val_days):-test_days_ew]
                val_train = series_ew.iloc[:-(test_days_ew + val_days)]
                inner_extra = extra_ew.reindex(val_train.index) if extra_ew is not None else None
                future_extra = (
                    extra_ew[DEMAND_FUTURE_FEATURE_COLS].reindex(val.index, fill_value=0)
                    if extra_ew is not None else None
                )
                xgb_m, xgb_c = train_xgb_forecaster(
                    val_train, lags=FORECAST_LAGS, extra_features=inner_extra)
                xgb_v = forecast_xgb(xgb_m, xgb_c, val_train, val_days,
                                     future_features=future_extra)
                prophet_v = forecast_prophet(train_prophet_forecaster(val_train), val_days)
                ensemble_weights[label] = find_ensemble_weights(
                    xgb_v.values, prophet_v["yhat"].values, val.values)
        except Exception as exc:
            log.warning("Ensemble weight computation failed: %s", exc)
        results["ensemble_weights"] = ensemble_weights
        _save_versioned("ensemble_weights", ensemble_weights, ENSEMBLE_WEIGHTS_VERSION, models_dir)

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

        # Time-based 80/20 split on the raw (pre-encoded) dataframe so we can
        # compute group rates and target encoding using training rows only.
        n_test = max(1, int(len(df_sorted) * 0.2))
        df_train_raw = df_sorted.iloc[:-n_test]
        df_full_raw = df_sorted  # both splits needed for group rate join

        # Step 1: Compute hotel / market_segment cancel rates on training rows only
        df_full_with_rates = compute_group_rates(df_train_raw, df_full_raw)

        # Step 2: Encode features — label-encode low-cardinality categoricals,
        # target-encode high-cardinality ones (country, agent) using training rows only.
        df_cancel = drop_leaky_features(df_full_with_rates)
        label_encoders = {}
        for col in _CATEGORICAL_COLS:
            if col in df_cancel.columns and col not in TARGET_ENCODED_COLS:
                le = LabelEncoder()
                df_cancel = df_cancel.copy()
                df_cancel[col] = le.fit_transform(df_cancel[col].astype(str))
                label_encoders[col] = le

        # Target-encode country and agent using training rows only
        global_cancel_mean = float(df_train_raw["is_canceled"].mean()) if "is_canceled" in df_train_raw.columns else 0.0
        target_encode_maps = {}
        df_cancel = df_cancel.copy()
        for col in TARGET_ENCODED_COLS:
            if col in df_cancel.columns and col in df_train_raw.columns and "is_canceled" in df_train_raw.columns:
                encoded = target_encode_column(df_train_raw, df_full_with_rates, col, global_mean=global_cancel_mean)
                df_cancel[col] = encoded.values
                # Store the rate map for inference-time use
                target_encode_maps[col] = df_train_raw.groupby(col)["is_canceled"].mean().to_dict()

        X, y_cancel, _ = encode_categoricals(drop_leaky_features(df_full_with_rates))

        # Re-apply target encoding to X (encode_categoricals re-runs label encoding).
        # TRAIN rows get a past-only smoothed *expanding* encoding — each row sees
        # only earlier rows' labels in its category, so a row never encodes its own
        # outcome, and the estimate tracks behaviour drift over 2015–2017. TEST rows
        # (and inference) use the static final train map, which stays the serving
        # contract that booking_risk_scorer / api consume unchanged.
        n_train = len(X) - n_test
        for col in TARGET_ENCODED_COLS:
            if col in X.columns and col in target_encode_maps:
                rate_map = target_encode_maps[col]
                original_col = df_full_with_rates[col] if col in df_full_with_rates.columns else None
                if original_col is None:
                    continue
                raw_vals = pd.Series(original_col.values, index=X.index)
                # Static map for all rows (test + fallback for unseen-in-train cats)
                static_enc = raw_vals.map(rate_map).fillna(global_cancel_mean).astype(float)
                # Expanding encoding for the train slice only
                train_raw = df_full_with_rates.iloc[:n_train]
                exp_enc = expanding_target_encode(
                    train_raw, col, global_mean=global_cancel_mean,
                )
                encoded = static_enc.copy()
                encoded.iloc[:n_train] = exp_enc.values
                X[col] = encoded.values

        feature_columns = list(X.columns)

        feature_defaults = {}
        for col in feature_columns:
            col_data = X[col].dropna()
            if col_data.dtype == float or col_data.nunique() > 20:
                feature_defaults[col] = float(col_data.median())
            else:
                feature_defaults[col] = float(col_data.mode().iloc[0])

        # Use the same n_test split on the encoded matrix
        X_tr = X.iloc[:-n_test]
        y_tr = y_cancel.iloc[:-n_test]
        X_te = X.iloc[-n_test:]
        y_te = y_cancel.iloc[-n_test:]

        spw = compute_scale_pos_weight(y_tr.values)

        # Tune all three gradient boosting models with TimeSeriesSplit
        log.info("Tuning XGBoost cancellation model...")
        best_xgb_params = tune_xgb_cancellation(X_tr.values, y_tr.values, scale_pos_weight=spw)
        log.info("Tuning LightGBM cancellation model...")
        best_lgb_params = tune_lgb_cancellation(X_tr.values, y_tr.values, scale_pos_weight=spw)

        log.info("Tuning CatBoost cancellation model...")
        best_cb_params = tune_catboost_cancellation(
            X_tr.values, y_tr.values,
            scale_pos_weight=spw,
            cat_feature_indices=[],
        )

        cancel_models = train_cancellation_models_balanced(
            X_tr.values,
            y_tr.values,
            scale_pos_weight=spw,
            xgb_params=best_xgb_params,
            lgb_params=best_lgb_params,
            catboost_params=best_cb_params,
            feature_columns=feature_columns,
        )

        # Build stacking meta-learner using out-of-fold predictions
        log.info("Training stacking meta-learner...")
        stacked_meta = train_stacked_cancellation_model(cancel_models, X_tr.values, y_tr.values)
        cancel_models["stacked_meta"] = stacked_meta

        # Fit isotonic calibrator on OOF stacked predictions so downstream
        # decisions (overbooking Poisson sum, alert thresholds) consume true
        # frequencies rather than raw stacker scores.
        log.info("Fitting probability calibrator on OOF stacked predictions...")
        try:
            calibrator, calib_oof = train_probability_calibrator(
                cancel_models, X_tr.values, y_tr.values
            )
        except Exception as exc:
            log.warning("Probability calibration failed, falling back to raw probs: %s", exc)
            calibrator, calib_oof = None, None
        cancel_models["probability_calibrator"] = calibrator

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Use XGBoost (still best single-model for SHAP interpretation) for feature importance
        try:
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
        except Exception as exc:
            log.warning("SHAP feature importance failed: %s", exc)
            selected_features = feature_columns

        # Choose the decision threshold on TRAINING-time out-of-fold predictions,
        # not the test holdout. Selecting on the same holdout used to report
        # accuracy biases the headline number and produces an unstable threshold
        # (it chases holdout noise). The OOF predictions are calibrated with the
        # same isotonic map production consumes, so the threshold matches serving.
        try:
            from sklearn.metrics import accuracy_score as _acc
            if calib_oof is not None and len(calib_oof) >= 10:
                oof_proba_cal = apply_probability_calibrator(
                    calibrator, calib_oof["proba_raw"].values
                )
                optimal_threshold, oof_acc = find_optimal_threshold(
                    calib_oof["y"].values, oof_proba_cal
                )
            else:
                optimal_threshold, oof_acc = 0.5, float("nan")
            cancel_models["optimal_threshold"] = optimal_threshold

            # Honest holdout accuracy: threshold fixed from OOF, evaluated on the
            # untouched test set. Also log the (biased) test-tuned accuracy once
            # for the record, to quantify the inflation that selection caused.
            stacked_proba_te = predict_stacked_proba(
                stacked_meta, cancel_models, X_te.values,
                calibrator=calibrator,
            )
            honest_acc = float(_acc(y_te.values, (stacked_proba_te >= optimal_threshold).astype(int)))
            _, biased_acc = find_optimal_threshold(y_te.values, stacked_proba_te)
            log.info(
                "Stacked (calibrated) — OOF-selected threshold: %.2f | "
                "OOF acc: %.3f | honest holdout acc: %.3f | "
                "(test-tuned acc would report %.3f — inflation %.3f)",
                optimal_threshold, oof_acc, honest_acc, biased_acc, biased_acc - honest_acc,
            )
        except Exception as exc:
            log.warning("Threshold selection failed: %s", exc)
            cancel_models["optimal_threshold"] = 0.5

        # Print per-model metrics on holdout
        log.info("Cancellation model holdout metrics:")
        for name, model in cancel_models.items():
            if name in ("stacked_meta", "optimal_threshold"):
                continue
            try:
                proba = model.predict_proba(X_te.values)[:, 1]
                thresh = float(cancel_models.get("optimal_threshold", 0.5))
                preds = (proba >= thresh).astype(int)
                from sklearn.metrics import accuracy_score as _acc, roc_auc_score as _auc
                log.info(
                    "  %-15s  Acc(thr=%.2f)=%.3f  ROC-AUC=%.3f",
                    name, thresh, _acc(y_te.values, preds), _auc(y_te.values, proba),
                )
            except Exception:
                pass
        try:
            stacked_proba_te = predict_stacked_proba(
                stacked_meta, cancel_models, X_te.values,
                calibrator=cancel_models.get("probability_calibrator"),
            )
            thresh = float(cancel_models.get("optimal_threshold", 0.5))
            stacked_preds = (stacked_proba_te >= thresh).astype(int)
            from sklearn.metrics import accuracy_score as _acc, roc_auc_score as _auc
            log.info(
                "  %-15s  Acc(thr=%.2f)=%.3f  ROC-AUC=%.3f",
                "stacked_cal", thresh,
                _acc(y_te.values, stacked_preds),
                _auc(y_te.values, stacked_proba_te),
            )
        except Exception as exc:
            log.warning("Stacked model eval failed: %s", exc)

        # Persist holdout metrics of the calibrated stacked classifier so drift
        # monitoring (src/monitoring.py) can snapshot them without re-running
        # the holdout evaluation.
        try:
            holdout_proba = predict_stacked_proba(
                stacked_meta, cancel_models, X_te.values,
                calibrator=cancel_models.get("probability_calibrator"),
            )
            thresh = float(cancel_models.get("optimal_threshold", 0.5))
            from sklearn.metrics import accuracy_score as _acc, roc_auc_score as _auc
            cal_diag = calibration_metrics(y_te.values, holdout_proba)
            cancel_models["holdout_metrics"] = {
                "accuracy": float(_acc(y_te.values, (holdout_proba >= thresh).astype(int))),
                "brier": float(cal_diag["brier"]),
                "ece": float(cal_diag["ece"]),
                "roc_auc": float(_auc(y_te.values, holdout_proba)),
                "threshold": thresh,
                "n_test": int(len(y_te)),
            }
        except Exception as exc:
            log.warning("Cancellation holdout metrics capture failed: %s", exc)
            cancel_models["holdout_metrics"] = None

        # Store target encoding maps alongside label encoders for inference
        cancel_models["target_encode_maps"] = target_encode_maps
        cancel_models["global_cancel_mean"] = global_cancel_mean

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
        # Train with cancel rate + trends + OTB; forecast with future values of
        # forward-looking signals (trends + OTB — historical values for the test
        # window, which is what would be available at real forecast time).
        dem_extra_train = demand_extra.iloc[:-test_days]
        dem_eval_xgb_m, dem_eval_xgb_c = train_xgb_forecaster(
            dem_eval_train, lags=FORECAST_LAGS, extra_features=dem_extra_train
        )
        # Forward-looking signals only (trends + OTB + pace) for the test window —
        # the columns genuinely known at real forecast time. cancel_rate_lag7 is
        # a lagged actual and is excluded.
        future_features_dem = demand_extra[DEMAND_FUTURE_FEATURE_COLS].reindex(
            dem_eval_test.index, fill_value=0
        )
        dem_xgb_preds = forecast_xgb(
            dem_eval_xgb_m, dem_eval_xgb_c, dem_eval_train, test_days,
            future_features=future_features_dem,
        )
        dem_xgb_preds.index = dem_eval_test.index
        dem_naive = naive_baseline_forecast(dem_eval_train, test_days)
        dem_naive.index = dem_eval_test.index
        dem_eval_prophet = train_prophet_forecaster(dem_eval_train)
        dem_prophet_df = forecast_prophet(dem_eval_prophet, test_days)

        # ----------------------------------------------------------------
        # Candidate models that were previously UI-only: XGB+Prophet ensemble
        # and direct multi-step. The ensemble weight is fit on the 60 days
        # before the test window (never the test window itself) so this metric
        # is honest — unlike the old app.py display weight that fit on the test
        # holdout. These rows let the Validation page and drift snapshots show
        # whether the ensemble/direct variants actually beat plain recursive XGB.
        # ----------------------------------------------------------------
        eval_ensemble = {}
        for label, e_train, e_test, e_xgb_preds, e_prophet_df in [
            ("revenue", rev_eval_train, rev_eval_test, rev_xgb_preds, rev_prophet_df),
            ("demand", dem_eval_train, dem_eval_test, dem_xgb_preds, dem_prophet_df),
        ]:
            try:
                w_val_days = 60
                if len(e_train) > w_val_days + MIN_FORECAST_HISTORY:
                    w_inner = e_train.iloc[:-w_val_days]
                    w_val = e_train.iloc[-w_val_days:]
                    if label == "demand":
                        w_inner_extra = demand_extra.reindex(w_inner.index)
                        w_future = demand_extra[DEMAND_FUTURE_FEATURE_COLS].reindex(
                            w_val.index, fill_value=0)
                    else:
                        w_inner_extra, w_future = None, None
                    w_xgb_m, w_xgb_c = train_xgb_forecaster(
                        w_inner, lags=FORECAST_LAGS, extra_features=w_inner_extra)
                    w_xgb = forecast_xgb(w_xgb_m, w_xgb_c, w_inner, w_val_days,
                                         future_features=w_future)
                    w_prophet = forecast_prophet(train_prophet_forecaster(w_inner), w_val_days)
                    weight = find_ensemble_weights(
                        w_xgb.values, w_prophet["yhat"].values, w_val.values)
                else:
                    weight = 1.0
                ens_preds = blend_forecasts(e_xgb_preds, e_prophet_df, weight)
                eval_ensemble[f"{label}_ensemble"] = {
                    **regression_metrics(e_test.values, ens_preds.values),
                    "xgb_weight": float(weight),
                }
            except Exception as exc:
                log.warning("Ensemble eval failed for %s: %s", label, exc)

        eval_direct = {}
        for label, e_train, e_test in [
            ("revenue", rev_eval_train, rev_eval_test),
            ("demand", dem_eval_train, dem_eval_test),
        ]:
            try:
                direct_models = train_xgb_direct_multistep(
                    e_train, horizons=test_days, lags=FORECAST_LAGS)
                direct_preds = forecast_xgb_direct(direct_models, e_train)
                eval_direct[f"{label}_xgb_direct"] = regression_metrics(
                    e_test.values, np.asarray(direct_preds)[:test_days])
            except Exception as exc:
                log.warning("Direct multi-step eval failed for %s: %s", label, exc)

        eval_metrics = {
            "revenue_xgb": regression_metrics(rev_eval_test.values, rev_xgb_preds.values),
            "revenue_prophet": regression_metrics(rev_eval_test.values, rev_prophet_df["yhat"].values),
            "revenue_naive": regression_metrics(rev_eval_test.values, rev_naive.values),
            "demand_xgb": regression_metrics(dem_eval_test.values, dem_xgb_preds.values),
            "demand_prophet": regression_metrics(dem_eval_test.values, dem_prophet_df["yhat"].values),
            "demand_naive": regression_metrics(dem_eval_test.values, dem_naive.values),
            **eval_ensemble,
            **eval_direct,
            "eval_note": (
                f"Holdout: last {test_days} days (unseen during training). "
                "Production models are retrained on the full dataset for live forecasts."
            ),
            # Rolling CV stability across 4 folds
            "cv_scores": ts_cross_val_scores(rev_series, n_splits=4, forecast_horizon=30),
            "cv_demand": ts_cross_val_scores(dem_series, n_splits=4, forecast_horizon=30),
            # Per-weekday and per-month MAPE breakdowns
            "demand_xgb_breakdown": forecast_error_breakdown(
                dem_eval_test.values, dem_xgb_preds.values, dem_eval_test.index
            ),
            "revenue_xgb_breakdown": forecast_error_breakdown(
                rev_eval_test.values, rev_xgb_preds.values, rev_eval_test.index
            ),
        }

        # Empirical interval coverage on the same 30-day holdout: calibrate a
        # conformal bundle on data up to the holdout, then evaluate band
        # coverage on the held-out window. Captures the gap between the 90%
        # nominal level and the actual rate of containment.
        try:
            rev_cal_bundle = split_conformal_calibrate(
                rev_eval_train, lags=FORECAST_LAGS, holdout_days=60, alpha=0.10,
            )
            rev_band = forecast_conformal(rev_cal_bundle, rev_eval_train, periods=test_days)
            eval_metrics["revenue_conformal_coverage"] = {
                **empirical_interval_coverage(
                    rev_eval_test.values,
                    rev_band["yhat_lower"].values,
                    rev_band["yhat_upper"].values,
                ),
                "alpha": float(rev_cal_bundle["alpha"]),
                "nominal_coverage": 1.0 - float(rev_cal_bundle["alpha"]),
                "by_bin": interval_coverage_by_horizon(
                    rev_eval_test.values,
                    rev_band["yhat_lower"].values,
                    rev_band["yhat_upper"].values,
                )["by_bin"],
            }
        except Exception as exc:
            log.warning("Revenue conformal coverage eval failed: %s", exc)

        try:
            dem_cal_bundle = split_conformal_calibrate(
                dem_eval_train, lags=FORECAST_LAGS, holdout_days=60, alpha=0.10,
                extra_features=dem_extra_train,
            )
            dem_band = forecast_conformal(
                dem_cal_bundle, dem_eval_train, periods=test_days,
                future_features=future_features_dem,
            )
            eval_metrics["demand_conformal_coverage"] = {
                **empirical_interval_coverage(
                    dem_eval_test.values,
                    dem_band["yhat_lower"].values,
                    dem_band["yhat_upper"].values,
                ),
                "alpha": float(dem_cal_bundle["alpha"]),
                "nominal_coverage": 1.0 - float(dem_cal_bundle["alpha"]),
                "by_bin": interval_coverage_by_horizon(
                    dem_eval_test.values,
                    dem_band["yhat_lower"].values,
                    dem_band["yhat_upper"].values,
                )["by_bin"],
            }
        except Exception as exc:
            log.warning("Demand conformal coverage eval failed: %s", exc)

        # Per-hotel-type MAPE (fixes Validation page placeholders)
        try:
            hotel_daily = aggregate_daily_by_hotel_type(df)
            for hotel_key, h_daily in hotel_daily.items():
                h_key = hotel_key.lower().replace(" ", "_")
                for series_col, metric_prefix in [
                    ("daily_bookings", f"demand_xgb_{h_key}"),
                    ("daily_booked_revenue", f"revenue_booked_xgb_{h_key}"),
                    ("daily_realized_revenue", f"revenue_realized_xgb_{h_key}"),
                ]:
                    if series_col not in h_daily.columns or len(h_daily) < test_days + 30:
                        continue
                    h_series = h_daily[series_col]
                    h_train = h_series.iloc[:-test_days]
                    h_test = h_series.iloc[-test_days:]
                    try:
                        h_m, h_c = train_xgb_forecaster(h_train, lags=FORECAST_LAGS)
                        h_preds = forecast_xgb(h_m, h_c, h_train, test_days)
                        h_preds.index = h_test.index
                        eval_metrics[metric_prefix] = regression_metrics(h_test.values, h_preds.values)
                    except Exception as exc:
                        log.warning("Per-hotel eval failed for %s/%s: %s", hotel_key, series_col, exc)
        except Exception as exc:
            log.warning("Per-hotel-type evaluation failed: %s", exc)

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
