"""
Phase 1 evaluation: report Brier / ECE for the cancellation model and
empirical conformal coverage for revenue & demand forecasts on a real
chronological holdout.

The conformal section is an ablation of horizon binning: the same adaptive
bundle is rendered as a flat band (pooled global q — the v1 construction with
a better-estimated quantile) and as a horizon-binned band, with per-bin
coverage and width side by side.

Run:
    python scripts/eval_phase1.py                      # full (slow: tunes cancellation)
    python scripts/eval_phase1.py --skip-cancellation  # conformal ablation only (~1 min)
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.preprocessing import (
    load_data, clean_data, engineer_features, encode_categoricals,
    drop_leaky_features, compute_group_rates,
)
from src.time_series import build_arrival_date, aggregate_daily
from src.cancellation import (
    TARGET_ENCODED_COLS,
    compute_scale_pos_weight,
    find_optimal_threshold,
    train_probability_calibrator,
    apply_probability_calibrator,
    predict_stacked_proba,
    train_cancellation_models_balanced,
    train_stacked_cancellation_model,
    tune_xgb_cancellation,
    tune_lgb_cancellation,
    tune_catboost_cancellation,
)
from src.conformal import split_conformal_calibrate, forecast_conformal
from src.evaluation import (
    calibration_metrics,
    classification_report_dict,
    empirical_interval_coverage,
    interval_coverage_by_horizon,
    regression_metrics,
)

DATA_PATH = "hotel_bookings.csv"


def evaluate_cancellation() -> dict:
    print("\n" + "=" * 60)
    print("CANCELLATION CALIBRATION (Phase 1)")
    print("=" * 60)

    df = engineer_features(clean_data(load_data(DATA_PATH)))
    df = build_arrival_date(df)
    df_sorted = df.sort_values("arrival_date")
    n_test = max(1, int(len(df_sorted) * 0.2))
    df_train_raw = df_sorted.iloc[:-n_test]
    df_with_rates = compute_group_rates(df_train_raw, df_sorted)

    X, y, _ = encode_categoricals(drop_leaky_features(df_with_rates))
    global_cancel_mean = float(df_train_raw["is_canceled"].mean())
    for col in TARGET_ENCODED_COLS:
        if col in X.columns and col in df_train_raw.columns:
            rate_map = df_train_raw.groupby(col)["is_canceled"].mean().to_dict()
            if col in df_with_rates.columns:
                X[col] = df_with_rates[col].values
                X[col] = X[col].map(rate_map).fillna(global_cancel_mean).astype(float)

    X_tr, y_tr = X.iloc[:-n_test], y.iloc[:-n_test]
    X_te, y_te = X.iloc[-n_test:], y.iloc[-n_test:]
    feature_columns = list(X.columns)

    spw = compute_scale_pos_weight(y_tr.values)
    print(f"  Training rows: {len(X_tr):,}  Test rows: {len(X_te):,}")
    print("  Tuning XGB / LGB / CatBoost ...")
    xgb_p = tune_xgb_cancellation(X_tr.values, y_tr.values, scale_pos_weight=spw)
    lgb_p = tune_lgb_cancellation(X_tr.values, y_tr.values, scale_pos_weight=spw)
    cb_p = tune_catboost_cancellation(
        X_tr.values, y_tr.values, scale_pos_weight=spw, cat_feature_indices=[]
    )
    models = train_cancellation_models_balanced(
        X_tr.values, y_tr.values, scale_pos_weight=spw,
        xgb_params=xgb_p, lgb_params=lgb_p, catboost_params=cb_p,
        feature_columns=feature_columns,
    )
    stacked_meta = train_stacked_cancellation_model(models, X_tr.values, y_tr.values)
    models["stacked_meta"] = stacked_meta

    # Uncalibrated stacked probabilities on the holdout
    proba_uncal = predict_stacked_proba(stacked_meta, models, X_te.values)

    # Fit isotonic calibrator on time-series OOF training predictions, then
    # apply to holdout probabilities.
    calibrator, _calib_oof = train_probability_calibrator(models, X_tr.values, y_tr.values)
    proba_cal = (apply_probability_calibrator(calibrator, proba_uncal)
                 if calibrator is not None else proba_uncal)
    if calibrator is None:
        print("  [warn] calibrator could not be fit (insufficient OOF data)")

    thr_uncal, _ = find_optimal_threshold(y_te.values, proba_uncal)
    thr_cal, _ = find_optimal_threshold(y_te.values, proba_cal)

    cls_uncal = classification_report_dict(
        y_te.values, (proba_uncal >= thr_uncal).astype(int), proba_uncal
    )
    cls_cal = classification_report_dict(
        y_te.values, (proba_cal >= thr_cal).astype(int), proba_cal
    )
    cal_uncal = calibration_metrics(y_te.values, proba_uncal)
    cal_cal = calibration_metrics(y_te.values, proba_cal)

    print("\n  Stacked classifier on holdout:")
    print(f"    {'Metric':<14s} {'Uncalibrated':>14s} {'Calibrated':>14s} {'Delta':>10s}")
    for k, fmt in [("accuracy", ".3f"), ("precision", ".3f"), ("recall", ".3f"),
                   ("f1", ".3f"), ("roc_auc", ".3f")]:
        u, c = cls_uncal[k], cls_cal[k]
        print(f"    {k:<14s} {u:>14{fmt}} {c:>14{fmt}} {c - u:>+10{fmt}}")
    print(f"    {'brier':<14s} {cal_uncal['brier']:>14.4f} {cal_cal['brier']:>14.4f} "
          f"{cal_cal['brier'] - cal_uncal['brier']:>+10.4f}  (lower is better)")
    print(f"    {'ece':<14s} {cal_uncal['ece']:>14.4f} {cal_cal['ece']:>14.4f} "
          f"{cal_cal['ece'] - cal_uncal['ece']:>+10.4f}  (lower is better)")
    print(f"    {'threshold':<14s} {thr_uncal:>14.3f} {thr_cal:>14.3f}")

    return {
        "uncal": {**cls_uncal, **cal_uncal, "threshold": float(thr_uncal)},
        "cal":   {**cls_cal,   **cal_cal,   "threshold": float(thr_cal)},
    }


def evaluate_conformal(daily: pd.Series, label: str, *, alpha: float = 0.10,
                       test_days: int = 60) -> dict:
    """Calibrate one adaptive bundle on an internal split, then compare flat
    (pooled q, v1-equivalent construction) vs horizon-binned bands on a final
    holdout the calibration never saw."""
    n = len(daily)
    cal_series = daily.iloc[: n - test_days]
    test_series = daily.iloc[n - test_days:]

    bundle = split_conformal_calibrate(cal_series, alpha=alpha)
    fc_flat = forecast_conformal(
        bundle, cal_series, periods=test_days, use_horizon_bins=False,
    ).set_index("ds")
    fc_binned = forecast_conformal(
        bundle, cal_series, periods=test_days, use_horizon_bins=True,
    ).set_index("ds")
    test_aligned = test_series.reindex(fc_flat.index)

    reg = regression_metrics(test_aligned.values, fc_flat["yhat"].values)
    bin_edges = bundle.get("bin_edges") or [7, 14, 30]
    cov_flat = interval_coverage_by_horizon(
        test_aligned.values, fc_flat["yhat_lower"].values,
        fc_flat["yhat_upper"].values, bin_edges,
    )
    cov_binned = interval_coverage_by_horizon(
        test_aligned.values, fc_binned["yhat_lower"].values,
        fc_binned["yhat_upper"].values, bin_edges,
    )

    nominal = 1.0 - alpha
    print(f"\n  {label} -- {test_days} day holdout, nominal coverage {nominal:.0%}")
    print(f"    MAE  : {reg['mae']:>10.2f}")
    print(f"    RMSE : {reg['rmse']:>10.2f}")
    print(f"    MAPE : {reg['mape']:>9.1f}%")
    print(f"    Calibration: method={bundle['method']}  folds={bundle['n_folds_used']}"
          f"  total cal days={bundle['holdout_days']}  fallback={bundle['fallback']}")
    if bundle.get("weekend_weekday_ratio") is not None:
        print(f"    Weekend/weekday |residual| ratio: {bundle['weekend_weekday_ratio']:.2f}")
    print(f"    Pooled (flat) q: {bundle['conformal_q']:.2f}")
    print(f"    {'bin':<8s} {'n_cal':>6s} {'q':>10s} {'cov_flat':>9s} {'cov_bin':>9s}"
          f" {'w_flat':>10s} {'w_bin':>10s}")
    bin_q = bundle.get("bin_q") or [bundle["conformal_q"]] * len(bin_edges)
    bin_counts = bundle.get("bin_counts") or [0] * len(bin_edges)
    for (blabel, cf), (_, cb), q, n_cal in zip(
        cov_flat["by_bin"].items(), cov_binned["by_bin"].items(), bin_q, bin_counts,
    ):
        print(f"    {blabel:<8s} {n_cal:>6d} {q:>10.2f} {cf['coverage']:>9.0%}"
              f" {cb['coverage']:>9.0%} {cf['mean_width']:>10.2f} {cb['mean_width']:>10.2f}")
    of, ob = cov_flat["overall"], cov_binned["overall"]
    print(f"    {'overall':<8s} {sum(bin_counts):>6d} {'':>10s} {of['coverage']:>9.0%}"
          f" {ob['coverage']:>9.0%} {of['mean_width']:>10.2f} {ob['mean_width']:>10.2f}")
    return {
        "reg": reg,
        "coverage_flat": cov_flat,
        "coverage_binned": cov_binned,
        "q": float(bundle["conformal_q"]),
        "bin_q": bin_q,
        "bin_counts": bin_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-cancellation", action="store_true",
        help="Skip the slow cancellation tuning; run only the conformal ablation.",
    )
    args = parser.parse_args()

    if not args.skip_cancellation:
        evaluate_cancellation()

    print("\n" + "=" * 60)
    print("CONFORMAL FORECAST INTERVALS (flat vs horizon-binned)")
    print("=" * 60)

    df = engineer_features(clean_data(load_data(DATA_PATH)))
    df = build_arrival_date(df)
    daily = aggregate_daily(df)

    rev = evaluate_conformal(daily["daily_revenue"], "Revenue")
    dem = evaluate_conformal(daily["daily_bookings"], "Demand ")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
