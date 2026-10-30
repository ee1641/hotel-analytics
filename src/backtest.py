"""
Rolling-origin backtest harness for forecast model comparison.

Single-holdout numbers on this dataset have proven misleading (the final
30-day window is the easiest in the series: Aug 2017, peak volume). This
module evaluates every candidate model on K rolling windows stepping back
through history so that claims like "X% MAPE" or "model A beats model B"
rest on a distribution of windows, not one lucky draw.

Candidates are thin wrappers around the existing training/forecast functions
in ``src/forecasting.py`` — no new model code lives here. Each candidate is a
callable ``(train_series, horizon, *, extras, future_extras, holiday_country,
lags) -> pd.Series`` and only ever sees data before the window origin.

Run via ``python scripts/backtest.py`` (see docs/retraining.md for context).
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from src.evaluation import forecast_error_breakdown, regression_metrics
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
)
from src.model_specs import FORECAST_LAGS, MIN_FORECAST_HISTORY

log = logging.getLogger(__name__)

_DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

CandidateFn = Callable[..., pd.Series]


# ---------------------------------------------------------------------------
# Built-in candidates
# ---------------------------------------------------------------------------

def xgb_recursive_candidate(
    train: pd.Series, horizon: int, *,
    extras=None, future_extras=None, holiday_country=None, lags=None,
) -> pd.Series:
    model, feat_cols = train_xgb_forecaster(
        train, lags=lags, extra_features=extras, holiday_country=holiday_country,
    )
    return forecast_xgb(
        model, feat_cols, train, periods=horizon,
        future_features=future_extras, holiday_country=holiday_country,
    )


def xgb_direct_candidate(
    train: pd.Series, horizon: int, *,
    extras=None, future_extras=None, holiday_country=None, lags=None,
) -> pd.Series:
    models = train_xgb_direct_multistep(
        train, horizons=horizon, lags=lags, holiday_country=holiday_country,
    )
    return forecast_xgb_direct(models, train, holiday_country=holiday_country)


def prophet_candidate(
    train: pd.Series, horizon: int, *,
    extras=None, future_extras=None, holiday_country=None, lags=None,
) -> pd.Series:
    model = train_prophet_forecaster(train, holiday_country=holiday_country)
    df = forecast_prophet(model, periods=horizon)
    return pd.Series(df["yhat"].values, index=pd.DatetimeIndex(df["ds"]))


def naive_candidate(
    train: pd.Series, horizon: int, *,
    extras=None, future_extras=None, holiday_country=None, lags=None,
) -> pd.Series:
    return naive_baseline_forecast(train, periods=horizon)


def ensemble_candidate(
    train: pd.Series, horizon: int, *,
    extras=None, future_extras=None, holiday_country=None, lags=None,
    val_days: int = 60,
) -> pd.Series:
    """
    XGB+Prophet blend. The blend weight is chosen on the last ``val_days`` of
    the TRAINING window (never the test window), then both models refit on the
    full training window — mirrors the production ensemble_weights artifact.
    """
    lags = list(lags) if lags is not None else list(FORECAST_LAGS)
    min_train = max(MIN_FORECAST_HISTORY, max(lags) + 1)

    weight = 1.0
    use_val = min(val_days, max(len(train) - min_train, 0))
    if use_val >= 14:  # need a meaningful validation window to fit a weight
        inner_train = train.iloc[:-use_val]
        val = train.iloc[-use_val:]
        inner_extras = extras.iloc[:-use_val] if extras is not None else None
        val_extras = (
            future_extras_for_index(extras, val.index, future_extras)
            if extras is not None else None
        )
        xgb_m, xgb_c = train_xgb_forecaster(
            inner_train, lags=lags, extra_features=inner_extras,
            holiday_country=holiday_country,
        )
        xgb_val = forecast_xgb(
            xgb_m, xgb_c, inner_train, periods=use_val,
            future_features=val_extras, holiday_country=holiday_country,
        )
        prophet_val = forecast_prophet(
            train_prophet_forecaster(inner_train, holiday_country=holiday_country),
            periods=use_val,
        )
        weight = find_ensemble_weights(
            xgb_val.values, prophet_val["yhat"].values, val.values,
        )

    xgb_m, xgb_c = train_xgb_forecaster(
        train, lags=lags, extra_features=extras, holiday_country=holiday_country,
    )
    xgb_fc = forecast_xgb(
        xgb_m, xgb_c, train, periods=horizon,
        future_features=future_extras, holiday_country=holiday_country,
    )
    prophet_fc = forecast_prophet(
        train_prophet_forecaster(train, holiday_country=holiday_country),
        periods=horizon,
    )
    return blend_forecasts(xgb_fc, prophet_fc, weight)


def future_extras_for_index(
    extras: pd.DataFrame | None,
    index: pd.DatetimeIndex,
    template: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """
    Slice exogenous features for a future window, keeping only the columns the
    caller marked as future-known (the columns of ``template``). Used so the
    ensemble's internal validation forecast sees the same feature availability
    as the outer test forecast.
    """
    if extras is None:
        return None
    cols = list(template.columns) if template is not None else list(extras.columns)
    cols = [c for c in cols if c in extras.columns]
    return extras[cols].reindex(index)


DEFAULT_CANDIDATES: dict[str, CandidateFn] = {
    "xgb_recursive": xgb_recursive_candidate,
    "xgb_direct": xgb_direct_candidate,
    "prophet": prophet_candidate,
    "naive": naive_candidate,
    "ensemble": ensemble_candidate,
}


def default_candidates(names: list[str] | None = None) -> dict[str, CandidateFn]:
    if names is None:
        return dict(DEFAULT_CANDIDATES)
    unknown = [n for n in names if n not in DEFAULT_CANDIDATES]
    if unknown:
        raise ValueError(
            f"Unknown candidates: {unknown}. Available: {sorted(DEFAULT_CANDIDATES)}"
        )
    return {n: DEFAULT_CANDIDATES[n] for n in names}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def rolling_backtest(
    series: pd.Series,
    *,
    candidates: dict[str, CandidateFn] | None = None,
    n_windows: int = 12,
    horizon: int = 30,
    step: int = 30,
    extra_features: pd.DataFrame | None = None,
    future_feature_cols: list[str] | None = None,
    holiday_country: str | None = None,
    lags: list | None = None,
) -> pd.DataFrame:
    """
    Evaluate candidate forecasters on rolling-origin windows.

    Window i (newest first): origin = n - horizon - i*step. Each candidate
    trains on ``series[:origin]`` and forecasts ``horizon`` days; windows are
    skipped (auto-shrink) when the training slice would fall below
    ``max(MIN_FORECAST_HISTORY, max(lags)+1)``.

    ``future_feature_cols`` restricts which exogenous columns are passed as
    future features (columns genuinely known ahead of time, e.g. trends/OTB —
    NOT lagged-actual columns like cancel_rate_lag7). None = all columns.

    Returns a long-form DataFrame: one row per (window, candidate) with
    mae/rmse/mape and per-weekday MAPE columns (mape_Mon..mape_Sun).
    """
    series = pd.Series(series).sort_index()
    candidates = candidates if candidates is not None else default_candidates()
    lags = list(lags) if lags is not None else list(FORECAST_LAGS)
    n = len(series)

    min_train = max(MIN_FORECAST_HISTORY, max(lags) + 1)
    if n - horizon >= min_train:
        k_max = (n - horizon - min_train) // step + 1
    else:
        k_max = 0
    n_used = min(int(n_windows), max(int(k_max), 0))
    if n_used <= 0:
        raise ValueError(
            f"Series too short for backtest: n={n}, horizon={horizon}, "
            f"min_train={min_train}."
        )
    if n_used < n_windows:
        log.warning("Backtest auto-shrink: %d windows requested, %d feasible.",
                    n_windows, n_used)

    rows = []
    for i in range(n_used):
        origin = n - horizon - i * step
        train = series.iloc[:origin]
        test = series.iloc[origin:origin + horizon]

        train_extras = extra_features.iloc[:origin] if extra_features is not None else None
        if extra_features is not None:
            cols = future_feature_cols or list(extra_features.columns)
            cols = [c for c in cols if c in extra_features.columns]
            future_extras = extra_features[cols].reindex(test.index)
        else:
            future_extras = None

        for name, fn in candidates.items():
            try:
                preds = fn(
                    train, horizon,
                    extras=train_extras, future_extras=future_extras,
                    holiday_country=holiday_country, lags=lags,
                )
                values = np.asarray(preds, dtype=float)[:horizon]
                metrics = regression_metrics(test.values, values)
                breakdown = forecast_error_breakdown(test.values, values, test.index)
                dow_mape = breakdown["by_weekday"]["mape"]
                row = {
                    "window": i,
                    "origin_date": train.index[-1].date(),
                    "candidate": name,
                    **metrics,
                }
                for day in _DAY_NAMES:
                    row[f"mape_{day}"] = (
                        float(dow_mape[day]) if day in dow_mape.index else float("nan")
                    )
                rows.append(row)
            except Exception as exc:
                log.warning("Backtest candidate '%s' failed on window %d: %s",
                            name, i, exc)

    return pd.DataFrame(rows)


def summarize_backtest(results: pd.DataFrame) -> pd.DataFrame:
    """
    Per-candidate aggregate: windows, mean/std/median MAPE, mean MAE,
    per-weekday mean MAPE, and head-to-head wins (lowest MAPE per window).
    Sorted by mean MAPE.
    """
    if results.empty:
        return pd.DataFrame()
    summary = results.groupby("candidate").agg(
        windows=("window", "count"),
        mape_mean=("mape", "mean"),
        mape_std=("mape", "std"),
        mape_median=("mape", "median"),
        mae_mean=("mae", "mean"),
    )
    wins = (
        results.loc[results.groupby("window")["mape"].idxmin(), "candidate"]
        .value_counts()
    )
    summary["wins"] = wins.reindex(summary.index).fillna(0).astype(int)
    for day in _DAY_NAMES:
        col = f"mape_{day}"
        if col in results.columns:
            summary[col] = results.groupby("candidate")[col].mean()
    return summary.sort_values("mape_mean").round(2)
