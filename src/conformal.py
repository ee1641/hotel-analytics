"""
Split-conformal prediction intervals for XGBoost forecasters.

Calibration is rolling-origin: K calibration folds step back through history,
each training a point model on ``series[:origin]`` and forecasting H days with
the same one-step-iterative path used at inference time. Pooling residuals
across folds gives every horizon bin enough samples for a stable quantile
(K=10, H=30 → 70/70/160 residuals for bins 1-7 / 8-14 / 15-30, versus a single
residual per horizon under the old one-holdout scheme).

Per-bin ``(1 - alpha)`` quantiles of ``|residual|`` become horizon-dependent
half-widths: tight bands where the model is accurate (short horizons), wider
bands where iterative error accumulates (long horizons). Bins with too few
residuals for a valid conformal quantile fall back to the pooled global
quantile, as do legacy bundles pickled before binning existed.

Under exchangeability the resulting intervals have marginal coverage of at
least ``1 - alpha``. The quantile-loss XGB intervals in ``forecasting.py``
optimise a different objective and have no coverage guarantee — this module
is the upgrade path.

After calibration the point model is **refit on the full series** so the
production forecaster sees every observation; only the residual quantiles come
from held-out windows. That is the "split" part of split-conformal.
"""
from __future__ import annotations

import logging
from bisect import bisect_left

import numpy as np
import pandas as pd

from src.forecasting import (
    forecast_xgb,
    train_xgb_forecaster,
)
from src.model_specs import FORECAST_LAGS, MIN_FORECAST_HISTORY

log = logging.getLogger(__name__)


def _finite_sample_quantile(abs_resid: np.ndarray, alpha: float) -> float:
    """Conformal quantile with finite-sample correction: ceil((n+1)(1-alpha))-th order statistic."""
    n = len(abs_resid)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(np.sort(abs_resid)[k - 1])


def _bin_index(horizon: int, bin_edges) -> int:
    """Map a 1-indexed horizon step to its bin; steps beyond the last edge use the last bin."""
    return min(bisect_left(list(bin_edges), horizon), len(bin_edges) - 1)


def split_conformal_calibrate(
    series: pd.Series,
    *,
    lags: list[int] | None = None,
    holdout_days: int = 60,
    alpha: float = 0.10,
    extra_features: pd.DataFrame | None = None,
    holiday_country: str | None = None,
    log_target: bool = True,
    n_folds: int = 10,
    fold_horizon: int = 30,
    fold_step: int = 30,
    bin_edges: tuple[int, ...] = (7, 14, 30),
) -> dict:
    """
    Fit a point model + horizon-binned conformity scores on a daily series.

    Parameters
    ----------
    series : daily target (revenue or demand), sorted by date.
    lags : forecast lags. Defaults to FORECAST_LAGS.
    holdout_days : deprecated (kept for call-site compatibility). Calibration
        is now rolling-origin; the returned ``holdout_days`` reports the total
        calibration days actually used (``n_folds_used * fold_horizon``).
    alpha : 1 - nominal coverage (0.10 → 90% intervals).
    extra_features : optional exogenous feature DataFrame, same index as series.
    holiday_country : ISO country code for holiday features.
    log_target : whether to log1p-transform the target during fitting.
    n_folds : max rolling-origin calibration folds; auto-shrinks on short series.
    fold_horizon : days forecast per fold (matches the production horizon).
    fold_step : days between fold origins; equal to fold_horizon → disjoint
        calibration windows, no residual double-counting.
    bin_edges : inclusive upper edges of the horizon bins, e.g. (7, 14, 30)
        → bins 1-7, 8-14, 15-30.

    Returns
    -------
    dict with keys:
        point_model    : fitted XGBRegressor (trained on full series)
        feat_cols      : feature column order for the point model
        conformal_q    : float, pooled global half-width (fallback)
        bin_edges      : list[int]
        bin_q          : list[float], per-bin half-widths ([] on fallback)
        bin_counts     : list[int], calibration residuals per bin
        method         : "rolling_origin_v2" | "gaussian_fallback"
        n_folds_used   : int
        fold_horizon   : int
        fold_step      : int
        weekend_weekday_ratio : float | None, median weekend / weekday |resid|
        alpha          : float
        holdout_days   : int, total calibration days used
        residual_std   : float, diagnostic
        log_target     : bool
        fallback       : bool, True when history was too short for rolling-
                         origin calibration and a Gaussian fallback was used.
    """
    series = pd.Series(series).sort_index()
    lags = list(lags) if lags is not None else list(FORECAST_LAGS)
    bin_edges = [int(e) for e in bin_edges]
    n = len(series)

    min_train = max(MIN_FORECAST_HISTORY, max(lags) + 1)
    if n - fold_horizon >= min_train:
        k_max = (n - fold_horizon - min_train) // fold_step + 1
    else:
        k_max = 0
    n_folds_used = min(int(n_folds), max(int(k_max), 0))

    if n_folds_used <= 0:
        log.warning(
            "Conformal calibration: history too short (%d) for fold_horizon=%d "
            "with min_train=%d; using Gaussian fallback.", n, fold_horizon, min_train,
        )
        # Train on full series and approximate q from the series' own spread
        # (optimistic, but better than no interval at all).
        model, feat_cols = train_xgb_forecaster(
            series, lags=lags,
            extra_features=extra_features,
            holiday_country=holiday_country,
            log_target=log_target,
        )
        residual_std = float(series.std(ddof=1) or 1.0)
        z = 1.6449 if alpha <= 0.10 else 1.96  # rough 90% / 95%
        return {
            "point_model": model,
            "feat_cols": feat_cols,
            "conformal_q": float(z * residual_std),
            "bin_edges": bin_edges,
            "bin_q": [],
            "bin_counts": [],
            "method": "gaussian_fallback",
            "n_folds_used": 0,
            "fold_horizon": int(fold_horizon),
            "fold_step": int(fold_step),
            "weekend_weekday_ratio": None,
            "alpha": float(alpha),
            "holdout_days": 0,
            "residual_std": residual_std,
            "log_target": bool(log_target),
            "fallback": True,
        }

    horizons: list[int] = []
    abs_resids: list[float] = []
    signed_resids: list[float] = []
    dows: list[int] = []

    for i in range(n_folds_used):
        origin = n - fold_horizon - i * fold_step
        train_series = series.iloc[:origin]
        holdout_series = series.iloc[origin:origin + fold_horizon]

        cal_extra = extra_features.iloc[:origin] if extra_features is not None else None
        cal_model, cal_feat_cols = train_xgb_forecaster(
            train_series, lags=lags,
            extra_features=cal_extra,
            holiday_country=holiday_country,
            log_target=log_target,
        )

        future_extra = (
            extra_features.reindex(holdout_series.index)
            if extra_features is not None else None
        )
        preds = forecast_xgb(
            cal_model, cal_feat_cols, train_series,
            periods=fold_horizon,
            future_features=future_extra,
            holiday_country=holiday_country,
        )

        fold_resid = holdout_series.values.astype(float) - preds.values.astype(float)
        for h in range(1, fold_horizon + 1):
            horizons.append(h)
            signed_resids.append(float(fold_resid[h - 1]))
            abs_resids.append(abs(float(fold_resid[h - 1])))
            dows.append(int(holdout_series.index[h - 1].dayofweek))

    horizons_arr = np.asarray(horizons)
    abs_arr = np.asarray(abs_resids, dtype=float)
    dows_arr = np.asarray(dows)

    conformal_q = _finite_sample_quantile(abs_arr, alpha)

    # A valid finite-sample (1-alpha) conformal quantile needs at least
    # ceil((1-alpha)/alpha) residuals; thinner bins fall back to the pooled q.
    min_bin_n = int(np.ceil((1.0 - alpha) / alpha))
    bin_q: list[float] = []
    bin_counts: list[int] = []
    for b in range(len(bin_edges)):
        mask = np.array([_bin_index(int(h), bin_edges) == b for h in horizons_arr])
        n_bin = int(mask.sum())
        bin_counts.append(n_bin)
        if n_bin < min_bin_n:
            bin_q.append(conformal_q)
        else:
            bin_q.append(_finite_sample_quantile(abs_arr[mask], alpha))

    weekend_mask = dows_arr >= 5
    weekend_weekday_ratio = None
    if weekend_mask.any() and (~weekend_mask).any():
        weekday_median = float(np.median(abs_arr[~weekend_mask]))
        if weekday_median > 0:
            weekend_weekday_ratio = float(np.median(abs_arr[weekend_mask]) / weekday_median)

    # Refit on the full series for production inference.
    full_model, full_feat_cols = train_xgb_forecaster(
        series, lags=lags,
        extra_features=extra_features,
        holiday_country=holiday_country,
        log_target=log_target,
    )

    return {
        "point_model": full_model,
        "feat_cols": full_feat_cols,
        "conformal_q": conformal_q,
        "bin_edges": bin_edges,
        "bin_q": bin_q,
        "bin_counts": bin_counts,
        "method": "rolling_origin_v2",
        "n_folds_used": int(n_folds_used),
        "fold_horizon": int(fold_horizon),
        "fold_step": int(fold_step),
        "weekend_weekday_ratio": weekend_weekday_ratio,
        "alpha": float(alpha),
        "holdout_days": int(n_folds_used * fold_horizon),
        "residual_std": float(np.std(signed_resids, ddof=1)) if len(signed_resids) > 1 else 0.0,
        "log_target": bool(log_target),
        "fallback": False,
    }


def forecast_conformal(
    bundle: dict,
    history: pd.Series,
    periods: int = 30,
    future_features: pd.DataFrame | None = None,
    holiday_country: str | None = None,
    non_negative: bool = True,
    use_horizon_bins: bool = True,
) -> pd.DataFrame:
    """
    Generate point + conformal-interval forecasts.

    Output schema matches ``forecasting.forecast_xgb_with_intervals``:
    columns ``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper``.

    Horizon-binned half-widths are applied per forecast step when the bundle
    carries them; steps beyond the last bin edge reuse the last bin's width.
    Legacy bundles (no ``bin_q``) and ``use_horizon_bins=False`` apply the flat
    pooled ``conformal_q`` at every step.

    Lower bound is clipped at 0 by default since revenue and demand are
    non-negative. Pass ``non_negative=False`` for unbounded series.
    """
    yhat = forecast_xgb(
        bundle["point_model"],
        bundle["feat_cols"],
        history,
        periods=periods,
        future_features=future_features,
        holiday_country=holiday_country,
    )
    bin_q = bundle.get("bin_q") or []
    bin_edges = bundle.get("bin_edges") or []
    if use_horizon_bins and bin_q and bin_edges and len(bin_q) == len(bin_edges):
        q_per_step = np.array(
            [float(bin_q[_bin_index(h, bin_edges)]) for h in range(1, periods + 1)]
        )
    else:
        q_per_step = np.full(periods, float(bundle["conformal_q"]))
    lower = yhat.values - q_per_step
    upper = yhat.values + q_per_step
    if non_negative:
        lower = np.clip(lower, 0.0, None)
        upper = np.clip(upper, 0.0, None)
    return pd.DataFrame(
        {
            "ds": yhat.index,
            "yhat": yhat.values,
            "yhat_lower": lower,
            "yhat_upper": upper,
        }
    )
