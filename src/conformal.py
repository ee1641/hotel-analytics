"""
Split-conformal prediction intervals for XGBoost forecasters.

Trains the point model on ``history[:-holdout_days]``, predicts the held-out
window with the same one-step-iterative path used at inference time, takes the
``(1 - alpha)``-quantile of ``|residual|`` as the conformity score ``q``, and
produces symmetric intervals ``yhat ± q``.

Under exchangeability the resulting intervals have marginal coverage of at
least ``1 - alpha``. The quantile-loss XGB intervals in ``forecasting.py``
optimise a different objective and have no coverage guarantee — this module
is the upgrade path.

After calibration the point model is **refit on the full series** so the
production forecaster sees every observation; only the residual quantile is
held out. That is the "split" part of split-conformal.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.forecasting import (
    forecast_xgb,
    train_xgb_forecaster,
)
from src.model_specs import FORECAST_LAGS, MIN_FORECAST_HISTORY

log = logging.getLogger(__name__)


def split_conformal_calibrate(
    series: pd.Series,
    *,
    lags: list[int] | None = None,
    holdout_days: int = 60,
    alpha: float = 0.10,
    extra_features: pd.DataFrame | None = None,
    holiday_country: str | None = None,
    log_target: bool = True,
) -> dict:
    """
    Fit a point model + conformity score on a daily series.

    Parameters
    ----------
    series : daily target (revenue or demand), sorted by date.
    lags : forecast lags. Defaults to FORECAST_LAGS.
    holdout_days : size of the conformity holdout (last N days).
    alpha : 1 - nominal coverage (0.10 → 90% intervals).
    extra_features : optional exogenous feature DataFrame, same index as series.
    holiday_country : ISO country code for holiday features.
    log_target : whether to log1p-transform the target during fitting.

    Returns
    -------
    dict with keys:
        point_model    : fitted XGBRegressor (trained on full series)
        feat_cols      : feature column order for the point model
        conformal_q    : float, additive half-width
        alpha          : float
        holdout_days   : int actually used
        residual_std   : float, diagnostic
        log_target     : bool
        fallback       : bool, True when the holdout was too small to compute a
                         data-driven quantile and a Gaussian fallback was used.
    """
    series = pd.Series(series).sort_index()
    lags = list(lags) if lags is not None else list(FORECAST_LAGS)
    n = len(series)
    fallback = False

    train_len = n - holdout_days
    if train_len < max(MIN_FORECAST_HISTORY, max(lags) + 1):
        log.warning(
            "Conformal calibration: history too short (%d) for holdout=%d; "
            "using Gaussian fallback.", n, holdout_days,
        )
        # Train on full series and approximate q from residuals on the training
        # data itself (optimistic, but better than no interval at all).
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
            "alpha": float(alpha),
            "holdout_days": 0,
            "residual_std": residual_std,
            "log_target": bool(log_target),
            "fallback": True,
        }

    train_series = series.iloc[:train_len]
    holdout_series = series.iloc[train_len:]

    cal_extra = extra_features.iloc[:train_len] if extra_features is not None else None
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
        periods=holdout_days,
        future_features=future_extra,
        holiday_country=holiday_country,
    )

    residuals = holdout_series.values.astype(float) - preds.values.astype(float)
    abs_resid = np.abs(residuals)
    # Finite-sample conformal correction: use ceil((n+1)(1-alpha)) / n empirical quantile.
    k = int(np.ceil((holdout_days + 1) * (1.0 - alpha)))
    k = min(max(k, 1), holdout_days)
    conformal_q = float(np.sort(abs_resid)[k - 1])

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
        "alpha": float(alpha),
        "holdout_days": int(holdout_days),
        "residual_std": float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0,
        "log_target": bool(log_target),
        "fallback": fallback,
    }


def forecast_conformal(
    bundle: dict,
    history: pd.Series,
    periods: int = 30,
    future_features: pd.DataFrame | None = None,
    holiday_country: str | None = None,
    non_negative: bool = True,
) -> pd.DataFrame:
    """
    Generate point + conformal-interval forecasts.

    Output schema matches ``forecasting.forecast_xgb_with_intervals``:
    columns ``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper``.

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
    q = float(bundle["conformal_q"])
    lower = yhat.values - q
    upper = yhat.values + q
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
