"""Tests for rolling-origin, horizon-binned conformal prediction intervals."""
import numpy as np
import pandas as pd
import pytest

from src.conformal import forecast_conformal, split_conformal_calibrate
from src.forecasting import (
    forecast_xgb_with_intervals,
    train_xgb_forecaster,
    train_xgb_forecaster_with_intervals,
)


@pytest.fixture(scope="module")
def daily_series():
    """Synthetic 400-day daily series with weekly seasonality + trend + noise."""
    dates = pd.date_range("2022-01-01", periods=400, freq="D")
    rng = np.random.RandomState(0)
    trend = np.linspace(800, 1200, 400)
    weekly = 80 * np.sin(2 * np.pi * np.arange(400) / 7.0)
    noise = rng.normal(0, 30, size=400)
    values = trend + weekly + noise
    return pd.Series(values, index=dates, name="target")


@pytest.fixture(scope="module")
def bundle5(daily_series):
    """One calibrated bundle shared by read-only assertions (5 folds for speed)."""
    return split_conformal_calibrate(daily_series, lags=[1, 7], alpha=0.10, n_folds=5)


def test_calibrate_returns_positive_finite_q(bundle5):
    assert bundle5["conformal_q"] > 0
    assert np.isfinite(bundle5["conformal_q"])
    assert bundle5["method"] == "rolling_origin_v2"
    assert bundle5["fallback"] is False
    assert bundle5["n_folds_used"] == 5
    assert bundle5["holdout_days"] == 5 * 30  # total calibration days
    assert len(bundle5["bin_q"]) == len(bundle5["bin_edges"])
    assert all(np.isfinite(q) and q > 0 for q in bundle5["bin_q"])


def test_accepts_deprecated_holdout_days(daily_series):
    """Old call sites pass holdout_days; it must be accepted without error."""
    bundle = split_conformal_calibrate(
        daily_series, lags=[1, 7], holdout_days=60, alpha=0.10, n_folds=2,
    )
    assert bundle["fallback"] is False


def test_bin_quantiles_and_counts(bundle5):
    """K folds x 30-day horizon -> 7K / 7K / 16K residuals per bin."""
    k = bundle5["n_folds_used"]
    assert bundle5["bin_counts"] == [7 * k, 7 * k, 16 * k]


def test_calibrate_fallback_on_tiny_history():
    """Short series should not crash — it should fall back to a Gaussian quantile."""
    short = pd.Series(
        np.arange(40, dtype=float) + 100,
        index=pd.date_range("2024-01-01", periods=40, freq="D"),
    )
    bundle = split_conformal_calibrate(short, lags=[1, 7], alpha=0.10)
    assert bundle["fallback"] is True
    assert bundle["method"] == "gaussian_fallback"
    assert bundle["conformal_q"] > 0
    assert bundle["bin_q"] == []


def test_fold_autoshrink_on_short_series():
    """A series with room for only a couple of folds shrinks K instead of crashing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    rng = np.random.RandomState(3)
    series = pd.Series(500 + rng.normal(0, 20, size=100), index=dates)
    bundle = split_conformal_calibrate(series, lags=[1, 7], alpha=0.10, n_folds=5)
    assert bundle["fallback"] is False
    assert 0 < bundle["n_folds_used"] < 5


def test_bin_min_samples_fallback(daily_series):
    """With one fold, bins 1-7 and 8-14 hold 7 residuals (< 9 needed for a
    valid 90% conformal quantile) and must fall back to the pooled q."""
    bundle = split_conformal_calibrate(daily_series, lags=[1, 7], alpha=0.10, n_folds=1)
    assert bundle["bin_counts"] == [7, 7, 16]
    assert bundle["bin_q"][0] == pytest.approx(bundle["conformal_q"])
    assert bundle["bin_q"][1] == pytest.approx(bundle["conformal_q"])


def test_forecast_conformal_band_ordering(bundle5, daily_series):
    df = forecast_conformal(bundle5, daily_series, periods=14)
    assert len(df) == 14
    assert list(df.columns) == ["ds", "yhat", "yhat_lower", "yhat_upper"]
    assert (df["yhat_lower"] <= df["yhat"] + 1e-9).all()
    assert (df["yhat"] <= df["yhat_upper"] + 1e-9).all()


def test_forecast_conformal_non_negative_clip(bundle5, daily_series):
    """Lower bound must be clipped at zero when non_negative=True (default)."""
    df = forecast_conformal(bundle5, daily_series, periods=14)
    assert (df["yhat_lower"] >= 0.0).all()


def test_forecast_applies_per_horizon_widths(bundle5, daily_series):
    """Band width at step h must equal 2x the q of h's bin (series level ~1000,
    so the zero-clip never engages)."""
    df = forecast_conformal(bundle5, daily_series, periods=30, non_negative=False)
    width = (df["yhat_upper"] - df["yhat_lower"]).values
    bin_q = bundle5["bin_q"]
    assert np.allclose(width[0:7], 2 * bin_q[0])
    assert np.allclose(width[7:14], 2 * bin_q[1])
    assert np.allclose(width[14:30], 2 * bin_q[2])


def test_horizon_beyond_last_bin_uses_last_q(bundle5, daily_series):
    """API horizons can exceed the last bin edge (up to 90 days); those steps
    reuse the last bin's width."""
    df = forecast_conformal(bundle5, daily_series, periods=40, non_negative=False)
    width = (df["yhat_upper"] - df["yhat_lower"]).values
    assert np.allclose(width[30:40], 2 * bundle5["bin_q"][-1])


def test_use_horizon_bins_false_gives_flat_band(bundle5, daily_series):
    df = forecast_conformal(
        bundle5, daily_series, periods=30, non_negative=False, use_horizon_bins=False,
    )
    width = (df["yhat_upper"] - df["yhat_lower"]).values
    assert np.allclose(width, 2 * bundle5["conformal_q"])


def test_legacy_bundle_without_bins_falls_back_to_flat_q(daily_series):
    """Bundles pickled before v2 carry no bin keys; they must still forecast
    with the flat global width."""
    model, feat_cols = train_xgb_forecaster(daily_series, lags=[1, 7])
    legacy = {
        "point_model": model,
        "feat_cols": feat_cols,
        "conformal_q": 50.0,
        "alpha": 0.10,
        "holdout_days": 60,
        "residual_std": 25.0,
        "log_target": True,
        "fallback": False,
    }
    df = forecast_conformal(legacy, daily_series, periods=10, non_negative=False)
    width = (df["yhat_upper"] - df["yhat_lower"]).values
    assert np.allclose(width, 2 * 50.0)


def test_long_horizon_band_at_least_as_wide_as_short(bundle5):
    """Iterative one-step forecasting accumulates error with horizon, so the
    15-30 bin's width must not be tighter than the 1-7 bin's."""
    assert bundle5["bin_q"][0] <= bundle5["bin_q"][2] + 1e-9


def test_conformal_coverage_near_nominal():
    """Empirical coverage on a fresh held-out window should sit near the 90% nominal."""
    rng = np.random.RandomState(7)
    n = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    values = (
        1000.0
        + np.linspace(0, 200, n)
        + 50 * np.sin(2 * np.pi * np.arange(n) / 7.0)
        + rng.normal(0, 25, size=n)
    )
    series = pd.Series(values, index=dates, name="target")

    # Hold out the final 90 days for empirical-coverage evaluation; calibrate on
    # everything before that.
    eval_days = 90
    train_part = series.iloc[:-eval_days]
    eval_part = series.iloc[-eval_days:]

    bundle = split_conformal_calibrate(train_part, lags=[1, 7], alpha=0.10, n_folds=5)
    df = forecast_conformal(bundle, train_part, periods=eval_days)
    inside = (
        (df["yhat_lower"].values <= eval_part.values)
        & (eval_part.values <= df["yhat_upper"].values)
    )
    coverage = float(inside.mean())
    # Iterative one-step error accumulation widens uncertainty downstream, so
    # tolerate a moderate band around the 90% nominal coverage.
    assert 0.75 <= coverage <= 1.0, f"Coverage {coverage:.2f} outside reasonable range"


def test_conformal_schema_matches_quantile_intervals(bundle5, daily_series):
    """Drop-in compatibility: same DataFrame schema as forecast_xgb_with_intervals."""
    conf_df = forecast_conformal(bundle5, daily_series, periods=10)

    quant_models = train_xgb_forecaster_with_intervals(daily_series, lags=[1, 7])
    quant_df = forecast_xgb_with_intervals(quant_models, daily_series, periods=10)

    assert list(conf_df.columns) == list(quant_df.columns)
    assert len(conf_df) == len(quant_df)


def test_conformal_tighter_alpha_widens_band(daily_series):
    """A more conservative alpha (smaller alpha → higher coverage) should give a wider band."""
    b_wide = split_conformal_calibrate(daily_series, lags=[1, 7], alpha=0.20, n_folds=3)
    b_tight = split_conformal_calibrate(daily_series, lags=[1, 7], alpha=0.05, n_folds=3)
    assert b_tight["conformal_q"] >= b_wide["conformal_q"]
