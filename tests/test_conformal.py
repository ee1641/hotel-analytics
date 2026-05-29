"""Tests for split-conformal prediction intervals."""
import numpy as np
import pandas as pd
import pytest

from src.conformal import forecast_conformal, split_conformal_calibrate
from src.forecasting import forecast_xgb_with_intervals, train_xgb_forecaster_with_intervals


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


def test_calibrate_returns_positive_finite_q(daily_series):
    bundle = split_conformal_calibrate(daily_series, lags=[1, 7], holdout_days=60, alpha=0.10)
    assert bundle["conformal_q"] > 0
    assert np.isfinite(bundle["conformal_q"])
    assert bundle["holdout_days"] == 60
    assert bundle["fallback"] is False


def test_calibrate_fallback_on_tiny_history():
    """Short series should not crash — it should fall back to a Gaussian quantile."""
    short = pd.Series(
        np.arange(40, dtype=float) + 100,
        index=pd.date_range("2024-01-01", periods=40, freq="D"),
    )
    bundle = split_conformal_calibrate(short, lags=[1, 7], holdout_days=60, alpha=0.10)
    assert bundle["fallback"] is True
    assert bundle["conformal_q"] > 0


def test_forecast_conformal_band_ordering(daily_series):
    bundle = split_conformal_calibrate(daily_series, lags=[1, 7], holdout_days=60, alpha=0.10)
    df = forecast_conformal(bundle, daily_series, periods=14)
    assert len(df) == 14
    assert list(df.columns) == ["ds", "yhat", "yhat_lower", "yhat_upper"]
    assert (df["yhat_lower"] <= df["yhat"] + 1e-9).all()
    assert (df["yhat"] <= df["yhat_upper"] + 1e-9).all()


def test_forecast_conformal_non_negative_clip(daily_series):
    """Lower bound must be clipped at zero when non_negative=True (default)."""
    bundle = split_conformal_calibrate(daily_series, lags=[1, 7], holdout_days=60, alpha=0.10)
    df = forecast_conformal(bundle, daily_series, periods=14)
    assert (df["yhat_lower"] >= 0.0).all()


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

    bundle = split_conformal_calibrate(train_part, lags=[1, 7], holdout_days=60, alpha=0.10)
    df = forecast_conformal(bundle, train_part, periods=eval_days)
    inside = (
        (df["yhat_lower"].values <= eval_part.values)
        & (eval_part.values <= df["yhat_upper"].values)
    )
    coverage = float(inside.mean())
    # Iterative one-step error accumulation widens uncertainty downstream, so
    # tolerate a moderate band around the 90% nominal coverage.
    assert 0.75 <= coverage <= 1.0, f"Coverage {coverage:.2f} outside reasonable range"


def test_conformal_schema_matches_quantile_intervals(daily_series):
    """Drop-in compatibility: same DataFrame schema as forecast_xgb_with_intervals."""
    bundle = split_conformal_calibrate(daily_series, lags=[1, 7], holdout_days=60, alpha=0.10)
    conf_df = forecast_conformal(bundle, daily_series, periods=10)

    quant_models = train_xgb_forecaster_with_intervals(daily_series, lags=[1, 7])
    quant_df = forecast_xgb_with_intervals(quant_models, daily_series, periods=10)

    assert list(conf_df.columns) == list(quant_df.columns)
    assert len(conf_df) == len(quant_df)


def test_conformal_tighter_alpha_widens_band(daily_series):
    """A more conservative alpha (smaller alpha → higher coverage) should give a wider band."""
    b_wide = split_conformal_calibrate(daily_series, lags=[1, 7], holdout_days=60, alpha=0.20)
    b_tight = split_conformal_calibrate(daily_series, lags=[1, 7], holdout_days=60, alpha=0.05)
    assert b_tight["conformal_q"] >= b_wide["conformal_q"]
