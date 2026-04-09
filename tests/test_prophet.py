"""
Tests for Prophet forecasting functions in src/forecasting.py.
"""
import pandas as pd
import numpy as np
import pytest
from src.forecasting import train_prophet_forecaster, forecast_prophet


@pytest.fixture(scope="module")
def daily_series():
    dates = pd.date_range("2020-01-01", periods=365, freq="D")
    rng = np.random.default_rng(42)
    values = 1000 + 200 * np.sin(np.linspace(0, 4 * np.pi, 365)) + rng.normal(0, 50, 365)
    return pd.Series(values, index=dates)


@pytest.fixture(scope="module")
def trained_prophet(daily_series):
    return train_prophet_forecaster(daily_series)


def test_train_returns_model(trained_prophet):
    from prophet import Prophet
    assert isinstance(trained_prophet, Prophet)


def test_forecast_shape(trained_prophet):
    df = forecast_prophet(trained_prophet, periods=30)
    assert len(df) == 30


def test_forecast_columns(trained_prophet):
    df = forecast_prophet(trained_prophet, periods=30)
    assert set(["ds", "yhat", "yhat_lower", "yhat_upper"]).issubset(df.columns)


def test_confidence_interval_ordering(trained_prophet):
    df = forecast_prophet(trained_prophet, periods=30)
    assert (df["yhat_upper"] >= df["yhat"]).all(), "yhat_upper should be >= yhat"
    assert (df["yhat"] >= df["yhat_lower"]).all(), "yhat should be >= yhat_lower"


def test_forecast_dates_are_future(trained_prophet, daily_series):
    df = forecast_prophet(trained_prophet, periods=30)
    last_hist = daily_series.index[-1]
    assert pd.to_datetime(df["ds"].iloc[0]) > last_hist
