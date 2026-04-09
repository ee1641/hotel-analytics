import pandas as pd
import numpy as np
import pytest
from src.forecasting import (
    make_lag_features,
    train_xgb_forecaster,
    forecast_xgb,
    train_sarima_forecaster,
    forecast_sarima,
)


@pytest.fixture(scope="module")
def daily_series():
    """Synthetic 200-day daily series."""
    dates = pd.date_range("2022-01-01", periods=200, freq="D")
    np.random.seed(42)
    values = 1000 + np.cumsum(np.random.randn(200) * 10)
    return pd.Series(values, index=dates, name="target")


def test_lag_features_columns(daily_series):
    df = make_lag_features(daily_series, lags=[1, 7, 14])
    assert "lag_1" in df.columns
    assert "lag_7" in df.columns
    assert "lag_14" in df.columns
    assert "dayofweek" in df.columns
    assert "month" in df.columns


def test_lag_features_no_nan(daily_series):
    df = make_lag_features(daily_series, lags=[1, 7])
    assert df.isna().sum().sum() == 0


def test_xgb_forecaster_returns_model_and_cols(daily_series):
    model, feat_cols = train_xgb_forecaster(daily_series, lags=[1, 7])
    assert model is not None
    assert len(feat_cols) > 0


def test_xgb_forecast_length(daily_series):
    model, feat_cols = train_xgb_forecaster(daily_series, lags=[1, 7])
    preds = forecast_xgb(model, feat_cols, daily_series, periods=30)
    assert len(preds) == 30


def test_xgb_forecast_index_is_future(daily_series):
    model, feat_cols = train_xgb_forecaster(daily_series, lags=[1, 7])
    preds = forecast_xgb(model, feat_cols, daily_series, periods=10)
    assert preds.index[0] > daily_series.index[-1]


def test_sarima_returns_result(daily_series):
    result = train_sarima_forecaster(daily_series, order=(1, 1, 1))
    assert result is not None


def test_sarima_forecast_length(daily_series):
    result = train_sarima_forecaster(daily_series, order=(1, 1, 1))
    preds = forecast_sarima(result, periods=30)
    assert len(preds) == 30
