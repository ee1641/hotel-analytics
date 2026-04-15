import pandas as pd
import numpy as np
import pytest
from src.forecasting import (
    make_lag_features,
    train_xgb_forecaster,
    train_xgb_forecaster_with_intervals,
    train_xgb_direct_multistep,
    forecast_xgb,
    forecast_xgb_direct,
    forecast_xgb_with_intervals,
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
    assert "is_holiday" in df.columns


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


def test_xgb_quantile_forecast_columns(daily_series):
    models = train_xgb_forecaster_with_intervals(daily_series, lags=[1, 7, 14])
    preds = forecast_xgb_with_intervals(models, daily_series, periods=10)
    assert set(preds.columns) == {"ds", "yhat", "yhat_lower", "yhat_upper"}


def test_xgb_quantile_forecast_length(daily_series):
    models = train_xgb_forecaster_with_intervals(daily_series, lags=[1, 7, 14])
    preds = forecast_xgb_with_intervals(models, daily_series, periods=12)
    assert len(preds) == 12


def test_lag_features_has_rolling_std_28(daily_series):
    df = make_lag_features(daily_series, lags=[1, 7])
    assert "rolling_std_28" in df.columns
    assert df["rolling_std_28"].isna().sum() == 0


def test_sarima_returns_result(daily_series):
    result = train_sarima_forecaster(daily_series, order=(1, 1, 1))
    assert result is not None


def test_sarima_forecast_length(daily_series):
    result = train_sarima_forecaster(daily_series, order=(1, 1, 1))
    preds = forecast_sarima(result, periods=30)
    assert len(preds) == 30


def test_direct_multistep_produces_correct_count(daily_series):
    models = train_xgb_direct_multistep(daily_series, horizons=5, lags=[1, 7])
    assert len(models) == 5
    assert all(isinstance(m, tuple) and len(m) == 2 for m in models)


def test_direct_multistep_forecast_no_accumulation(daily_series):
    models = train_xgb_direct_multistep(daily_series, horizons=5, lags=[1, 7])
    preds = forecast_xgb_direct(models, daily_series)
    assert len(preds) == 5
    assert preds.index[0] > daily_series.index[-1]
    assert preds.isna().sum() == 0
