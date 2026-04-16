import pandas as pd
import numpy as np
import pytest
from src.forecasting import (
    blend_forecasts,
    find_ensemble_weights,
    make_calendar_features,
    make_lag_features,
    train_xgb_forecaster,
    train_xgb_forecaster_with_intervals,
    train_xgb_direct_multistep,
    forecast_xgb,
    forecast_xgb_direct,
    forecast_xgb_with_intervals,
    train_sarima_forecaster,
    forecast_sarima,
    train_stacking_ensemble,
    forecast_stacked,
    _school_holiday_features,
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


# ---------------------------------------------------------------------------
# OTB extra_features / future_features tests
# ---------------------------------------------------------------------------

def test_xgb_forecast_with_extra_features(daily_series):
    """Model trains and forecasts without error when extra_features are provided."""
    otb = pd.DataFrame(
        {"otb_7d": 10, "otb_14d": 8, "otb_30d": 5, "otb_60d": 2},
        index=daily_series.index,
    )
    model, feat_cols = train_xgb_forecaster(daily_series, lags=[1, 7], extra_features=otb)
    assert any(c.startswith("otb_") for c in feat_cols)
    preds = forecast_xgb(model, feat_cols, daily_series, periods=10)
    assert len(preds) == 10


def test_xgb_forecast_sparse_future_features(daily_series):
    """forecast_xgb does not crash when future_features covers fewer dates than periods."""
    model, feat_cols = train_xgb_forecaster(daily_series, lags=[1, 7])
    # Only cover 5 of the 10 future dates
    future_dates = pd.date_range(
        daily_series.index[-1] + pd.Timedelta(days=1), periods=5, freq="D"
    )
    future_features = pd.DataFrame({"dummy": 0}, index=future_dates)
    preds = forecast_xgb(model, feat_cols, daily_series, periods=10,
                         future_features=future_features)
    assert len(preds) == 10
    assert preds.index[0] > daily_series.index[-1]
    assert preds.isna().sum() == 0


# ---------------------------------------------------------------------------
# School holiday features
# ---------------------------------------------------------------------------

def test_school_holiday_columns():
    """_school_holiday_features returns the expected columns."""
    idx = pd.date_range("2023-01-01", periods=365, freq="D")
    df = _school_holiday_features(idx)
    assert "is_summer_holiday" in df.columns
    assert "is_easter_holiday" in df.columns
    assert "is_christmas_holiday" in df.columns
    assert "is_school_holiday" in df.columns


def test_summer_holiday_flag():
    """August 15 is in summer holiday."""
    idx = pd.DatetimeIndex([pd.Timestamp("2023-08-15")])
    df = _school_holiday_features(idx)
    assert int(df["is_summer_holiday"].iloc[0]) == 1


def test_non_holiday_flag():
    """October 20 is not a school holiday."""
    idx = pd.DatetimeIndex([pd.Timestamp("2023-10-20")])
    df = _school_holiday_features(idx)
    assert int(df["is_school_holiday"].iloc[0]) == 0


def test_calendar_features_includes_school_holidays():
    """make_calendar_features includes school holiday columns."""
    idx = pd.date_range("2023-01-01", periods=10, freq="D")
    df = make_calendar_features(idx)
    assert "is_school_holiday" in df.columns
    assert "is_summer_holiday" in df.columns


def test_lag_features_includes_school_holidays():
    """make_lag_features propagates school holiday columns into the feature matrix."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    series = pd.Series(np.ones(100), index=dates)
    df = make_lag_features(series, lags=[1, 7])
    assert "is_school_holiday" in df.columns


# ---------------------------------------------------------------------------
# Ensemble blending
# ---------------------------------------------------------------------------

def test_find_ensemble_weights_returns_valid_weight():
    """find_ensemble_weights returns a float in [0, 1]."""
    xgb   = np.array([10.0, 11.0, 12.0, 9.0])
    prophet = np.array([9.5, 10.5, 11.5, 9.5])
    actual  = np.array([10.2, 10.8, 11.8, 9.2])
    w = find_ensemble_weights(xgb, prophet, actual)
    assert 0.0 <= w <= 1.0


def test_find_ensemble_weights_prefers_better_model():
    """When XGB perfectly predicts, weight should be 1.0."""
    actual  = np.array([10.0, 20.0, 30.0, 40.0])
    perfect = actual.copy()
    noisy   = actual + 5.0
    w = find_ensemble_weights(perfect, noisy, actual)
    assert w == 1.0


def test_blend_forecasts_correct_shape(daily_series):
    """blend_forecasts returns a Series with the same index as xgb_forecast."""
    model, feat_cols = train_xgb_forecaster(daily_series, lags=[1, 7])
    xgb_fc = forecast_xgb(model, feat_cols, daily_series, periods=10)
    prophet_vals = pd.Series(xgb_fc.values * 1.05, index=xgb_fc.index)
    result = blend_forecasts(xgb_fc, prophet_vals, xgb_weight=0.7)
    assert len(result) == len(xgb_fc)
    assert list(result.index) == list(xgb_fc.index)


def test_blend_forecasts_weight_1_equals_xgb(daily_series):
    """Weight=1.0 produces exactly the XGBoost forecast."""
    model, feat_cols = train_xgb_forecaster(daily_series, lags=[1, 7])
    xgb_fc = forecast_xgb(model, feat_cols, daily_series, periods=10)
    other = xgb_fc * 2.0
    result = blend_forecasts(xgb_fc, other, xgb_weight=1.0)
    np.testing.assert_array_almost_equal(result.values, xgb_fc.values)


# ---------------------------------------------------------------------------
# Track 4: log1p target transform
# ---------------------------------------------------------------------------

def test_log_target_forecast_positive(daily_series):
    """log_target=True model returns strictly positive forecasts."""
    model, feat_cols = train_xgb_forecaster(daily_series, lags=[1, 7], log_target=True)
    assert getattr(model, "log_target", False) is True
    preds = forecast_xgb(model, feat_cols, daily_series, periods=10)
    assert (preds.values > 0).all()


def test_log_target_off_returns_raw(daily_series):
    """log_target=False model does not set log_target attribute."""
    model, _ = train_xgb_forecaster(daily_series, lags=[1, 7], log_target=False)
    assert getattr(model, "log_target", False) is False


def test_log_target_direct_positive(daily_series):
    """Direct multi-step with log_target=True returns positive values."""
    models = train_xgb_direct_multistep(daily_series, horizons=5, lags=[1, 7], log_target=True)
    preds = forecast_xgb_direct(models, daily_series)
    assert (preds.values > 0).all()


# ---------------------------------------------------------------------------
# Track 5a: interaction features
# ---------------------------------------------------------------------------

def test_interaction_features_present(daily_series):
    """make_lag_features includes all 6 interaction columns."""
    df = make_lag_features(daily_series, lags=[1, 7])
    for col in ["season", "dow_sin_x_month", "dow_cos_x_month",
                "is_weekend_x_holiday", "school_x_weekend", "season_x_is_weekend"]:
        assert col in df.columns, f"Missing interaction column: {col}"


def test_season_values_in_range(daily_series):
    """season column is in {0, 1, 2, 3}."""
    df = make_lag_features(daily_series, lags=[1, 7])
    assert set(df["season"].unique()).issubset({0, 1, 2, 3})


# ---------------------------------------------------------------------------
# Track 5d: stacking ensemble
# ---------------------------------------------------------------------------

def test_stacking_ensemble_trains(daily_series):
    """train_stacking_ensemble completes without error on a 200-day series."""
    meta, components = train_stacking_ensemble(daily_series, n_splits=3, lags=[1, 7])
    assert meta is not None
    assert set(components.keys()) == {"xgb", "prophet", "sarima"}


def test_stacking_forecast_length(daily_series):
    """forecast_stacked returns a Series of the requested length."""
    meta, components = train_stacking_ensemble(daily_series, n_splits=3, lags=[1, 7])
    preds = forecast_stacked(meta, components, daily_series, periods=14)
    assert len(preds) == 14
    assert preds.index[0] > daily_series.index[-1]


# ---------------------------------------------------------------------------
# Booking velocity features
# ---------------------------------------------------------------------------

def test_velocity_features_present(daily_series):
    """make_lag_features includes booking velocity and acceleration sign columns."""
    df = make_lag_features(daily_series, lags=[1, 7])
    assert "booking_velocity_7" in df.columns
    assert "booking_velocity_14" in df.columns
    assert "booking_accel_sign" in df.columns


def test_accel_sign_values(daily_series):
    """booking_accel_sign is in {-1, 0, 1}."""
    df = make_lag_features(daily_series, lags=[1, 7])
    assert set(df["booking_accel_sign"].unique()).issubset({-1, 0, 1})
