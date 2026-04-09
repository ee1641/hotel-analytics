"""
Time-series forecasting for hotel revenue and demand.

Models:
  - XGBoost with iterative one-step-ahead lag + calendar features
  - SARIMA (statsmodels) with weekly seasonality (order=(1,1,1), seasonal=(1,1,1,7))

NOTE: SARIMA captures weekly seasonality only. Annual patterns (~365-day period)
are not modeled -- computationally intractable with pure SARIMA at daily frequency.
"""
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX


def make_lag_features(series: pd.Series, lags: list) -> pd.DataFrame:
    """
    Build a rich DataFrame of lag + rolling + calendar features from a daily series.
    Rows with NaN (due to lagging/rolling) are dropped.
    """
    df = pd.DataFrame({"target": series})

    # Lag features
    for lag in lags:
        df[f"lag_{lag}"] = series.shift(lag)

    # Rolling statistics (use min_periods=1 to reduce NaN rows)
    df["rolling_mean_7"]  = series.shift(1).rolling(7,  min_periods=1).mean()
    df["rolling_mean_14"] = series.shift(1).rolling(14, min_periods=1).mean()
    df["rolling_mean_28"] = series.shift(1).rolling(28, min_periods=1).mean()
    df["rolling_std_7"]   = series.shift(1).rolling(7,  min_periods=2).std().fillna(0)

    # Calendar features
    df["dayofweek"]      = series.index.dayofweek
    df["month"]          = series.index.month
    df["weekofyear"]     = series.index.isocalendar().week.astype(int).values
    df["is_weekend"]     = (series.index.dayofweek >= 5).astype(int)
    df["is_month_start"] = series.index.is_month_start.astype(int)
    df["is_month_end"]   = series.index.is_month_end.astype(int)

    df.dropna(inplace=True)
    return df


def train_xgb_forecaster(
    series: pd.Series,
    lags: list = None,
    n_estimators: int = 200,
    max_depth: int = 5,
):
    """
    Train an XGBoost regressor on lag + calendar features.

    Returns:
        model      : fitted XGBRegressor
        feat_cols  : list of feature column names (in order)
    """
    if lags is None:
        lags = [1, 7, 14, 30]
    df = make_lag_features(series, lags)
    feat_cols = [c for c in df.columns if c != "target"]
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        random_state=42,
        verbosity=0,
    )
    model.fit(df[feat_cols].values, df["target"].values)
    return model, feat_cols


def forecast_xgb(
    model: XGBRegressor,
    feat_cols: list,
    history: pd.Series,
    periods: int = 30,
) -> pd.Series:
    """
    Iterative one-step-ahead forecast.
    Each step: build row from current history -> predict -> append to history.
    Calendar features (dayofweek, month) are computed from the real future date.
    """
    lags = [int(c.split("_")[1]) for c in feat_cols if c.startswith("lag_")]
    extended = list(history.values)
    future_dates = pd.date_range(
        history.index[-1] + pd.Timedelta(days=1), periods=periods, freq="D"
    )
    predictions = []

    for future_date in future_dates:
        lag_vals = {f"lag_{lag}": extended[-lag] for lag in lags}
        # Rolling stats from the extended history tail
        tail7  = extended[-7:]  if len(extended) >= 7  else extended
        tail14 = extended[-14:] if len(extended) >= 14 else extended
        tail28 = extended[-28:] if len(extended) >= 28 else extended
        rolling = {
            "rolling_mean_7":  float(np.mean(tail7)),
            "rolling_mean_14": float(np.mean(tail14)),
            "rolling_mean_28": float(np.mean(tail28)),
            "rolling_std_7":   float(np.std(tail7)) if len(tail7) > 1 else 0.0,
        }
        calendar = {
            "dayofweek":      future_date.dayofweek,
            "month":          future_date.month,
            "weekofyear":     future_date.isocalendar()[1],
            "is_weekend":     int(future_date.dayofweek >= 5),
            "is_month_start": int(future_date.is_month_start),
            "is_month_end":   int(future_date.is_month_end),
        }
        row = {**lag_vals, **rolling, **calendar}
        # Only include columns present in feat_cols (backward-compat with old models)
        row_df = pd.DataFrame([{k: row.get(k, 0) for k in feat_cols}])
        pred = float(model.predict(row_df.values)[0])
        predictions.append(pred)
        extended.append(pred)

    return pd.Series(predictions, index=future_dates, name="forecast")


def train_sarima_forecaster(
    series: pd.Series,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 7),
):
    """
    Fit a SARIMA model to a daily time series.
    Returns the fitted SARIMAXResults object.
    """
    warnings.filterwarnings("ignore")
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def forecast_sarima(result, periods: int = 30) -> pd.Series:
    """Out-of-sample forecast from a fitted SARIMA result."""
    return result.forecast(steps=periods)


def naive_baseline_forecast(series: pd.Series, periods: int = 30) -> pd.Series:
    """
    Seasonal naive baseline: each future day = the value from 7 days prior
    (last-week-same-day). For step i (0-indexed), uses series.iloc[-(7 - i % 7)].

    Returns a Series of length `periods` with a DatetimeIndex starting the
    day after series.index[-1].
    """
    future_dates = pd.date_range(
        series.index[-1] + pd.Timedelta(days=1), periods=periods, freq="D"
    )
    predictions = [
        float(series.iloc[-(7 - (i % 7))]) for i in range(periods)
    ]
    return pd.Series(predictions, index=future_dates, name="naive_baseline")


# ---------------------------------------------------------------------------
# Prophet forecasting (replaces SARIMA in the dashboard)
# ---------------------------------------------------------------------------

def train_prophet_forecaster(series: pd.Series):
    """
    Fit a Prophet model with weekly + yearly seasonality and Portuguese public holidays.

    Returns:
        Fitted Prophet model object.
    """
    from prophet import Prophet  # deferred import — optional dependency
    df_p = pd.DataFrame({"ds": series.index, "y": series.values})
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        uncertainty_samples=200,
    )
    m.add_country_holidays(country_name="PT")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(df_p)
    return m


def forecast_prophet(model, periods: int = 30) -> pd.DataFrame:
    """
    Generate a forward forecast from a fitted Prophet model.

    Returns:
        DataFrame with columns: ds, yhat, yhat_lower, yhat_upper
    """
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return (
        forecast.tail(periods)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# XGBoost quantile forecasting (confidence intervals)
# ---------------------------------------------------------------------------

def train_xgb_forecaster_with_intervals(
    series: pd.Series,
    lags: list = None,
    quantiles: tuple = (0.1, 0.5, 0.9),
) -> dict:
    """
    Train three XGBoost models at specified quantiles (p10, p50, p90).
    Uses quantile regression via objective="reg:quantileerror".

    Returns:
        dict with keys matching quantile labels: {"q10": (model, feat_cols), ...}
    """
    if lags is None:
        lags = [1, 2, 3, 7, 14, 21, 28]
    df = make_lag_features(series, lags)
    feat_cols = [c for c in df.columns if c != "target"]
    X_vals = df[feat_cols].values
    y_vals = df["target"].values

    result = {}
    label_map = {0.1: "q10", 0.5: "q50", 0.9: "q90"}
    for q in quantiles:
        m = XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            objective="reg:quantileerror",
            quantile_alpha=q,
            random_state=42,
            verbosity=0,
        )
        m.fit(X_vals, y_vals)
        result[label_map.get(q, f"q{int(q*100)}")] = (m, feat_cols)
    return result


def forecast_xgb_with_intervals(
    models_dict: dict,
    history: pd.Series,
    periods: int = 30,
) -> pd.DataFrame:
    """
    Generate forecasts for q10, q50, q90 and return a DataFrame
    with columns: ds, yhat, yhat_lower, yhat_upper.
    """
    forecasts = {}
    for key, (model, feat_cols) in models_dict.items():
        forecasts[key] = forecast_xgb(model, feat_cols, history, periods)

    q50 = forecasts.get("q50", list(forecasts.values())[1])
    q10 = forecasts.get("q10", list(forecasts.values())[0])
    q90 = forecasts.get("q90", list(forecasts.values())[2])

    return pd.DataFrame({
        "ds":          q50.index,
        "yhat":        q50.values,
        "yhat_lower":  q10.values,
        "yhat_upper":  q90.values,
    })
