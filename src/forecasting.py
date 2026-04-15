"""
Time-series forecasting for hotel revenue and demand.

Models:
  - XGBoost with iterative one-step-ahead forecasting
  - SARIMA (statsmodels) with weekly seasonality
  - Prophet with holidays and interval output
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from src.config import APP_CONFIG
from src.model_specs import FORECAST_LAGS, FORECAST_QUANTILES


def resolve_lags(lags: list | None, series_length: int) -> list[int]:
    """
    Keep only lags that fit the available series length.
    Falls back to the shortest available lag if needed.
    """
    requested = list(lags or FORECAST_LAGS)
    resolved = [int(lag) for lag in requested if 0 < int(lag) < int(series_length)]
    if resolved:
        return sorted(set(resolved))
    if series_length > 1:
        return [1]
    raise ValueError("Forecasting requires at least 2 observations.")


def _holiday_index(index: pd.DatetimeIndex, holiday_country: str | None = None) -> np.ndarray:
    country = holiday_country or APP_CONFIG.get("holiday_country", "PT")
    try:
        import holidays

        years = range(index.min().year - 1, index.max().year + 2)
        holiday_map = holidays.country_holidays(country, years=years)
        return np.array(sorted(pd.to_datetime(list(holiday_map.keys()))), dtype="datetime64[ns]")
    except Exception:
        return np.array([], dtype="datetime64[ns]")


def _holiday_distance_features(
    index: pd.DatetimeIndex,
    holiday_country: str | None = None,
) -> pd.DataFrame:
    holiday_dates = _holiday_index(index, holiday_country=holiday_country)
    idx_values = index.values.astype("datetime64[ns]")

    result = pd.DataFrame(index=index)
    if len(holiday_dates) == 0:
        result["is_holiday"] = 0
        result["days_to_holiday"] = 0
        result["days_after_holiday"] = 0
        return result

    result["is_holiday"] = np.isin(idx_values, holiday_dates).astype(int)

    positions = np.searchsorted(holiday_dates, idx_values)
    next_pos = np.clip(positions, 0, len(holiday_dates) - 1)
    prev_pos = np.clip(positions - 1, 0, len(holiday_dates) - 1)

    next_days = (
        (holiday_dates[next_pos].astype("datetime64[D]") - idx_values.astype("datetime64[D]"))
        .astype("timedelta64[D]")
        .astype(int)
    )
    prev_days = (
        (idx_values.astype("datetime64[D]") - holiday_dates[prev_pos].astype("datetime64[D]"))
        .astype("timedelta64[D]")
        .astype(int)
    )

    result["days_to_holiday"] = np.maximum(next_days, 0)
    result["days_after_holiday"] = np.maximum(prev_days, 0)
    return result


def make_calendar_features(
    index: pd.DatetimeIndex,
    holiday_country: str | None = None,
) -> pd.DataFrame:
    """Create calendar and holiday features for a DatetimeIndex."""
    df = pd.DataFrame(index=index)
    week = index.isocalendar().week.astype(int).values

    df["dayofweek"] = index.dayofweek
    df["month"] = index.month
    df["weekofyear"] = week
    df["dayofyear"] = index.dayofyear
    df["is_weekend"] = (index.dayofweek >= 5).astype(int)
    df["is_month_start"] = index.is_month_start.astype(int)
    df["is_month_end"] = index.is_month_end.astype(int)
    df["dow_sin"] = np.sin(2 * np.pi * index.dayofweek / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * index.dayofweek / 7.0)
    df["week_sin"] = np.sin(2 * np.pi * week / 52.0)
    df["week_cos"] = np.cos(2 * np.pi * week / 52.0)

    holiday_df = _holiday_distance_features(index, holiday_country=holiday_country)
    return pd.concat([df, holiday_df], axis=1)


def make_lag_features(
    series: pd.Series,
    lags: list | None = None,
    extra_features: pd.DataFrame | None = None,
    holiday_country: str | None = None,
) -> pd.DataFrame:
    """
    Build lag, rolling, calendar, and optional exogenous features from a daily series.
    Rows with NaN (due to lagging/rolling) are dropped.
    """
    series = pd.Series(series).sort_index()
    resolved_lags = resolve_lags(lags, len(series))

    df = pd.DataFrame({"target": series})
    for lag in resolved_lags:
        df[f"lag_{lag}"] = series.shift(lag)

    df["rolling_mean_7"] = series.shift(1).rolling(7, min_periods=1).mean()
    df["rolling_mean_14"] = series.shift(1).rolling(14, min_periods=1).mean()
    df["rolling_mean_28"] = series.shift(1).rolling(28, min_periods=1).mean()
    df["rolling_std_7"] = series.shift(1).rolling(7, min_periods=2).std().fillna(0.0)
    df["rolling_std_28"] = series.shift(1).rolling(28, min_periods=2).std().fillna(0.0)
    df["trend_7"] = series.shift(1).diff(7).fillna(0.0)

    if "lag_364" in df.columns and "lag_365" in df.columns:
        df["swly_mean"] = df[["lag_364", "lag_365"]].mean(axis=1)
    elif "lag_364" in df.columns:
        df["swly_mean"] = df["lag_364"]
    elif "lag_365" in df.columns:
        df["swly_mean"] = df["lag_365"]

    calendar_df = make_calendar_features(df.index, holiday_country=holiday_country)
    df = pd.concat([df, calendar_df], axis=1)

    if extra_features is not None and len(extra_features) > 0:
        aligned = extra_features.reindex(df.index).copy()
        aligned = aligned.fillna(0.0)
        df = pd.concat([df, aligned], axis=1)

    df.dropna(inplace=True)
    return df


def train_xgb_forecaster(
    series: pd.Series,
    lags: list | None = None,
    n_estimators: int = 300,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    extra_features: pd.DataFrame | None = None,
    holiday_country: str | None = None,
):
    """
    Train an XGBoost regressor on lag, rolling, calendar, holiday, and optional extra features.

    Returns:
        model      : fitted XGBRegressor
        feat_cols  : list of feature column names (in order)
    """
    df = make_lag_features(
        series,
        lags=lags,
        extra_features=extra_features,
        holiday_country=holiday_country,
    )
    feat_cols = [c for c in df.columns if c != "target"]

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        verbosity=0,
    )
    model.fit(df[feat_cols].values, df["target"].values)
    return model, feat_cols


def _infer_model_lags(feat_cols: list[str]) -> list[int]:
    return sorted(
        {
            int(c.split("_")[1])
            for c in feat_cols
            if c.startswith("lag_") and c.split("_")[1].isdigit()
        }
    )


def _build_forecast_row(
    future_date: pd.Timestamp,
    feat_cols: list[str],
    extended: list[float],
    future_features: pd.DataFrame | None = None,
    holiday_country: str | None = None,
) -> pd.DataFrame:
    lags = _infer_model_lags(feat_cols)
    lag_vals = {f"lag_{lag}": extended[-lag] for lag in lags}

    tail7 = extended[-7:] if len(extended) >= 7 else extended
    tail14 = extended[-14:] if len(extended) >= 14 else extended
    tail28 = extended[-28:] if len(extended) >= 28 else extended

    row = {
        **lag_vals,
        "rolling_mean_7": float(np.mean(tail7)),
        "rolling_mean_14": float(np.mean(tail14)),
        "rolling_mean_28": float(np.mean(tail28)),
        "rolling_std_7": float(np.std(tail7)) if len(tail7) > 1 else 0.0,
        "rolling_std_28": float(np.std(tail28)) if len(tail28) > 1 else 0.0,
        "trend_7": float(tail7[-1] - tail7[0]) if len(tail7) > 1 else 0.0,
    }

    calendar = make_calendar_features(
        pd.DatetimeIndex([future_date]),
        holiday_country=holiday_country,
    ).iloc[0].to_dict()
    row.update(calendar)

    if "swly_mean" in feat_cols:
        swly_candidates = [row.get("lag_364"), row.get("lag_365")]
        swly_values = [v for v in swly_candidates if v is not None]
        row["swly_mean"] = float(np.mean(swly_values)) if swly_values else 0.0

    if future_features is not None and future_date in future_features.index:
        extra_row = future_features.loc[future_date]
        if isinstance(extra_row, pd.DataFrame):
            extra_row = extra_row.iloc[0]
        row.update(extra_row.fillna(0.0).to_dict())

    return pd.DataFrame([{k: row.get(k, 0.0) for k in feat_cols}])


def forecast_xgb(
    model: XGBRegressor,
    feat_cols: list,
    history: pd.Series,
    periods: int = 30,
    future_features: pd.DataFrame | None = None,
    holiday_country: str | None = None,
) -> pd.Series:
    """
    Iterative one-step-ahead forecast.
    Calendar and holiday features are computed from the future date.
    Optional future_features can inject pickup or segment-context features.
    """
    history = pd.Series(history).sort_index()
    extended = list(history.values.astype(float))
    future_dates = pd.date_range(
        history.index[-1] + pd.Timedelta(days=1),
        periods=periods,
        freq="D",
    )
    predictions = []

    for future_date in future_dates:
        row_df = _build_forecast_row(
            future_date,
            feat_cols,
            extended,
            future_features=future_features,
            holiday_country=holiday_country,
        )
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
    Seasonal naive baseline: each future day equals the same weekday from last week.
    """
    future_dates = pd.date_range(
        series.index[-1] + pd.Timedelta(days=1),
        periods=periods,
        freq="D",
    )
    predictions = [float(series.iloc[-(7 - (i % 7))]) for i in range(periods)]
    return pd.Series(predictions, index=future_dates, name="naive_baseline")


def train_prophet_forecaster(
    series: pd.Series,
    holiday_country: str | None = None,
):
    """
    Fit a Prophet model with weekly and yearly seasonality plus public holidays.
    """
    from prophet import Prophet

    country = holiday_country or APP_CONFIG.get("holiday_country", "PT")
    df_p = pd.DataFrame({"ds": series.index, "y": series.values})
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        uncertainty_samples=200,
    )
    try:
        model.add_country_holidays(country_name=country)
    except Exception:
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(df_p)
    return model


def forecast_prophet(model, periods: int = 30) -> pd.DataFrame:
    """
    Generate a forward forecast from a fitted Prophet model.
    """
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast.tail(periods)[["ds", "yhat", "yhat_lower", "yhat_upper"]].reset_index(drop=True)


def train_xgb_forecaster_with_intervals(
    series: pd.Series,
    lags: list | None = None,
    quantiles: tuple = FORECAST_QUANTILES,
    extra_features: pd.DataFrame | None = None,
    holiday_country: str | None = None,
) -> dict:
    """
    Train XGBoost models at specified quantiles (p10, p50, p90).

    Returns:
        dict like {"q10": (model, feat_cols), "q50": (model, feat_cols), "q90": (model, feat_cols)}
    """
    df = make_lag_features(
        series,
        lags=lags,
        extra_features=extra_features,
        holiday_country=holiday_country,
    )
    feat_cols = [c for c in df.columns if c != "target"]
    X_vals = df[feat_cols].values
    y_vals = df["target"].values

    result = {}
    label_map = {0.1: "q10", 0.5: "q50", 0.9: "q90"}
    for q in quantiles:
        model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:quantileerror",
            quantile_alpha=q,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_vals, y_vals)
        result[label_map.get(q, f"q{int(q * 100)}")] = (model, feat_cols)
    return result


def forecast_xgb_with_intervals(
    models_dict: dict,
    history: pd.Series,
    periods: int = 30,
    future_features: pd.DataFrame | None = None,
    holiday_country: str | None = None,
) -> pd.DataFrame:
    """
    Generate q10/q50/q90 forecasts and return a DataFrame with columns:
    ds, yhat, yhat_lower, yhat_upper.
    """
    forecasts = {}
    for key, (model, feat_cols) in models_dict.items():
        forecasts[key] = forecast_xgb(
            model,
            feat_cols,
            history,
            periods=periods,
            future_features=future_features,
            holiday_country=holiday_country,
        )

    q50 = forecasts.get("q50")
    if q50 is None:
        q50 = next(iter(forecasts.values()))
    q10 = forecasts.get("q10", q50)
    q90 = forecasts.get("q90", q50)

    return pd.DataFrame(
        {
            "ds": q50.index,
            "yhat": q50.values,
            "yhat_lower": q10.values,
            "yhat_upper": q90.values,
        }
    )
