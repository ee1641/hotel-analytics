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


def _easter_sunday(year: int) -> pd.Timestamp:
    """Anonymous Gregorian algorithm for Easter Sunday."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    month = (h + el - 7 * m + 114) // 31
    day = ((h + el - 7 * m + 114) % 31) + 1
    return pd.Timestamp(year=year, month=month, day=day)


def _school_holiday_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Portuguese school holiday indicators.
    Returns DataFrame with is_summer_holiday, is_easter_holiday, is_christmas_holiday.
    """
    result = pd.DataFrame(index=index)
    is_summer = pd.Series(False, index=index)
    is_easter = pd.Series(False, index=index)
    is_christmas = pd.Series(False, index=index)

    for year in range(index.min().year - 1, index.max().year + 2):
        # Summer: mid-June to mid-September
        summer_start = pd.Timestamp(year=year, month=6, day=15)
        summer_end   = pd.Timestamp(year=year, month=9, day=15)
        is_summer |= (index >= summer_start) & (index <= summer_end)

        # Easter: 10 days before Easter Sunday through 7 days after
        try:
            easter = _easter_sunday(year)
            easter_start = easter - pd.Timedelta(days=10)
            easter_end   = easter + pd.Timedelta(days=7)
            is_easter |= (index >= easter_start) & (index <= easter_end)
        except Exception:
            pass

        # Christmas / New Year: Dec 20 – Jan 3
        xmas_start = pd.Timestamp(year=year, month=12, day=20)
        xmas_end   = pd.Timestamp(year=year + 1, month=1, day=3)
        is_christmas |= (index >= xmas_start) & (index <= xmas_end)

    result["is_summer_holiday"]   = is_summer.astype(int)
    result["is_easter_holiday"]   = is_easter.astype(int)
    result["is_christmas_holiday"] = is_christmas.astype(int)
    result["is_school_holiday"]   = (is_summer | is_easter | is_christmas).astype(int)
    return result


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
    school_df = _school_holiday_features(index)
    return pd.concat([df, holiday_df, school_df], axis=1)


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


def tune_xgb_forecaster(
    series: pd.Series,
    lags: list | None = None,
    n_iter: int | None = None,
    n_splits: int | None = None,
    holiday_country: str | None = None,
) -> dict:
    """
    Find the best XGBoost hyperparameters for a daily time series using
    expanding-window time-series cross-validation.

    Tries `n_iter` random parameter combinations, evaluating each with
    `TimeSeriesSplit(n_splits)` folds.  Returns the param dict that achieves
    the lowest mean-fold MAE, ready to be unpacked into train_xgb_forecaster.
    Falls back to safe defaults if anything goes wrong.
    """
    from sklearn.model_selection import TimeSeriesSplit
    from src.model_specs import FORECAST_TUNE_N_ITER, FORECAST_TUNE_N_SPLITS

    _n_iter = n_iter if n_iter is not None else FORECAST_TUNE_N_ITER
    _n_splits = n_splits if n_splits is not None else FORECAST_TUNE_N_SPLITS

    _defaults = {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
                 "subsample": 0.9, "colsample_bytree": 0.9}

    try:
        df = make_lag_features(series, lags=lags, holiday_country=holiday_country)
        feat_cols = [c for c in df.columns if c != "target"]
        X = df[feat_cols].values
        y = df["target"].values

        if len(X) < _n_splits * 30:
            return _defaults

        param_grid = {
            "n_estimators": [200, 400, 600],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
        }

        rng = np.random.RandomState(42)
        candidates = [
            {k: rng.choice(v) for k, v in param_grid.items()}
            for _ in range(_n_iter)
        ]

        tscv = TimeSeriesSplit(n_splits=_n_splits)
        best_params = _defaults
        best_mae = float("inf")

        for params in candidates:
            fold_maes = []
            for train_idx, val_idx in tscv.split(X):
                model = XGBRegressor(
                    **params,
                    objective="reg:squarederror",
                    random_state=42,
                    verbosity=0,
                )
                model.fit(X[train_idx], y[train_idx])
                preds = model.predict(X[val_idx])
                fold_maes.append(float(np.mean(np.abs(y[val_idx] - preds))))
            avg_mae = float(np.mean(fold_maes))
            if avg_mae < best_mae:
                best_mae = avg_mae
                best_params = params

        return best_params
    except Exception:
        return _defaults


def train_xgb_forecaster(
    series: pd.Series,
    lags: list | None = None,
    n_estimators: int = 300,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    extra_features: pd.DataFrame | None = None,
    holiday_country: str | None = None,
    **kwargs,
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
        n_estimators=kwargs.get("n_estimators", n_estimators),
        max_depth=kwargs.get("max_depth", max_depth),
        learning_rate=kwargs.get("learning_rate", learning_rate),
        subsample=kwargs.get("subsample", 0.9),
        colsample_bytree=kwargs.get("colsample_bytree", 0.9),
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


def train_xgb_direct_multistep(
    series: pd.Series,
    horizons: int = 30,
    lags: list | None = None,
    holiday_country: str | None = None,
    **kwargs,
) -> list[tuple[XGBRegressor, list]]:
    """
    Train one XGBRegressor per forecast horizon 1..horizons.
    Model h is trained to predict the value h days ahead from the last observed
    day, using only actual historical values as features.
    Avoids the error accumulation of iterative one-step-ahead forecasting.

    Returns a list of (model, feat_cols) tuples, one per horizon.
    """
    models = []
    for h in range(1, horizons + 1):
        shifted_target = series.shift(-h)
        df = make_lag_features(series, lags=lags, holiday_country=holiday_country)
        valid_idx = df.index[df.index.isin(shifted_target.dropna().index)]
        df_h = df.loc[valid_idx].copy()
        y_h = shifted_target.loc[valid_idx].values
        feat_cols = [c for c in df_h.columns if c != "target"]
        model = XGBRegressor(
            n_estimators=kwargs.get("n_estimators", 300),
            max_depth=kwargs.get("max_depth", 5),
            learning_rate=kwargs.get("learning_rate", 0.05),
            subsample=kwargs.get("subsample", 0.9),
            colsample_bytree=kwargs.get("colsample_bytree", 0.9),
            objective="reg:squarederror",
            random_state=42,
            verbosity=0,
        )
        model.fit(df_h[feat_cols].values, y_h)
        models.append((model, feat_cols))
    return models


def forecast_xgb_direct(
    models: list[tuple[XGBRegressor, list]],
    history: pd.Series,
    holiday_country: str | None = None,
) -> pd.Series:
    """
    Direct multi-step forecast using per-horizon models.
    Each model sees only actual historical values as features — no error accumulation.
    The `extended` list is never modified, so all 30 predictions use the same
    historical context.
    """
    history = pd.Series(history).sort_index()
    extended = list(history.values.astype(float))
    future_dates = pd.date_range(
        history.index[-1] + pd.Timedelta(days=1),
        periods=len(models),
        freq="D",
    )
    predictions = []
    for i, (model, feat_cols) in enumerate(models):
        row_df = _build_forecast_row(
            future_dates[i],
            feat_cols,
            extended,
            holiday_country=holiday_country,
        )
        pred = float(model.predict(row_df.values)[0])
        predictions.append(pred)
        # Key: do NOT append pred to extended.
        # Every model uses actual historical values only.
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


def find_ensemble_weights(
    xgb_preds: np.ndarray | list,
    prophet_preds: np.ndarray | list,
    actuals: np.ndarray | list,
) -> float:
    """
    Grid-search the XGBoost weight w in the blend: w * XGB + (1-w) * Prophet.
    Optimises mean absolute error on the provided validation window.
    Returns the weight (0.0 – 1.0) that minimises MAE.
    """
    xgb_arr     = np.asarray(xgb_preds, dtype=float)
    prophet_arr = np.asarray(prophet_preds, dtype=float)
    actual_arr  = np.asarray(actuals, dtype=float)

    best_w, best_mae = 1.0, float("inf")
    for w in np.arange(0.0, 1.01, 0.05):
        blend = w * xgb_arr + (1.0 - w) * prophet_arr
        mae = float(np.mean(np.abs(blend - actual_arr)))
        if mae < best_mae:
            best_mae = mae
            best_w = float(w)
    return best_w


def blend_forecasts(
    xgb_forecast: pd.Series,
    prophet_forecast: pd.Series | pd.DataFrame,
    xgb_weight: float,
) -> pd.Series:
    """
    Weighted blend of XGBoost and Prophet forecasts.
    prophet_forecast may be a Series or a DataFrame with a 'yhat' column.
    """
    if isinstance(prophet_forecast, pd.DataFrame) and "yhat" in prophet_forecast.columns:
        prophet_values = prophet_forecast["yhat"].values
    else:
        prophet_values = np.asarray(prophet_forecast, dtype=float)

    blended = xgb_weight * xgb_forecast.values + (1.0 - xgb_weight) * prophet_values
    return pd.Series(blended, index=xgb_forecast.index, name="forecast_ensemble")


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
    **kwargs,
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
            n_estimators=kwargs.get("n_estimators", 300),
            max_depth=kwargs.get("max_depth", 5),
            learning_rate=kwargs.get("learning_rate", 0.05),
            subsample=kwargs.get("subsample", 0.9),
            colsample_bytree=kwargs.get("colsample_bytree", 0.9),
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
