"""
Weather feature pipeline using Open-Meteo free API (no key required).

Provides daily weather features for Portugal (default: Lisbon lat/lon)
aligned to the hotel booking dataset's date range.

Training:  build_weather_features(date_index)  — fetches/caches historical archive
Inference: build_inference_weather(future_dates) — blends 16-day API forecast
           with climatological means for dates beyond the forecast window.
"""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# Portugal (Lisbon) — hotel_bookings.csv is a Portuguese hotel dataset
DEFAULT_LAT      = 38.72
DEFAULT_LON      = -9.14
DEFAULT_TIMEZONE = "Europe/Lisbon"

_ARCHIVE_API  = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST_API = "https://api.open-meteo.com/v1/forecast"
_DAILY_VARS   = "temperature_2m_max,temperature_2m_min,precipitation_sum,sunshine_duration"

WEATHER_CACHE_PATH = Path("data/weather_portugal.csv")

WEATHER_FEATURE_COLS = [
    "temp_mean",
    "temp_max",
    "precip_mm",
    "sunshine_hours",
    "is_hot_day",
    "is_rainy_day",
    "temp_anomaly",
    "sunshine_anomaly",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _api_get(url: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def _parse_api_response(data: dict) -> pd.DataFrame:
    """Convert Open-Meteo JSON response → clean daily DataFrame."""
    df = pd.DataFrame(data["daily"]).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index = df.index.normalize()
    df["sunshine_hours"] = df["sunshine_duration"] / 3600.0
    df["temp_mean"] = (df["temperature_2m_max"] + df["temperature_2m_min"]) / 2.0
    df = df.rename(columns={
        "temperature_2m_max": "temp_max",
        "precipitation_sum": "precip_mm",
    })
    return df.drop(columns=["sunshine_duration", "temperature_2m_min"], errors="ignore")


def _climatological_means(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute smoothed day-of-year climatological means from historical data.
    Returns a DataFrame indexed 1–366 with mean columns.
    Used to fill in weather for dates with no archive/forecast coverage.
    """
    df = df.copy()
    df["_doy"] = df.index.dayofyear
    cols = [c for c in ["temp_mean", "temp_max", "precip_mm", "sunshine_hours"] if c in df.columns]
    clim = df.groupby("_doy")[cols].mean()

    # 15-day circular smoothing (wrap boundary so Jan 1 smooths into Dec and vice versa)
    pad = 15
    padded = pd.concat([clim.iloc[-pad:], clim, clim.iloc[:pad]])
    smoothed = padded.rolling(window=pad * 2 + 1, min_periods=1, center=True).mean()
    result = smoothed.iloc[pad:-pad].copy()
    result.index = clim.index
    return result


def _apply_anomalies(result: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    """Add temp_anomaly and sunshine_anomaly columns relative to climatological means."""
    doy = result.index.dayofyear
    clim_temp = clim["temp_mean"].reindex(doy).values
    clim_sun = clim["sunshine_hours"].reindex(doy).values if "sunshine_hours" in clim.columns else np.zeros(len(doy))
    result = result.copy()
    result["temp_anomaly"] = result["temp_mean"].values - clim_temp
    result["sunshine_anomaly"] = result["sunshine_hours"].values - clim_sun
    return result


def _fill_from_climatology(result: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN rows in result using climatological means for the same day-of-year."""
    result = result.copy()
    missing_mask = result["temp_mean"].isna()
    if not missing_mask.any():
        return result
    doy = result.index[missing_mask].dayofyear
    for col in clim.columns:
        if col in result.columns:
            result.loc[missing_mask, col] = clim[col].reindex(doy).values
    return result


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_hot_day"] = (df.get("temp_max", df.get("temp_mean", pd.Series(0, index=df.index))) >= 25).astype(float)
    df["is_rainy_day"] = (df.get("precip_mm", pd.Series(0, index=df.index)) >= 2.0).astype(float)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_historical_weather(
    start_date: str,
    end_date: str,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
) -> pd.DataFrame:
    """
    Fetch daily weather from Open-Meteo historical archive.
    Returns DataFrame indexed by date.
    """
    tz = urllib.parse.quote(DEFAULT_TIMEZONE)
    url = (
        f"{_ARCHIVE_API}?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily={_DAILY_VARS}&timezone={tz}"
    )
    return _parse_api_response(_api_get(url))


def _load_or_fetch_cache(
    date_min: pd.Timestamp,
    date_max: pd.Timestamp,
    cache_path: Path,
    lat: float,
    lon: float,
) -> pd.DataFrame | None:
    """Load cache; fetch and extend if date range exceeds cached range."""
    cache_path = Path(cache_path)
    cached = None
    if cache_path.exists():
        try:
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            cached.index = pd.DatetimeIndex(cached.index).normalize()
        except Exception as exc:
            logging.warning("Weather cache read failed: %s", exc)
            cached = None

    # Determine what range to fetch (only archive dates — not future)
    today = pd.Timestamp.now().normalize() - pd.Timedelta(days=6)  # archive has ~5-day lag
    fetch_start = date_min
    fetch_end = min(date_max, today)

    needs_fetch = (
        cached is None
        or (fetch_start < cached.index.min())
        or (fetch_end > cached.index.max())
    ) and fetch_start <= fetch_end

    if needs_fetch:
        try:
            fs = fetch_start.strftime("%Y-%m-%d")
            fe = fetch_end.strftime("%Y-%m-%d")
            fetched = fetch_historical_weather(fs, fe, lat=lat, lon=lon)
            if cached is not None:
                combined = pd.concat([cached, fetched])
                combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            else:
                combined = fetched
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(cache_path)
            cached = combined
            logging.info("Weather cache updated: %s to %s", fs, fe)
        except Exception as exc:
            logging.warning("Weather fetch failed: %s — using cached/climatology.", exc)

    return cached


def build_weather_features(
    date_index: pd.DatetimeIndex,
    cache_path: Path = WEATHER_CACHE_PATH,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
) -> pd.DataFrame:
    """
    Build weather feature matrix for a date index (training / historical dates).

    Fetches from Open-Meteo archive if not cached. Fills any remaining gaps
    with climatological (day-of-year) averages computed from the cached data.

    Returns DataFrame with WEATHER_FEATURE_COLS, indexed by date_index.
    Returns empty DataFrame if weather data is completely unavailable.
    """
    date_index = pd.DatetimeIndex(date_index).normalize()

    cached = _load_or_fetch_cache(
        date_index.min(), date_index.max(), cache_path, lat, lon
    )

    if cached is None or cached.empty:
        logging.warning("No weather data available — returning empty features.")
        return pd.DataFrame(index=date_index)

    clim = _climatological_means(cached)
    result = cached.reindex(date_index)
    result = _fill_from_climatology(result, clim)
    result = result.fillna(0.0)
    result = _add_derived_features(result)
    result = _apply_anomalies(result, clim)

    available = [c for c in WEATHER_FEATURE_COLS if c in result.columns]
    return result[available]


def build_inference_weather(
    future_dates: pd.DatetimeIndex,
    cache_path: Path = WEATHER_CACHE_PATH,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
) -> pd.DataFrame:
    """
    Build weather feature matrix for future forecast dates.

    - Days 1-16: live Open-Meteo 16-day forecast
    - Days 17+: climatological means from cached historical data

    Returns DataFrame with WEATHER_FEATURE_COLS, indexed by future_dates.
    Returns empty DataFrame if no data available at all.
    """
    future_dates = pd.DatetimeIndex(future_dates).normalize()

    # Fetch live forecast
    forecast_df = None
    try:
        tz = urllib.parse.quote(DEFAULT_TIMEZONE)
        url = (
            f"{_FORECAST_API}?latitude={lat}&longitude={lon}"
            f"&daily={_DAILY_VARS}&forecast_days=16&timezone={tz}"
        )
        forecast_df = _parse_api_response(_api_get(url, timeout=10))
    except Exception as exc:
        logging.warning("Live weather forecast unavailable: %s — using climatology.", exc)

    # Load historical cache (for climatological means)
    cached = None
    cache_path = Path(cache_path)
    if cache_path.exists():
        try:
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            cached.index = pd.DatetimeIndex(cached.index).normalize()
        except Exception:
            pass

    # Merge forecast into cache for a unified DataFrame
    if cached is not None and forecast_df is not None:
        combined = pd.concat([cached, forecast_df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    elif forecast_df is not None:
        combined = forecast_df
    elif cached is not None:
        combined = cached
    else:
        logging.warning("No weather data available for inference.")
        return pd.DataFrame(index=future_dates)

    clim = _climatological_means(combined)
    result = combined.reindex(future_dates)
    result = _fill_from_climatology(result, clim)
    result = result.fillna(0.0)
    result = _add_derived_features(result)
    result = _apply_anomalies(result, clim)

    available = [c for c in WEATHER_FEATURE_COLS if c in result.columns]
    return result[available]
