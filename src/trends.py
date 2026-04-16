"""
Google Trends demand-signal pipeline for hotel forecasting.

Uses pytrends (unofficial Google Trends API) to fetch weekly search interest
for hotel-related keywords in Portugal, then builds daily features aligned to
the booking dataset.

Key insight: hotel searches typically precede arrivals by 2–4 weeks, so lagged
trends values are a *leading indicator* of upcoming demand.

Training:  build_trends_features(date_index)         — fetches/caches historical weekly data
Inference: build_inference_trends(future_dates)       — extends with most-recent available data

Cache: data/google_trends.csv  (weekly index values, raw 0-100 scale)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TRENDS_CACHE_PATH = Path("data/google_trends.csv")

# Keywords fetched together so pytrends normalises them on a common 0-100 scale.
# "hotel Portugal" anchors the scale; regional terms capture resort vs city split.
TRENDS_KEYWORDS = ["hotel Portugal", "hotel Algarve", "hotel Lisboa"]
TRENDS_GEO = "PT"

# Column names after fetching
TRENDS_RAW_COLS = ["trends_hotel_pt", "trends_hotel_algarve", "trends_hotel_lisboa"]

# Final feature columns exposed to the forecasting model
TRENDS_FEATURE_COLS = [
    "trends_composite",       # weighted average of the three keywords
    "trends_yoy_anomaly",     # composite vs same week last year
    "trends_momentum_4w",     # 4-week rolling change in composite
    "trends_lag14d",          # composite lagged 14 days (2-week lead signal)
    "trends_lag21d",          # composite lagged 21 days (3-week lead signal)
]


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def _fetch_from_api(
    start_date: str,
    end_date: str,
    retries: int = 3,
    backoff: float = 5.0,
) -> pd.DataFrame:
    """
    Fetch weekly Google Trends interest for TRENDS_KEYWORDS in Portugal.

    Returns a DataFrame with weekly DatetimeIndex and columns matching
    TRENDS_RAW_COLS. Values are 0-100 (relative interest index, normalised
    across all keywords in the same request).
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        raise ImportError("pytrends is required: pip install pytrends")

    timeframe = f"{start_date} {end_date}"
    for attempt in range(retries):
        try:
            pt = TrendReq(hl="en-US", tz=0, timeout=(10, 30))
            pt.build_payload(
                TRENDS_KEYWORDS,
                cat=0,
                timeframe=timeframe,
                geo=TRENDS_GEO,
                gprop="",
            )
            raw = pt.interest_over_time()
            if raw.empty:
                raise ValueError("Empty response from Google Trends API")
            raw = raw.drop(columns=["isPartial"], errors="ignore")
            raw.columns = TRENDS_RAW_COLS
            raw.index = pd.to_datetime(raw.index).normalize()
            return raw.astype(float)
        except Exception as exc:
            log.warning("Trends fetch attempt %d/%d failed: %s", attempt + 1, retries, exc)
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"Google Trends fetch failed after {retries} attempts")


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def _load_cache() -> pd.DataFrame | None:
    if not TRENDS_CACHE_PATH.exists():
        return None
    try:
        df = pd.read_csv(TRENDS_CACHE_PATH, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index).normalize()
        return df
    except Exception as exc:
        log.warning("Failed to read trends cache: %s", exc)
        return None


def _save_cache(df: pd.DataFrame) -> None:
    TRENDS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TRENDS_CACHE_PATH)


def load_or_fetch_trends(
    start_date: str = "2014-01-01",
    end_date: str = "2018-12-31",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return raw weekly trends DataFrame. Loads from cache when available and
    the cache covers the requested date range; fetches from API otherwise.
    """
    cache = _load_cache() if not force_refresh else None

    if cache is not None:
        cache_start = cache.index.min()
        cache_end = cache.index.max()
        req_start = pd.Timestamp(start_date)
        req_end = pd.Timestamp(end_date)
        if cache_start <= req_start and cache_end >= req_end - pd.Timedelta(days=7):
            log.info("Trends: loaded %d weeks from cache", len(cache))
            return cache

    log.info("Trends: fetching from Google Trends API (%s → %s)", start_date, end_date)
    raw = _fetch_from_api(start_date, end_date)
    _save_cache(raw)
    log.info("Trends: cached %d weeks to %s", len(raw), TRENDS_CACHE_PATH)
    return raw


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _build_composite(raw: pd.DataFrame) -> pd.Series:
    """Weighted composite: Portugal 50%, Algarve 30%, Lisboa 20%."""
    weights = [0.5, 0.3, 0.2]
    cols = [c for c in TRENDS_RAW_COLS if c in raw.columns]
    w = weights[: len(cols)]
    total = sum(w)
    composite = sum(raw[c] * (wi / total) for c, wi in zip(cols, w))
    return composite.rename("trends_composite")


def _interpolate_to_daily(weekly: pd.Series, date_index: pd.DatetimeIndex) -> pd.Series:
    """
    Linearly interpolate weekly values to a daily DatetimeIndex.
    Values outside the weekly range are forward/backward filled.
    """
    full_idx = date_index.union(weekly.index).sort_values()
    reindexed = weekly.reindex(full_idx).interpolate(method="time", limit_direction="both")
    return reindexed.reindex(date_index)


def build_trends_features(
    date_index: pd.DatetimeIndex,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Build daily trends features aligned to `date_index`.

    Returns a DataFrame with columns matching TRENDS_FEATURE_COLS.
    Falls back to zero-filled features if the API is unavailable or pytrends
    is not installed — the model degrades gracefully without the signal.
    """
    try:
        start = (date_index.min() - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
        end   = (date_index.max() + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        raw = load_or_fetch_trends(start_date=start, end_date=end, force_refresh=force_refresh)
    except Exception as exc:
        log.warning("Trends unavailable, returning zeros: %s", exc)
        return pd.DataFrame(0.0, index=date_index, columns=TRENDS_FEATURE_COLS)

    composite_weekly = _build_composite(raw)
    composite = _interpolate_to_daily(composite_weekly, date_index)

    result = pd.DataFrame(index=date_index)
    result["trends_composite"] = composite

    # Year-over-year anomaly: composite vs same week 52 weeks ago
    lag_52w = composite_weekly.shift(52)
    lag_52w_daily = _interpolate_to_daily(lag_52w, date_index)
    denom = lag_52w_daily.replace(0, np.nan)
    result["trends_yoy_anomaly"] = ((composite - lag_52w_daily) / denom).fillna(0.0).clip(-2, 2)

    # 4-week momentum: rolling 28-day change in composite
    result["trends_momentum_4w"] = composite.diff(28).fillna(0.0)

    # Lagged composites as leading indicators for demand
    result["trends_lag14d"] = composite.shift(14).bfill().fillna(0.0)
    result["trends_lag21d"] = composite.shift(21).bfill().fillna(0.0)

    return result.fillna(0.0)


def build_inference_trends(
    future_dates: pd.DatetimeIndex,
    history_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Build trends features for a future forecast window.

    For lead-time features (lag14d, lag21d): uses the most recently available
    composite values shifted appropriately.
    For YoY anomaly and momentum: extends using the last known trend direction.
    """
    all_dates = history_index.union(future_dates).sort_values()
    all_features = build_trends_features(all_dates)
    return all_features.reindex(future_dates)
