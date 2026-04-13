"""
Revenue-management analytics: pickup / pace, segment mix, and pricing suggestions.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.config import APP_CONFIG
from src.forecasting import (
    forecast_xgb,
    forecast_xgb_with_intervals,
    train_xgb_forecaster,
    train_xgb_forecaster_with_intervals,
)
from src.model_specs import FORECAST_LAGS


def prepare_booking_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add booking lifecycle dates needed for pickup and survival analytics.
    """
    result = df.copy()
    result["arrival_date"] = pd.to_datetime(result["arrival_date"], errors="coerce")
    result["lead_time"] = pd.to_numeric(result.get("lead_time", 0), errors="coerce").fillna(0).clip(lower=0)
    result["booking_date"] = result["arrival_date"] - pd.to_timedelta(result["lead_time"], unit="D")
    result["reservation_status_date"] = pd.to_datetime(
        result.get("reservation_status_date"),
        errors="coerce",
    )
    result["cancel_date"] = pd.NaT
    if "is_canceled" in result.columns:
        canceled = result["is_canceled"].fillna(0).astype(int) == 1
        result.loc[canceled, "cancel_date"] = result.loc[canceled, "reservation_status_date"]

    result["booking_window_bucket"] = pd.cut(
        result["lead_time"],
        bins=[-1, 3, 7, 30, 90, np.inf],
        labels=["0_3", "4_7", "8_30", "31_90", "91_plus"],
    ).astype(str)
    result["adr"] = pd.to_numeric(result.get("adr", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    return result.dropna(subset=["arrival_date", "booking_date"])


def default_snapshot_date(df: pd.DataFrame, periods: int = 30) -> pd.Timestamp:
    """
    Choose the latest snapshot date that still leaves `periods` future arrival dates.
    """
    arrival_max = pd.to_datetime(df["arrival_date"]).max()
    arrival_min = pd.to_datetime(df["arrival_date"]).min()
    candidate = arrival_max - pd.Timedelta(days=max(periods, 1))
    return max(candidate.normalize(), arrival_min.normalize())


def _active_bookings(
    timeline: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    arrival_date: pd.Timestamp | None = None,
    segment: str | None = None,
) -> pd.DataFrame:
    snapshot = pd.Timestamp(snapshot_date).normalize()
    active = timeline[timeline["booking_date"] <= snapshot]
    active = active[(active["cancel_date"].isna()) | (active["cancel_date"] > snapshot)]
    if arrival_date is not None:
        active = active[active["arrival_date"] == pd.Timestamp(arrival_date).normalize()]
    if segment is not None:
        active = active[active["market_segment"] == segment]
    return active


def _otb_count(
    timeline: pd.DataFrame,
    arrival_date: pd.Timestamp,
    snapshot_date: pd.Timestamp,
    segment: str | None = None,
) -> int:
    return int(len(_active_bookings(timeline, snapshot_date, arrival_date=arrival_date, segment=segment)))


def _otb_avg_adr(
    timeline: pd.DataFrame,
    arrival_date: pd.Timestamp,
    snapshot_date: pd.Timestamp,
    fallback_adr: float,
    segment: str | None = None,
) -> float:
    active = _active_bookings(timeline, snapshot_date, arrival_date=arrival_date, segment=segment)
    if len(active) > 0:
        return float(active["adr"].mean())
    return float(fallback_adr)


def _historical_same_weekday_adr(timeline: pd.DataFrame, arrival_date: pd.Timestamp, segment: str | None = None) -> float:
    subset = timeline[timeline["arrival_date"].dt.dayofweek == arrival_date.dayofweek]
    subset = subset[subset["arrival_date"].dt.month == arrival_date.month]
    if segment is not None:
        subset = subset[subset["market_segment"] == segment]
    if subset.empty:
        subset = timeline if segment is None else timeline[timeline["market_segment"] == segment]
    if subset.empty:
        return 0.0
    return float(subset["adr"].mean())


def build_pickup_report(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp | None = None,
    periods: int | None = None,
    comparison_offsets: tuple[int, ...] = (7, 14, 30),
    stly_offset_days: int = 364,
    segment: str | None = None,
    capacity: int | None = None,
) -> pd.DataFrame:
    """
    Build an on-the-books / pace report for future arrival dates.
    """
    periods = periods or APP_CONFIG.get("forecast_horizon_days", 30)
    timeline = prepare_booking_timeline(df)
    snapshot = snapshot_date or default_snapshot_date(timeline, periods=periods)
    snapshot = pd.Timestamp(snapshot).normalize()

    future_dates = pd.date_range(snapshot + pd.Timedelta(days=1), periods=periods, freq="D")
    rows = []
    for arrival_date in future_dates:
        fallback_adr = _historical_same_weekday_adr(timeline, arrival_date, segment=segment)
        otb_today = _otb_count(timeline, arrival_date, snapshot, segment=segment)
        row = {
            "date": arrival_date,
            "days_out": int((arrival_date - snapshot).days),
            "otb_today": otb_today,
            "current_adr": _otb_avg_adr(timeline, arrival_date, snapshot, fallback_adr, segment=segment),
        }

        for offset in comparison_offsets:
            row[f"otb_{offset}d_ago"] = _otb_count(
                timeline,
                arrival_date,
                snapshot - pd.Timedelta(days=offset),
                segment=segment,
            )
        row["pickup_7d"] = row["otb_today"] - row.get("otb_7d_ago", row["otb_today"])

        stly_arrival = arrival_date - pd.Timedelta(days=stly_offset_days)
        stly_snapshot = snapshot - pd.Timedelta(days=stly_offset_days)
        row["otb_stly_today"] = _otb_count(
            timeline,
            stly_arrival,
            stly_snapshot,
            segment=segment,
        )

        stly_base = row["otb_stly_today"]
        row["pace_vs_stly"] = (
            (row["otb_today"] - stly_base) / stly_base
            if stly_base > 0
            else np.nan
        )
        if capacity is not None:
            row["rooms_remaining"] = int(capacity - row["otb_today"])
        rows.append(row)

    report = pd.DataFrame(rows)
    return report


def build_segment_mix_report(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp | None = None,
    periods: int | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    """
    Segment mix of active on-the-books bookings for the future horizon.
    """
    periods = periods or APP_CONFIG.get("forecast_horizon_days", 30)
    top_n = top_n or APP_CONFIG.get("segment_top_n", 4)
    timeline = prepare_booking_timeline(df)
    snapshot = snapshot_date or default_snapshot_date(timeline, periods=periods)
    snapshot = pd.Timestamp(snapshot).normalize()

    active = _active_bookings(timeline, snapshot)
    active = active[
        (active["arrival_date"] > snapshot)
        & (active["arrival_date"] <= snapshot + pd.Timedelta(days=periods))
    ]
    if active.empty or "market_segment" not in active.columns:
        return pd.DataFrame(columns=["date", "segment", "otb_bookings", "segment_share"])

    top_segments = (
        active["market_segment"].value_counts().head(top_n).index.tolist()
    )
    active = active.copy()
    active["segment_group"] = np.where(
        active["market_segment"].isin(top_segments),
        active["market_segment"],
        "Other",
    )
    grouped = (
        active.groupby(["arrival_date", "segment_group"])
        .size()
        .rename("otb_bookings")
        .reset_index()
        .rename(columns={"arrival_date": "date", "segment_group": "segment"})
    )
    totals = grouped.groupby("date")["otb_bookings"].transform("sum")
    grouped["segment_share"] = grouped["otb_bookings"] / totals
    return grouped.sort_values(["date", "otb_bookings"], ascending=[True, False]).reset_index(drop=True)


def build_segment_daily_series(
    df: pd.DataFrame,
    top_n: int | None = None,
) -> pd.DataFrame:
    """
    Daily arrival-date demand split by market segment.
    """
    top_n = top_n or APP_CONFIG.get("segment_top_n", 4)
    timeline = prepare_booking_timeline(df)
    if "market_segment" not in timeline.columns or timeline.empty:
        return pd.DataFrame()

    top_segments = timeline["market_segment"].value_counts().head(top_n).index.tolist()
    timeline = timeline.copy()
    timeline["segment_group"] = np.where(
        timeline["market_segment"].isin(top_segments),
        timeline["market_segment"],
        "Other",
    )
    grouped = (
        timeline.groupby(["arrival_date", "segment_group"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    full_range = pd.date_range(grouped.index.min(), grouped.index.max(), freq="D")
    grouped = grouped.reindex(full_range, fill_value=0)
    grouped.index.name = "arrival_date"
    return grouped


def train_segment_demand_models(
    df: pd.DataFrame,
    top_n: int | None = None,
    lags: list | None = None,
    min_history: int = 90,
) -> dict:
    """
    Train separate demand forecasters per segment with interval models.
    """
    segment_df = build_segment_daily_series(df, top_n=top_n)
    models = {}
    for segment in segment_df.columns:
        series = segment_df[segment].astype(float)
        if len(series) < min_history:
            continue
        models[segment] = {
            "series": series,
            "point": train_xgb_forecaster(series, lags=lags or FORECAST_LAGS),
            "intervals": train_xgb_forecaster_with_intervals(series, lags=lags or FORECAST_LAGS),
        }
    return models


def forecast_segment_demand(
    segment_models: dict,
    periods: int | None = None,
) -> pd.DataFrame:
    """
    Forecast each trained segment and return long-format output.
    """
    periods = periods or APP_CONFIG.get("forecast_horizon_days", 30)
    rows = []
    for segment, bundle in segment_models.items():
        interval_df = forecast_xgb_with_intervals(
            bundle["intervals"],
            bundle["series"],
            periods=periods,
        )
        interval_df["segment"] = segment
        rows.append(interval_df)
    if not rows:
        return pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper", "segment"])
    return pd.concat(rows, ignore_index=True)


def _elasticity_training_frame(df: pd.DataFrame, segment: str | None = None) -> pd.DataFrame:
    timeline = prepare_booking_timeline(df)
    if segment is not None:
        timeline = timeline[timeline["market_segment"] == segment]
    if timeline.empty:
        return pd.DataFrame()

    daily = (
        timeline.groupby("arrival_date")
        .agg(
            demand=("arrival_date", "size"),
            avg_adr=("adr", "mean"),
            avg_lead_time=("lead_time", "mean"),
            share_short=("lead_time", lambda x: float((pd.Series(x) <= 7).mean())),
            share_long=("lead_time", lambda x: float((pd.Series(x) >= 30).mean())),
        )
        .reset_index()
    )
    daily = daily[(daily["demand"] > 0) & (daily["avg_adr"] > 0)].copy()
    if daily.empty:
        return daily
    daily["dow"] = daily["arrival_date"].dt.day_name()
    daily["month"] = daily["arrival_date"].dt.month.astype(str)
    daily["log_demand"] = np.log(daily["demand"])
    daily["log_adr"] = np.log(daily["avg_adr"])
    return daily


def _fit_one_elasticity_model(training_df: pd.DataFrame) -> dict | None:
    if len(training_df) < 30 or training_df["log_adr"].nunique() < 5:
        return None

    X = pd.concat(
        [
            training_df[["log_adr", "avg_lead_time", "share_short", "share_long"]],
            pd.get_dummies(training_df["dow"], prefix="dow", dtype=float),
            pd.get_dummies(training_df["month"], prefix="month", dtype=float),
        ],
        axis=1,
    )
    y = training_df["log_demand"].values
    model = LinearRegression()
    model.fit(X, y)

    elasticity = float(model.coef_[0])
    if not np.isfinite(elasticity) or elasticity >= -0.05:
        elasticity = -0.35

    return {
        "model": model,
        "feature_columns": X.columns.tolist(),
        "elasticity": elasticity,
        "avg_adr": float(training_df["avg_adr"].mean()),
        "avg_demand": float(training_df["demand"].mean()),
        "n_rows": int(len(training_df)),
    }


def fit_price_elasticity_models(
    df: pd.DataFrame,
    top_n: int | None = None,
) -> dict:
    """
    Fit a log-log price elasticity model overall and for the largest segments.
    """
    top_n = top_n or APP_CONFIG.get("segment_top_n", 4)
    timeline = prepare_booking_timeline(df)
    segments = timeline["market_segment"].value_counts().head(top_n).index.tolist()

    models = {}
    overall = _fit_one_elasticity_model(_elasticity_training_frame(timeline))
    if overall is not None:
        models["overall"] = overall

    for segment in segments:
        fitted = _fit_one_elasticity_model(_elasticity_training_frame(timeline, segment=segment))
        if fitted is not None:
            models[segment] = fitted

    return models


def recommend_dynamic_prices(
    forecast_df: pd.DataFrame,
    pickup_report: pd.DataFrame,
    elasticity_models: dict,
    capacity: int,
) -> pd.DataFrame:
    """
    Convert demand forecasts + pickup into directional ADR suggestions.
    """
    if forecast_df.empty:
        return pd.DataFrame()

    base_model = elasticity_models.get("overall", {})
    default_adr = float(base_model.get("avg_adr", 0.0))
    beta = float(base_model.get("elasticity", -0.35))

    forecast = forecast_df.copy()
    forecast["date"] = pd.to_datetime(forecast.get("ds", forecast.get("date")))
    merged = forecast.merge(pickup_report, on="date", how="left")
    if "yhat_lower" not in merged.columns:
        merged["yhat_lower"] = merged["yhat"]
    if "yhat_upper" not in merged.columns:
        merged["yhat_upper"] = merged["yhat"]

    rows = []
    for row in merged.itertuples(index=False):
        baseline_demand = max(float(row.yhat), float(getattr(row, "otb_today", 0.0) or 0.0))
        current_adr = float(getattr(row, "current_adr", default_adr) or default_adr)
        pace_vs_stly = getattr(row, "pace_vs_stly", np.nan)
        otb_today = float(getattr(row, "otb_today", 0.0) or 0.0)

        if baseline_demand > capacity:
            target_demand = float(capacity)
            note = "Raise ADR to protect occupancy above capacity."
        elif np.isfinite(pace_vs_stly) and pace_vs_stly < -0.10 and baseline_demand < capacity * 0.8:
            target_demand = min(float(capacity), baseline_demand * 1.08)
            note = "Pace is below STLY; ease ADR to stimulate demand."
        elif np.isfinite(pace_vs_stly) and pace_vs_stly > 0.10 and baseline_demand >= capacity * 0.9:
            target_demand = float(capacity)
            note = "Pace is ahead of STLY; hold or lift ADR."
        else:
            target_demand = baseline_demand
            note = "Hold ADR; demand looks balanced."

        if current_adr <= 0 or baseline_demand <= 0 or abs(beta) < 0.05:
            recommended_adr = current_adr
        else:
            price_change_log = math.log(max(target_demand, 1.0) / max(baseline_demand, 1.0)) / beta
            price_change_log = float(np.clip(price_change_log, -0.20, 0.25))
            recommended_adr = float(current_adr * math.exp(price_change_log))

        expected_demand = (
            baseline_demand * ((recommended_adr / current_adr) ** beta)
            if current_adr > 0 and baseline_demand > 0
            else baseline_demand
        )
        expected_occupancy = min(float(capacity), max(otb_today, expected_demand))

        rows.append(
            {
                "date": row.date,
                "forecast_demand": baseline_demand,
                "forecast_low": float(row.yhat_lower),
                "forecast_high": float(row.yhat_upper),
                "otb_today": otb_today,
                "pace_vs_stly": pace_vs_stly,
                "current_adr": current_adr,
                "recommended_adr": recommended_adr,
                "adr_change": recommended_adr - current_adr,
                "expected_occupancy": expected_occupancy,
                "capacity": capacity,
                "recommendation": note,
            }
        )

    return pd.DataFrame(rows)
