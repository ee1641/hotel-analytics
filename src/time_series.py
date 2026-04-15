"""
Date construction and daily time-series aggregation for hotel bookings.
"""
import logging

import pandas as pd

from src.preprocessing import ensure_revenue_columns


_MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}


def build_arrival_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a proper arrival_date datetime column from:
      arrival_date_year, arrival_date_month (string name), arrival_date_day_of_month
    """
    df = df.copy()
    if "arrival_date" in df.columns:
        df["arrival_date"] = pd.to_datetime(df["arrival_date"], errors="coerce")
        return df

    month_num = df["arrival_date_month"].map(_MONTH_MAP)
    df["arrival_date"] = pd.to_datetime({
        "year": df["arrival_date_year"],
        "month": month_num,
        "day": df["arrival_date_day_of_month"],
    })
    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate booking-level rows to a complete daily time series.

    Returns a DataFrame indexed by date (no gaps, zero-filled) with:
      daily_revenue       : sum of booking revenue
      daily_bookings      : count of all bookings
      daily_cancellations : count of canceled bookings
    """
    df = ensure_revenue_columns(df)

    aggregations = {
        "daily_booked_revenue": ("booked_revenue", "sum"),
        "daily_realized_revenue": ("realized_revenue", "sum"),
        "daily_cancellations": ("is_canceled", "sum"),
    }
    if "expected_realized_revenue" in df.columns:
        aggregations["daily_expected_realized_revenue"] = ("expected_realized_revenue", "sum")

    grouped = df.groupby("arrival_date").agg(**aggregations)
    grouped["daily_bookings"] = df.groupby("arrival_date").size()
    if "daily_expected_realized_revenue" not in grouped.columns:
        grouped["daily_expected_realized_revenue"] = grouped["daily_realized_revenue"]
    grouped["daily_revenue"] = grouped["daily_booked_revenue"]
    grouped = grouped[
        [
            "daily_booked_revenue",
            "daily_realized_revenue",
            "daily_expected_realized_revenue",
            "daily_revenue",
            "daily_bookings",
            "daily_cancellations",
        ]
    ]
    # Fill gaps so lag features and SARIMA differencing are not corrupted
    full_range = pd.date_range(grouped.index.min(), grouped.index.max(), freq="D")
    grouped = grouped.reindex(full_range, fill_value=0)
    grouped.index.name = "arrival_date"
    return grouped


def aggregate_daily_by_hotel_type(
    df: pd.DataFrame,
    min_history: int = 31,
) -> dict[str, pd.DataFrame]:
    """
    Segment bookings by the `hotel` column and aggregate each type to a daily
    time series using the same logic as aggregate_daily().

    Returns a dict mapping hotel type name → daily DataFrame (same schema as
    aggregate_daily()). Hotel types with fewer than `min_history` daily
    observations are skipped with a warning.
    """
    result: dict[str, pd.DataFrame] = {}
    if "hotel" not in df.columns:
        logging.warning(
            "aggregate_daily_by_hotel_type: no 'hotel' column found; returning empty dict."
        )
        return result
    for hotel_type in df["hotel"].dropna().unique():
        subset = df[df["hotel"] == hotel_type]
        daily = aggregate_daily(subset)
        if len(daily) < min_history:
            logging.warning(
                "Hotel type '%s' has only %d days of history (min %d); skipping.",
                hotel_type,
                len(daily),
                min_history,
            )
            continue
        result[str(hotel_type)] = daily
    return result


def expand_stay_nights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand each non-cancelled booking to one row per occupied night.

    Returns a DataFrame with columns: date, revenue_per_night.
    Cancelled bookings are excluded.
    """
    active = df[df["is_canceled"] == 0].copy()
    if active.empty:
        return pd.DataFrame(columns=["date", "revenue_per_night"])

    total_nights = active["total_nights"].clip(lower=1).astype(int)
    repeated = active.loc[active.index.repeat(total_nights)].copy()
    repeated["_night"] = repeated.groupby(level=0).cumcount()
    repeated["date"] = repeated["arrival_date"] + pd.to_timedelta(
        repeated["_night"].astype(int), unit="D"
    )
    return (
        repeated[["date", "adr"]]
        .rename(columns={"adr": "revenue_per_night"})
        .reset_index(drop=True)
    )


def aggregate_stay_nights(stay_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate stay-night rows to a complete daily time series.

    Returns a DataFrame indexed by date (no gaps, zero-filled) with:
      occupied_rooms      : number of rooms occupied on each night
      daily_stay_revenue  : sum of ADR across all occupied rooms
    """
    if stay_df.empty:
        return pd.DataFrame(columns=["occupied_rooms", "daily_stay_revenue"])

    result = stay_df.groupby("date").agg(
        occupied_rooms=("revenue_per_night", "count"),
        daily_stay_revenue=("revenue_per_night", "sum"),
    ).sort_index()
    full_range = pd.date_range(result.index.min(), result.index.max(), freq="D")
    result = result.reindex(full_range, fill_value=0)
    result.index.name = "date"
    return result
