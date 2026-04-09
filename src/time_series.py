"""
Date construction and daily time-series aggregation for hotel bookings.
"""
import pandas as pd


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
    grouped = df.groupby("arrival_date").agg(
        daily_revenue=("revenue", "sum"),
        daily_bookings=("hotel", "count"),
        daily_cancellations=("is_canceled", "sum"),
    )
    # Fill gaps so lag features and SARIMA differencing are not corrupted
    full_range = pd.date_range(grouped.index.min(), grouped.index.max(), freq="D")
    grouped = grouped.reindex(full_range, fill_value=0)
    grouped.index.name = "arrival_date"
    return grouped
