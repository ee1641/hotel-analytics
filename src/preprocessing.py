"""
Data loading, cleaning, feature engineering, and encoding for hotel bookings.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.ingestion import normalize_uploaded_dataframe


def load_data(path: str) -> pd.DataFrame:
    """Load hotel_bookings.csv and return raw DataFrame."""
    return normalize_uploaded_dataframe(pd.read_csv(path))


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix data quality issues:
    - remove accidental exact duplicate booking rows
    - agent/company NULL  -> 0 (no agent / no company)
    - country NULL        -> 'Unknown'
    - children NULL       -> 0
    - adr < 0             -> 0 (invalid rates)
    """
    df = remove_accidental_duplicate_bookings(df.copy())

    defaults = {
        "agent": 0,
        "company": 0,
        "country": "Unknown",
        "children": 0,
        "babies": 0,
        "adults": 2,
        "adr": 0.0,
        "stays_in_weekend_nights": 0,
        "stays_in_week_nights": 1,
        "hotel": "Custom Hotel",
        "meal": "BB",
        "market_segment": "Direct",
        "distribution_channel": "Direct",
        "customer_type": "Transient",
        "reserved_room_type": "A",
        "total_of_special_requests": 0,
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "required_car_parking_spaces": 0,
    }
    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value

    df["agent"] = df["agent"].fillna(0).astype(int)
    df["company"] = df["company"].fillna(0).astype(int)
    df["country"] = df["country"].fillna("Unknown")
    df["children"] = df["children"].fillna(0).astype(int)
    df["babies"] = df["babies"].fillna(0).astype(int)
    df["adults"] = df["adults"].fillna(2).astype(int)
    df["adr"] = pd.to_numeric(df["adr"], errors="coerce").fillna(0.0).clip(lower=0)
    return df


def remove_accidental_duplicate_bookings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate booking rows before downstream analysis.

    This is intentionally conservative: only rows that are identical across all
    currently available columns are removed.
    """
    return df.drop_duplicates().reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive new columns.
    NOTE: reservation_status is kept here — it is needed by time_series.py
    for is_no_show creation and is dropped later inside encode_categoricals.
    """
    df = df.copy()
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df = ensure_revenue_columns(df)
    if "reservation_status" not in df.columns:
        df["reservation_status"] = "Check-Out"
    df["is_no_show"] = (df["reservation_status"] == "No-Show").astype(int)
    return df


def ensure_revenue_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the shared revenue columns exist on any booking-level DataFrame.

    This is idempotent so callers can safely apply it to filtered frames or
    older cached/uploaded frames before aggregation.
    """
    df = df.copy()

    def _series_or_default(column_name: str, default_value: float = 0.0) -> pd.Series:
        if column_name in df.columns:
            return pd.to_numeric(df[column_name], errors="coerce").fillna(default_value)
        return pd.Series(default_value, index=df.index, dtype=float)

    if "total_nights" not in df.columns:
        weekend = _series_or_default("stays_in_weekend_nights", 0.0)
        week = _series_or_default("stays_in_week_nights", 0.0)
        df["total_nights"] = weekend + week

    if "revenue" not in df.columns:
        adr = _series_or_default("adr", 0.0).clip(lower=0)
        df["revenue"] = (adr * df["total_nights"]).clip(lower=0)

    if "booked_revenue" not in df.columns:
        df["booked_revenue"] = df["revenue"]

    if "realized_revenue" not in df.columns:
        is_canceled = _series_or_default("is_canceled", 0.0)
        df["realized_revenue"] = (df["revenue"] * (1 - is_canceled.astype(float))).clip(lower=0)

    return df


# These columns leak the outcome or are replaced by arrival_date
_LEAKY_COLS = [
    "reservation_status",
    "reservation_status_date",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
]

# Public list: features unavailable at booking time (used by cancellation pipeline)
LEAKY_COLUMNS = [
    "reservation_status",        # directly encodes the target
    "reservation_status_date",   # directly encodes the target
    "deposit_type",              # set after booking confirmation, not at booking time
    "assigned_room_type",        # assigned at check-in, not at booking time
    "booking_changes",           # accumulated post-booking
    "days_in_waiting_list",      # accumulated post-booking
]

# Columns to label-encode before training
_CATEGORICAL_COLS = [
    "hotel",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
]


_SEASON_MAP = {
    12: 3, 1: 3, 2: 3,   # winter
    3: 0, 4: 0, 5: 0,    # spring
    6: 1, 7: 1, 8: 1,    # summer
    9: 2, 10: 2, 11: 2,  # fall
}


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Requires arrival_date column (datetime). Adds:
      booking_month      : int 1-12
      season             : int 0=spring, 1=summer, 2=fall, 3=winter
      is_weekend_arrival : int 0/1 (Saturday=5 or Sunday=6)
    """
    df = df.copy()
    df["booking_month"] = df["arrival_date"].dt.month.astype(int)
    df["season"] = df["booking_month"].map(_SEASON_MAP).astype(int)
    df["is_weekend_arrival"] = (df["arrival_date"].dt.dayofweek >= 5).astype(int)
    return df


def encode_categoricals(df: pd.DataFrame):
    """
    Prepare feature matrix and target vectors.

    Returns:
        X          : DataFrame -- all numeric, no leaky columns
        y_cancel   : Series   -- is_canceled
        y_noshow   : Series   -- is_no_show
    """
    df = df.copy()
    y_cancel = df["is_canceled"].copy()
    y_noshow = df["is_no_show"].copy()

    drop_cols = _LEAKY_COLS + ["is_canceled", "is_no_show", "revenue", "arrival_date"]
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    for col in _CATEGORICAL_COLS:
        if col in feature_df.columns:  # gracefully skips dropped leaky cols
            le = LabelEncoder()
            feature_df[col] = le.fit_transform(feature_df[col].astype(str))

    # Safety: drop any remaining non-numeric columns
    feature_df = feature_df.select_dtypes(exclude="object")

    return feature_df, y_cancel, y_noshow


def drop_leaky_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove features that are unavailable at booking time to prevent data leakage.
    Should be called before encode_categoricals() for the cancellation training path.
    Does NOT affect the forecasting pipeline (which uses daily-aggregated data).
    """
    cols_to_drop = [c for c in LEAKY_COLUMNS if c in df.columns]
    return df.drop(columns=cols_to_drop)
