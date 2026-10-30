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
    booking_dow is computed in add_temporal_features() where arrival_date is a datetime.
    """
    df = df.copy()
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df = ensure_revenue_columns(df)
    if "reservation_status" not in df.columns:
        df["reservation_status"] = "Check-Out"
    df["is_no_show"] = (df["reservation_status"] == "No-Show").astype(int)

    # Booking urgency bucket: 0=short(<7 days), 1=medium(7–30), 2=long(>30)
    if "lead_time" in df.columns:
        df["lead_time_bucket"] = pd.cut(
            df["lead_time"],
            bins=[-1, 6, 30, float("inf")],
            labels=[0, 1, 2],
        ).astype(int)

    # ADR relative to this hotel's typical rate (price-positioning signal)
    if "adr" in df.columns:
        adr_mean = df["adr"].replace(0, np.nan).mean()
        if pd.notna(adr_mean) and adr_mean > 0:
            df["adr_vs_hotel_mean"] = (df["adr"] / adr_mean).clip(0, 10)
        else:
            df["adr_vs_hotel_mean"] = 1.0

    # ADR relative to segment+month average — captures price sensitivity more precisely.
    # A guest who paid 2× the typical July Online-TA rate has much higher cancel risk.
    # Uses arrival_date_month (raw column) since arrival_date is built in a later step.
    if "adr" in df.columns and "market_segment" in df.columns and "arrival_date_month" in df.columns:
        seg_month_mean = (
            df.groupby(["market_segment", "arrival_date_month"])["adr"]
            .transform(lambda x: x.replace(0, np.nan).mean())
        )
        df["adr_vs_segment_month_mean"] = (df["adr"] / seg_month_mean.clip(lower=1)).clip(0, 10).fillna(1.0)

    # Previous cancellation rate ratio — more informative than raw counts alone
    if "previous_cancellations" in df.columns and "previous_bookings_not_canceled" in df.columns:
        df["previous_cancellation_rate"] = (
            df["previous_cancellations"]
            / (df["previous_cancellations"] + df["previous_bookings_not_canceled"] + 1)
        )

    # Log-transform lead time — cancellation risk grows exponentially with lead time
    if "lead_time" in df.columns:
        df["log_lead_time"] = np.log1p(df["lead_time"])

    # Deposit-type binary flag — no-deposit bookings cancel at far higher rates
    if "deposit_type" in df.columns:
        df["deposit_is_no_deposit"] = (df["deposit_type"] == "No Deposit").astype(int)
        # High-risk interaction: no deposit AND long lead time
        if "lead_time" in df.columns:
            df["no_deposit_long_lead"] = df["deposit_is_no_deposit"] * (df["lead_time"] > 30).astype(int)

    # NOTE: hotel_cancel_rate_mean and market_segment_cancel_rate_mean are intentionally
    # NOT computed here. They are computed in services.py after the train/test split so
    # that test-set labels cannot leak into the training features.

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
    "realized_revenue",          # realized_revenue = revenue × (1 - is_canceled) → 0 for cancellations
    "assigned_room_type",        # assigned at check-in, not at booking time
    "booking_changes",           # terminal amendment count at check-in/cancellation, not a booking-time snapshot
    # days_in_waiting_list: fully known at booking confirmation — kept as a feature
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
      booking_dow        : int 0-6 (day of week when the booking was placed)
    """
    df = df.copy()
    df["booking_month"] = df["arrival_date"].dt.month.astype(int)
    df["season"] = df["booking_month"].map(_SEASON_MAP).astype(int)
    df["is_weekend_arrival"] = (df["arrival_date"].dt.dayofweek >= 5).astype(int)

    # Day of week the booking was placed (arrival_date minus lead_time days)
    booking_date = pd.to_datetime(df["arrival_date"]) - pd.to_timedelta(
        df["lead_time"].clip(lower=0), unit="D"
    )
    df["booking_dow"] = booking_date.dt.dayofweek

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

    # realized_revenue and booked_revenue leak the target at booking time:
    # realized_revenue = revenue * (1 - is_canceled) → 0 for every cancellation
    drop_cols = _LEAKY_COLS + [
        "is_canceled", "is_no_show", "revenue", "arrival_date",
        "realized_revenue", "booked_revenue",
    ]
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    for col in _CATEGORICAL_COLS:
        if col in feature_df.columns:  # gracefully skips dropped leaky cols
            le = LabelEncoder()
            feature_df[col] = le.fit_transform(feature_df[col].astype(str))

    # Safety: drop any remaining non-numeric columns
    feature_df = feature_df.select_dtypes(exclude="object")

    return feature_df, y_cancel, y_noshow


def target_encode_column(
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
    col: str,
    target_col: str = "is_canceled",
    global_mean: float | None = None,
) -> pd.Series:
    """
    Mean target rate per category, computed on train_df only.
    Unseen categories in full_df fall back to the global training mean.
    Use after the train/test split to prevent label leakage.
    """
    rates = train_df.groupby(col)[target_col].mean()
    if global_mean is None:
        global_mean = float(train_df[target_col].mean())
    return full_df[col].map(rates).fillna(global_mean).astype(float)


def expanding_target_encode(
    train_df: pd.DataFrame,
    col: str,
    target_col: str = "is_canceled",
    *,
    sort_col: str = "arrival_date",
    smoothing: float = 20.0,
    global_mean: float | None = None,
) -> pd.Series:
    """
    Past-only, smoothed target encoding for TRAINING rows.

    For each row (ordered by ``sort_col``), the encoding uses only the labels of
    earlier rows in the same category — never the row's own label — so it does
    not leak the target into its own feature. Each estimate is smoothed toward
    the global mean so rare/new categories don't get noisy extremes::

        enc_i = (sum_prior + k * global_mean) / (count_prior + k)

    A category's first occurrence falls back to ``global_mean`` (no prior). This
    reflects how cancellation behaviour per agent/country drifts over the
    2015–2017 span, and removes the self-label leak the static full-train mean
    carried. The static final map (``train_df.groupby(col).mean()``) is still
    used for TEST rows and at inference — see services.py.

    Returns a Series aligned to ``train_df.index``.
    """
    if global_mean is None:
        global_mean = float(train_df[target_col].mean())
    k = float(smoothing)

    work = train_df[[col, target_col]].copy()
    if sort_col in train_df.columns:
        order = train_df[sort_col].values
        work = work.assign(_order=order).sort_values("_order", kind="stable")
    grp = work.groupby(col)[target_col]
    # Exclusive per-category cumsum: subtract the row's own label so it never
    # leaks into its own prior (a global .shift(1) would cross categories).
    prior_sum = grp.cumsum() - work[target_col]
    prior_count = grp.cumcount()  # number of earlier rows in this category
    enc = (prior_sum + k * global_mean) / (prior_count + k)
    enc = enc.reindex(train_df.index)  # restore original row order
    return enc.fillna(global_mean).astype(float)


def compute_group_rates(
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute hotel and market_segment cancel rate means using only training rows,
    then join them onto full_df (covers both train and test splits).
    Returns full_df with hotel_cancel_rate_mean and market_segment_cancel_rate_mean columns added.
    """
    global_mean = float(train_df["is_canceled"].mean()) if "is_canceled" in train_df.columns else 0.0
    result = full_df.copy()
    if "hotel" in train_df.columns and "is_canceled" in train_df.columns:
        hotel_rates = train_df.groupby("hotel")["is_canceled"].mean()
        result["hotel_cancel_rate_mean"] = full_df["hotel"].map(hotel_rates).fillna(global_mean)
    if "market_segment" in train_df.columns and "is_canceled" in train_df.columns:
        seg_rates = train_df.groupby("market_segment")["is_canceled"].mean()
        result["market_segment_cancel_rate_mean"] = full_df["market_segment"].map(seg_rates).fillna(global_mean)
    return result


def drop_leaky_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove features that are unavailable at booking time to prevent data leakage.
    Should be called before encode_categoricals() for the cancellation training path.
    Does NOT affect the forecasting pipeline (which uses daily-aggregated data).
    """
    cols_to_drop = [c for c in LEAKY_COLUMNS if c in df.columns]
    return df.drop(columns=cols_to_drop)
