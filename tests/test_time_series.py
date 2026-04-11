import pandas as pd
import pytest
from src.preprocessing import load_data, clean_data, engineer_features
from src.time_series import build_arrival_date, aggregate_daily

@pytest.fixture(scope="module")
def df():
    raw = load_data("hotel_bookings.csv")
    cleaned = clean_data(raw)
    return engineer_features(cleaned)


def test_arrival_date_is_datetime(df):
    result = build_arrival_date(df)
    assert pd.api.types.is_datetime64_any_dtype(result["arrival_date"])


def test_arrival_date_no_nulls(df):
    result = build_arrival_date(df)
    assert result["arrival_date"].isna().sum() == 0


def test_aggregate_has_required_columns(df):
    dated = build_arrival_date(df)
    daily = aggregate_daily(dated)
    assert "daily_revenue" in daily.columns
    assert "daily_booked_revenue" in daily.columns
    assert "daily_realized_revenue" in daily.columns
    assert "daily_expected_realized_revenue" in daily.columns
    assert "daily_bookings" in daily.columns
    assert "daily_cancellations" in daily.columns


def test_aggregate_index_is_datetime(df):
    dated = build_arrival_date(df)
    daily = aggregate_daily(dated)
    assert pd.api.types.is_datetime64_any_dtype(daily.index)


def test_aggregate_index_is_sorted(df):
    dated = build_arrival_date(df)
    daily = aggregate_daily(dated)
    assert daily.index.is_monotonic_increasing


def test_aggregate_no_date_gaps(df):
    dated = build_arrival_date(df)
    daily = aggregate_daily(dated)
    expected_len = (daily.index.max() - daily.index.min()).days + 1
    assert len(daily) == expected_len


def test_daily_revenue_non_negative(df):
    dated = build_arrival_date(df)
    daily = aggregate_daily(dated)
    assert (daily["daily_revenue"] >= 0).all()


def test_booked_revenue_alias_matches_daily_revenue(df):
    dated = build_arrival_date(df)
    daily = aggregate_daily(dated)
    assert daily["daily_revenue"].equals(daily["daily_booked_revenue"])


def test_expected_realized_defaults_to_realized_when_missing(df):
    dated = build_arrival_date(df)
    daily = aggregate_daily(dated)
    assert daily["daily_expected_realized_revenue"].equals(daily["daily_realized_revenue"])


def test_aggregate_daily_recovers_missing_split_revenue_columns():
    df = pd.DataFrame(
        {
            "arrival_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "revenue": [100.0, 120.0],
            "is_canceled": [0, 1],
            "stays_in_weekend_nights": [1, 1],
            "stays_in_week_nights": [1, 1],
            "adr": [50.0, 60.0],
        }
    )

    daily = aggregate_daily(df)

    assert daily.loc[pd.Timestamp("2024-01-01"), "daily_booked_revenue"] == 100.0
    assert daily.loc[pd.Timestamp("2024-01-01"), "daily_realized_revenue"] == 100.0
    assert daily.loc[pd.Timestamp("2024-01-02"), "daily_realized_revenue"] == 0.0


def test_duplicate_rows_do_not_inflate_daily_metrics():
    df = pd.DataFrame(
        {
            "arrival_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
            "is_canceled": [0, 0, 1],
            "adr": [50.0, 50.0, 60.0],
            "stays_in_weekend_nights": [1, 1, 1],
            "stays_in_week_nights": [1, 1, 1],
            "agent": [0, 0, 0],
            "company": [0, 0, 0],
            "country": ["PRT", "PRT", "GBR"],
            "children": [0, 0, 0],
            "babies": [0, 0, 0],
            "adults": [2, 2, 2],
        }
    )

    from src.preprocessing import clean_data, engineer_features

    cleaned = clean_data(df)
    engineered = engineer_features(cleaned)
    daily = aggregate_daily(engineered)

    assert len(cleaned) == 2
    assert daily.loc[pd.Timestamp("2024-01-01"), "daily_bookings"] == 1
    assert daily.loc[pd.Timestamp("2024-01-01"), "daily_booked_revenue"] == 100.0
