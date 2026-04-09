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
