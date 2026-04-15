import logging

import pandas as pd
import pytest
from src.preprocessing import load_data, clean_data, engineer_features
from src.time_series import (
    aggregate_daily,
    aggregate_daily_by_hotel_type,
    build_arrival_date,
    expand_stay_nights,
    aggregate_stay_nights,
)

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


# ---------------------------------------------------------------------------
# aggregate_daily_by_hotel_type tests
# ---------------------------------------------------------------------------

def _make_hotel_df(n_resort=40, n_city=40):
    """Synthetic booking-level DataFrame with two hotel types."""
    resort_dates = pd.date_range("2022-01-01", periods=n_resort, freq="D")
    city_dates = pd.date_range("2022-01-01", periods=n_city, freq="D")
    resort = pd.DataFrame({
        "arrival_date": resort_dates,
        "hotel": "Resort Hotel",
        "is_canceled": 0,
        "adr": 120.0,
        "stays_in_weekend_nights": 1,
        "stays_in_week_nights": 1,
    })
    city = pd.DataFrame({
        "arrival_date": city_dates,
        "hotel": "City Hotel",
        "is_canceled": 0,
        "adr": 90.0,
        "stays_in_weekend_nights": 1,
        "stays_in_week_nights": 1,
    })
    return pd.concat([resort, city], ignore_index=True)


def test_aggregate_daily_by_hotel_type_returns_both_types():
    df = _make_hotel_df()
    result = aggregate_daily_by_hotel_type(df)
    assert "Resort Hotel" in result
    assert "City Hotel" in result


def test_aggregate_daily_by_hotel_type_schema_matches():
    df = _make_hotel_df()
    combined = aggregate_daily(df)
    result = aggregate_daily_by_hotel_type(df)
    for daily in result.values():
        assert set(daily.columns) == set(combined.columns)


def test_aggregate_daily_by_hotel_type_skips_insufficient_history(caplog):
    df = _make_hotel_df(n_resort=5, n_city=40)  # Resort has only 5 days — below default 31
    with caplog.at_level(logging.WARNING, logger="root"):
        result = aggregate_daily_by_hotel_type(df)
    assert "Resort Hotel" not in result
    assert "City Hotel" in result
    assert any("Resort Hotel" in m for m in caplog.messages)


def test_aggregate_daily_by_hotel_type_no_hotel_column(caplog):
    df = pd.DataFrame({
        "arrival_date": pd.date_range("2022-01-01", periods=10),
        "is_canceled": 0,
        "adr": 100.0,
        "stays_in_weekend_nights": 1,
        "stays_in_week_nights": 1,
    })
    with caplog.at_level(logging.WARNING, logger="root"):
        result = aggregate_daily_by_hotel_type(df)
    assert result == {}


def test_aggregate_daily_by_hotel_type_no_gaps():
    """Each per-type daily DataFrame must have no date gaps."""
    df = _make_hotel_df()
    result = aggregate_daily_by_hotel_type(df)
    for daily in result.values():
        expected_len = (daily.index.max() - daily.index.min()).days + 1
        assert len(daily) == expected_len


# ---------------------------------------------------------------------------
# Stay-night expansion tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_bookings_stay():
    return pd.DataFrame({
        "arrival_date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-01"]),
        "total_nights": [2, 3, 1],
        "adr": [100.0, 200.0, 50.0],
        "is_canceled": [0, 0, 1],  # third booking cancelled — excluded
    })


def test_expand_stay_nights_excludes_cancelled(sample_bookings_stay):
    stay = expand_stay_nights(sample_bookings_stay)
    assert len(stay) == 5  # 2 + 3 nights only


def test_expand_stay_nights_columns(sample_bookings_stay):
    stay = expand_stay_nights(sample_bookings_stay)
    assert set(stay.columns) >= {"date", "revenue_per_night"}


def test_expand_stay_nights_dates(sample_bookings_stay):
    stay = expand_stay_nights(sample_bookings_stay)
    dates = sorted(stay["date"].dt.date.tolist())
    expected = sorted([
        pd.Timestamp("2023-01-01").date(),
        pd.Timestamp("2023-01-02").date(),
        pd.Timestamp("2023-01-02").date(),
        pd.Timestamp("2023-01-03").date(),
        pd.Timestamp("2023-01-04").date(),
    ])
    assert dates == expected


def test_aggregate_stay_nights_columns(sample_bookings_stay):
    stay = expand_stay_nights(sample_bookings_stay)
    occ = aggregate_stay_nights(stay)
    assert "occupied_rooms" in occ.columns
    assert "daily_stay_revenue" in occ.columns


def test_aggregate_stay_nights_no_gaps(sample_bookings_stay):
    stay = expand_stay_nights(sample_bookings_stay)
    occ = aggregate_stay_nights(stay)
    expected_range = pd.date_range(occ.index.min(), occ.index.max(), freq="D")
    assert len(occ) == len(expected_range)


def test_aggregate_stay_nights_occupied_count(sample_bookings_stay):
    stay = expand_stay_nights(sample_bookings_stay)
    occ = aggregate_stay_nights(stay)
    # 2023-01-02: both non-cancelled bookings overlap → 2 rooms occupied
    assert occ.loc[pd.Timestamp("2023-01-02"), "occupied_rooms"] == 2
