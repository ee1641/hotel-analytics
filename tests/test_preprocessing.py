import pandas as pd
import numpy as np
import pytest
from src.preprocessing import (
    load_data, clean_data, engineer_features, encode_categoricals,
    drop_leaky_features, LEAKY_COLUMNS, remove_accidental_duplicate_bookings,
)

DATA_PATH = "hotel_bookings.csv"

@pytest.fixture(scope="module")
def raw_df():
    return load_data(DATA_PATH)

@pytest.fixture(scope="module")
def clean_df(raw_df):
    return clean_data(raw_df)

@pytest.fixture(scope="module")
def engineered_df(clean_df):
    return engineer_features(clean_df)


def test_load_shape(raw_df):
    assert raw_df.shape[0] > 100_000
    assert raw_df.shape[1] == 32


def test_no_negative_adr(clean_df):
    assert (clean_df["adr"] >= 0).all()


def test_no_null_country(clean_df):
    assert clean_df["country"].isna().sum() == 0


def test_no_null_agent(clean_df):
    assert clean_df["agent"].isna().sum() == 0


def test_no_null_children(clean_df):
    assert clean_df["children"].isna().sum() == 0


def test_exact_duplicate_rows_removed():
    df = pd.DataFrame(
        [
            {"arrival_date": "2024-01-01", "is_canceled": 0, "adr": 100.0},
            {"arrival_date": "2024-01-01", "is_canceled": 0, "adr": 100.0},
            {"arrival_date": "2024-01-02", "is_canceled": 1, "adr": 120.0},
        ]
    )

    deduped = remove_accidental_duplicate_bookings(df)

    assert len(deduped) == 2


def test_clean_data_removes_exact_duplicates():
    df = pd.DataFrame(
        [
            {"arrival_date": "2024-01-01", "is_canceled": 0, "adr": 100.0, "agent": None, "company": None, "country": None, "children": None},
            {"arrival_date": "2024-01-01", "is_canceled": 0, "adr": 100.0, "agent": None, "company": None, "country": None, "children": None},
            {"arrival_date": "2024-01-02", "is_canceled": 1, "adr": 120.0, "agent": None, "company": None, "country": None, "children": None},
        ]
    )

    cleaned = clean_data(df)

    assert len(cleaned) == 2


def test_total_nights(engineered_df):
    expected = engineered_df["stays_in_weekend_nights"] + engineered_df["stays_in_week_nights"]
    pd.testing.assert_series_equal(engineered_df["total_nights"], expected, check_names=False)


def test_total_guests(engineered_df):
    expected = engineered_df["adults"] + engineered_df["children"] + engineered_df["babies"]
    pd.testing.assert_series_equal(engineered_df["total_guests"], expected, check_names=False)


def test_revenue_non_negative(engineered_df):
    assert (engineered_df["revenue"] >= 0).all()


def test_is_no_show(engineered_df):
    expected = (engineered_df["reservation_status"] == "No-Show").astype(int)
    pd.testing.assert_series_equal(engineered_df["is_no_show"], expected, check_names=False)


def test_encode_drops_leaky_columns(engineered_df):
    X, _, _ = encode_categoricals(engineered_df)
    assert "reservation_status" not in X.columns
    assert "reservation_status_date" not in X.columns


def test_encode_no_object_columns(engineered_df):
    X, _, _ = encode_categoricals(engineered_df)
    assert X.select_dtypes(include="object").empty


def test_encode_retains_lead_time(engineered_df):
    X, _, _ = encode_categoricals(engineered_df)
    assert "lead_time" in X.columns


def test_drop_leaky_removes_all_leaky_columns(engineered_df):
    clean = drop_leaky_features(engineered_df)
    for col in LEAKY_COLUMNS:
        assert col not in clean.columns, f"Leaky column '{col}' still present after drop"


def test_drop_leaky_preserves_non_leaky_columns(engineered_df):
    clean = drop_leaky_features(engineered_df)
    assert "lead_time" in clean.columns
    assert "adr" in clean.columns
    assert "total_nights" in clean.columns


def test_drop_leaky_then_encode_has_no_leaky_cols(engineered_df):
    X, _, _ = encode_categoricals(drop_leaky_features(engineered_df))
    for col in LEAKY_COLUMNS:
        assert col not in X.columns, f"Leaky column '{col}' leaked into feature matrix"
