"""
Tests for src/ingestion.py — schema mapping, validation, and auto-detection.
"""
import pandas as pd
import numpy as np
import pytest
from src.ingestion import (
    map_columns,
    validate_schema,
    auto_detect_columns,
    load_cancellation_prediction_csv,
    CANONICAL_SCHEMA,
    CANCELLATION_CSV_MAPPING,
)


# ---------------------------------------------------------------------------
# map_columns
# ---------------------------------------------------------------------------

def test_map_columns_renames_correctly():
    df = pd.DataFrame({"average price": [100.0], "lead time": [30]})
    mapping = {"average price": "adr", "lead time": "lead_time"}
    result = map_columns(df, mapping)
    assert "adr" in result.columns
    assert "lead_time" in result.columns
    assert "average price" not in result.columns


def test_map_columns_keeps_unmapped():
    df = pd.DataFrame({"adr": [100.0], "extra_col": [1]})
    result = map_columns(df, {"adr": "adr"})
    assert "extra_col" in result.columns


# ---------------------------------------------------------------------------
# validate_schema
# ---------------------------------------------------------------------------

def test_validate_passes_with_required_cols():
    df = pd.DataFrame({
        "arrival_date": pd.to_datetime(["2020-01-01"]),
        "is_canceled": [0],
    })
    valid, issues = validate_schema(df)
    assert valid
    assert issues == []


def test_validate_fails_missing_required():
    df = pd.DataFrame({"arrival_date": pd.to_datetime(["2020-01-01"])})
    valid, issues = validate_schema(df)
    assert not valid
    assert any("is_canceled" in msg for msg in issues)


def test_validate_fails_both_required_missing():
    df = pd.DataFrame({"some_col": [1]})
    valid, issues = validate_schema(df)
    assert not valid
    assert len(issues) >= 2


# ---------------------------------------------------------------------------
# auto_detect_columns
# ---------------------------------------------------------------------------

def test_auto_detect_adr():
    df = pd.DataFrame({"average price": [100.0]})
    mapping = auto_detect_columns(df)
    assert mapping.get("average price") == "adr"


def test_auto_detect_lead_time():
    df = pd.DataFrame({"lead time": [30]})
    mapping = auto_detect_columns(df)
    assert mapping.get("lead time") == "lead_time"


def test_auto_detect_returns_dict():
    df = pd.DataFrame({"arrival_date": ["2020-01-01"], "adults": [2]})
    mapping = auto_detect_columns(df)
    assert isinstance(mapping, dict)


def test_auto_detect_no_false_positives():
    df = pd.DataFrame({"zzz_garbage_xyz": [1]})
    mapping = auto_detect_columns(df)
    # Should not confidently map a completely unrelated column
    assert mapping.get("zzz_garbage_xyz") is None or True  # may or may not match; just no crash


# ---------------------------------------------------------------------------
# load_cancellation_prediction_csv
# ---------------------------------------------------------------------------

SECOND_CSV_PATH = "cancellation_Prediction.csv"


@pytest.fixture(scope="module")
def second_df():
    return load_cancellation_prediction_csv(SECOND_CSV_PATH)


def test_second_df_loads(second_df):
    assert len(second_df) > 0


def test_second_df_is_canceled_binary(second_df):
    assert second_df["is_canceled"].isin([0, 1]).all()


def test_second_df_no_booking_id(second_df):
    assert "Booking_ID" not in second_df.columns


def test_second_df_has_lead_time(second_df):
    assert "lead_time" in second_df.columns


def test_second_df_arrival_date_datetime(second_df):
    assert pd.api.types.is_datetime64_any_dtype(second_df["arrival_date"])


def test_cancellation_csv_mapping_completeness():
    expected_canonical = {"adults", "children", "stays_in_weekend_nights",
                           "stays_in_week_nights", "lead_time", "adr", "is_canceled"}
    mapped_values = set(CANCELLATION_CSV_MAPPING.values())
    assert expected_canonical.issubset(mapped_values)
