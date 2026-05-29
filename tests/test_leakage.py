"""
Tests verifying that leaky features are excluded from cancellation model training.
Also tests cross-dataset loading and evaluation.
"""
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    load_data, clean_data, engineer_features,
    drop_leaky_features, encode_categoricals, LEAKY_COLUMNS,
)
from src.ingestion import load_cancellation_prediction_csv
from src.cancellation_evaluation import evaluate_cross_dataset


# ---------------------------------------------------------------------------
# Feature leakage checks
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engineered_df():
    df = load_data("hotel_bookings.csv")
    df = clean_data(df)
    df = engineer_features(df)
    return df


def test_leaky_cols_absent_after_drop_and_encode(engineered_df):
    X, _, _ = encode_categoricals(drop_leaky_features(engineered_df))
    for col in LEAKY_COLUMNS:
        assert col not in X.columns, f"Leaky column '{col}' present in feature matrix"


def test_reservation_status_not_in_X(engineered_df):
    X, _, _ = encode_categoricals(drop_leaky_features(engineered_df))
    assert "reservation_status" not in X.columns


def test_deposit_type_in_X(engineered_df):
    """deposit_type is set at booking time — must be a feature, not dropped."""
    X, _, _ = encode_categoricals(drop_leaky_features(engineered_df))
    assert "deposit_type" in X.columns


def test_realized_revenue_not_in_X(engineered_df):
    """realized_revenue encodes is_canceled directly — must never reach feature matrix."""
    X, _, _ = encode_categoricals(drop_leaky_features(engineered_df))
    assert "realized_revenue" not in X.columns


def test_assigned_room_type_not_in_X(engineered_df):
    X, _, _ = encode_categoricals(drop_leaky_features(engineered_df))
    assert "assigned_room_type" not in X.columns


def test_booking_changes_not_in_X(engineered_df):
    # booking_changes is the terminal amendment count at check-in/cancellation,
    # not a snapshot available at any earlier prediction time — using it leaks
    # post-decision information into training.
    X, _, _ = encode_categoricals(drop_leaky_features(engineered_df))
    assert "booking_changes" not in X.columns


# ---------------------------------------------------------------------------
# Second dataset loading
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def second_df():
    return load_cancellation_prediction_csv("cancellation_Prediction.csv")


def test_second_df_is_canceled_binary(second_df):
    assert second_df["is_canceled"].isin([0, 1]).all()


def test_second_df_has_no_booking_id(second_df):
    assert "Booking_ID" not in second_df.columns


def test_second_df_arrival_date_is_datetime(second_df):
    assert pd.api.types.is_datetime64_any_dtype(second_df["arrival_date"])


def test_second_df_has_expected_cols(second_df):
    for col in ["lead_time", "adr", "adults", "is_canceled"]:
        assert col in second_df.columns, f"Expected column '{col}' missing"


# ---------------------------------------------------------------------------
# Cross-dataset evaluation
# ---------------------------------------------------------------------------

def test_evaluate_cross_dataset_returns_expected_keys(second_df):
    """Smoke test with tiny synthetic models."""
    from sklearn.dummy import DummyClassifier
    import numpy as np

    feature_columns = ["lead_time", "adr", "adults"]
    label_encoders = {}
    feature_defaults = {"lead_time": 30.0, "adr": 100.0, "adults": 2.0}

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((10, 3)), np.array([0, 1] * 5))
    models = {"dummy": dummy}

    results = evaluate_cross_dataset(
        models, feature_columns, label_encoders, feature_defaults, second_df
    )
    assert "dummy" in results
    assert "roc_auc" in results["dummy"]
    assert "accuracy" in results["dummy"]
