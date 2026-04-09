"""Tests for src/booking_risk_scorer.py"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch


def _make_bookings(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        "arrival_date": pd.to_datetime(["2024-01-15"] * n),
        "is_canceled": [0] * n,
        "adr": [100.0] * n,
        "lead_time": [30] * n,
        "adults": [2] * n,
        "children": [0] * n,
        "hotel": ["City Hotel"] * n,
        "meal": ["BB"] * n,
        "market_segment": ["Online TA"] * n,
        "customer_type": ["Transient"] * n,
        "is_repeated_guest": [0] * n,
        "previous_cancellations": [0] * n,
        "previous_bookings_not_canceled": [0] * n,
        "total_of_special_requests": [1] * n,
        "stays_in_weekend_nights": [1] * n,
        "stays_in_week_nights": [3] * n,
        "reserved_room_type": ["A"] * n,
        "required_car_parking_spaces": [0] * n,
    })


def _make_model(proba: float = 0.5):
    m = MagicMock()
    m.predict_proba = MagicMock(
        side_effect=lambda X: np.column_stack([
            np.full(len(X), 1 - proba),
            np.full(len(X), proba),
        ])
    )
    return m


@pytest.fixture(autouse=True)
def patch_preprocessing(monkeypatch):
    """Patch preprocessing so scorer tests don't need the full pipeline."""
    def fake_drop(df):
        return df

    def fake_encode(df):
        n = len(df)
        return pd.DataFrame(np.zeros((n, 5)), columns=["a", "b", "c", "d", "e"]), None, None

    import src.booking_risk_scorer as _mod
    monkeypatch.setattr(_mod, "drop_leaky_features", fake_drop)
    monkeypatch.setattr(_mod, "encode_categoricals", fake_encode)


def test_cancel_probability_column_added():
    from src.booking_risk_scorer import score_bookings
    result = score_bookings(_make_bookings(), _make_model(0.5))
    assert "cancel_probability" in result.columns
    assert result["cancel_probability"].between(0, 1).all()


def test_risk_tier_column_added():
    from src.booking_risk_scorer import score_bookings
    result = score_bookings(_make_bookings(), _make_model(0.5))
    assert "risk_tier" in result.columns
    assert set(result["risk_tier"].unique()).issubset({"Low", "Medium", "High"})


def test_low_probability_gives_low_tier():
    from src.booking_risk_scorer import score_bookings
    result = score_bookings(_make_bookings(), _make_model(0.1))
    assert (result["risk_tier"] == "Low").all()


def test_high_probability_gives_high_tier():
    from src.booking_risk_scorer import score_bookings
    result = score_bookings(_make_bookings(), _make_model(0.9))
    assert (result["risk_tier"] == "High").all()


def test_sorted_descending_by_probability():
    from src.booking_risk_scorer import score_bookings
    model = MagicMock()
    # Varying probabilities
    probas = np.linspace(0.1, 0.9, 10)
    model.predict_proba = MagicMock(
        side_effect=lambda X: np.column_stack([1 - probas, probas])
    )
    result = score_bookings(_make_bookings(10), model)
    vals = result["cancel_probability"].values
    assert list(vals) == sorted(vals, reverse=True)


def test_no_top_risk_factors_without_explainer():
    from src.booking_risk_scorer import score_bookings
    result = score_bookings(_make_bookings(), _make_model())
    assert "top_risk_factors" not in result.columns


def test_top_risk_factors_with_explainer():
    from src.booking_risk_scorer import score_bookings
    n = 5
    explainer = MagicMock()
    shap_vals = np.random.rand(n, 5)
    explainer.shap_values = MagicMock(return_value=shap_vals)
    result = score_bookings(_make_bookings(n), _make_model(), shap_explainer=explainer)
    assert "top_risk_factors" in result.columns
    # Each row should have 3 risk factors
    for factors in result["top_risk_factors"]:
        assert len(factors) == 3
        assert all("feature" in f and "impact" in f for f in factors)
