"""Tests for src/overbooking.py"""
import pytest
import pandas as pd
import numpy as np
from datetime import date
from unittest.mock import MagicMock, patch


def _make_bookings(n: int, adr: float = 100.0) -> pd.DataFrame:
    """Create n non-cancelled bookings for 2024-01-15."""
    return pd.DataFrame({
        "arrival_date": pd.to_datetime(["2024-01-15"] * n),
        "is_canceled": [0] * n,
        "adr": [adr] * n,
        "lead_time": [30] * n,
        "adults": [2] * n,
    })


def _make_model(proba: float = 0.3):
    """Mock classifier that always returns the same cancel probability."""
    m = MagicMock()
    m.predict_proba = MagicMock(
        side_effect=lambda X: np.column_stack([
            np.full(len(X), 1 - proba),
            np.full(len(X), proba),
        ])
    )
    return m


def _patched_score(proba: float, n: int):
    """Return a DataFrame that score_bookings would produce."""
    return pd.DataFrame({
        "cancel_probability": [proba] * n,
        "risk_tier": ["High" if proba > 0.6 else ("Medium" if proba > 0.3 else "Low")] * n,
    })


@pytest.fixture(autouse=True)
def patch_score_bookings(monkeypatch):
    """Patch score_bookings so overbooking tests don't need a full ML stack."""
    def _fake_score(
        bookings_df,
        model,
        shap_explainer=None,
        feature_columns=None,
        label_encoders=None,
        feature_defaults=None,
    ):
        n = len(bookings_df)
        proba = float(model.predict_proba(np.zeros((n, 1)))[:, 1].mean())
        return _patched_score(proba, n)
    import src.overbooking as _mod
    monkeypatch.setattr(_mod, "score_bookings", _fake_score)


def test_empty_bookings_returns_zeros():
    from src.overbooking import calculate_overbooking_recommendation
    df = _make_bookings(0)
    rec = calculate_overbooking_recommendation(df, _make_model(), date(2024, 1, 15), 100)
    assert rec["recommended_overbooks"] == 0
    assert rec["expected_cancellations"] == 0.0
    assert rec["risk_level"] == "Low"


def test_high_cancel_proba_recommends_overbooks():
    from src.overbooking import calculate_overbooking_recommendation
    df = _make_bookings(50)
    rec = calculate_overbooking_recommendation(df, _make_model(0.8), date(2024, 1, 15), 100)
    assert rec["recommended_overbooks"] > 0


def test_zero_cancel_proba_no_overbooks():
    from src.overbooking import calculate_overbooking_recommendation
    df = _make_bookings(20)
    rec = calculate_overbooking_recommendation(df, _make_model(0.0), date(2024, 1, 15), 100)
    assert rec["recommended_overbooks"] == 0


def test_result_has_required_keys():
    from src.overbooking import calculate_overbooking_recommendation
    df = _make_bookings(10)
    rec = calculate_overbooking_recommendation(df, _make_model(), date(2024, 1, 15), 100)
    for key in ["expected_cancellations", "recommended_overbooks", "confidence_interval", "risk_level", "expected_revenue_impact"]:
        assert key in rec, f"Missing key: {key}"


def test_risk_level_is_valid():
    from src.overbooking import calculate_overbooking_recommendation
    df = _make_bookings(30)
    rec = calculate_overbooking_recommendation(df, _make_model(0.9), date(2024, 1, 15), 100)
    assert rec["risk_level"] in ("Low", "Medium", "High")


def test_confidence_interval_ordered():
    from src.overbooking import calculate_overbooking_recommendation
    df = _make_bookings(20)
    rec = calculate_overbooking_recommendation(df, _make_model(0.5), date(2024, 1, 15), 100)
    low, high = rec["confidence_interval"]
    assert low <= high


def test_no_matching_date_returns_zeros():
    from src.overbooking import calculate_overbooking_recommendation
    df = _make_bookings(10)
    # Use a date with no bookings
    rec = calculate_overbooking_recommendation(df, _make_model(), date(2025, 12, 31), 100)
    assert rec["recommended_overbooks"] == 0


def test_overbooking_passes_saved_feature_schema(monkeypatch):
    captured = {}

    def fake_score(bookings_df, model, shap_explainer=None, feature_columns=None, label_encoders=None, feature_defaults=None):
        captured["feature_columns"] = feature_columns
        captured["label_encoders"] = label_encoders
        captured["feature_defaults"] = feature_defaults
        return _patched_score(0.5, len(bookings_df))

    import src.overbooking as _mod
    monkeypatch.setattr(_mod, "score_bookings", fake_score)

    from src.overbooking import calculate_overbooking_recommendation
    df = _make_bookings(5)
    rec = calculate_overbooking_recommendation(
        df,
        _make_model(0.5),
        date(2024, 1, 15),
        100,
        feature_columns=["lead_time", "adr"],
        label_encoders={"market_segment": object()},
        feature_defaults={"lead_time": 0.0, "adr": 0.0},
    )

    assert rec["expected_cancellations"] > 0
    assert captured["feature_columns"] == ["lead_time", "adr"]
    assert "market_segment" in captured["label_encoders"]
    assert captured["feature_defaults"]["adr"] == 0.0
