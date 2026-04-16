"""
Tests for src/finetuning.py and versioned model_cache.py.
"""
import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from src.model_cache import save_artifact, load_artifact, load_artifact_with_metadata
from src.finetuning import finetune_cancellation_model, finetune_forecast_model, evaluate_before_deploy


# ---------------------------------------------------------------------------
# model_cache versioning
# ---------------------------------------------------------------------------

def test_save_and_load_versioned():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        save_artifact("eval_metrics", {"mae": 42.0}, metadata={"n_samples": 1000}, models_dir=d)
        obj = load_artifact("eval_metrics", models_dir=d)
        assert obj == {"mae": 42.0}


def test_load_with_metadata_returns_tuple():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        save_artifact("eval_metrics", {"mae": 5.0}, metadata={"source": "test"}, models_dir=d)
        obj, meta = load_artifact_with_metadata("eval_metrics", models_dir=d)
        assert obj == {"mae": 5.0}
        assert meta.get("source") == "test"


def test_load_with_metadata_legacy_format():
    """Load a raw (legacy) artifact — metadata should be {}."""
    import joblib
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        (d).mkdir(parents=True, exist_ok=True)
        # Write raw object (no wrapper)
        joblib.dump({"mae": 99.0}, d / "eval_metrics.pkl")
        obj, meta = load_artifact_with_metadata("eval_metrics", models_dir=d)
        assert obj == {"mae": 99.0}
        assert meta == {}


def test_load_missing_returns_none():
    with tempfile.TemporaryDirectory() as tmpdir:
        obj = load_artifact("eval_metrics", models_dir=Path(tmpdir))
        assert obj is None


# ---------------------------------------------------------------------------
# finetune_cancellation_model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_hotel_df():
    """Minimal hotel-like DataFrame for testing finetuning."""
    rng = np.random.default_rng(0)
    n = 400
    return pd.DataFrame({
        "is_canceled":          rng.integers(0, 2, n),
        "is_no_show":           rng.integers(0, 2, n),
        "lead_time":            rng.integers(0, 300, n),
        "adr":                  rng.uniform(50, 300, n),
        "total_nights":         rng.integers(1, 10, n),
        "total_guests":         rng.integers(1, 5, n),
        "adults":               rng.integers(1, 4, n),
        "children":             rng.integers(0, 2, n),
        "babies":               rng.integers(0, 1, n),
        "stays_in_weekend_nights": rng.integers(0, 4, n),
        "stays_in_week_nights": rng.integers(0, 6, n),
        "hotel":                rng.choice(["City Hotel", "Resort Hotel"], n),
        "meal":                 rng.choice(["BB", "HB", "SC"], n),
        "country":              rng.choice(["PRT", "GBR", "FRA"], n),
        "market_segment":       rng.choice(["Direct", "Online TA", "Corporate"], n),
        "distribution_channel": rng.choice(["Direct", "TA/TO"], n),
        "reserved_room_type":   rng.choice(["A", "B", "C"], n),
        "customer_type":        rng.choice(["Transient", "Group"], n),
        "is_repeated_guest":    rng.integers(0, 2, n),
        "previous_cancellations": rng.integers(0, 3, n),
        "previous_bookings_not_canceled": rng.integers(0, 5, n),
        "total_of_special_requests": rng.integers(0, 4, n),
        "booking_month":        rng.integers(1, 13, n),
        "season":               rng.integers(0, 4, n),
        "is_weekend_arrival":   rng.integers(0, 2, n),
        "required_car_parking_spaces": rng.integers(0, 2, n),
        "agent":                rng.integers(0, 10, n),
        "company":              rng.integers(0, 5, n),
    })


def test_finetune_returns_three_models(small_hotel_df):
    models, metrics = finetune_cancellation_model(small_hotel_df)
    assert {"logistic", "random_forest", "xgboost"}.issubset(set(models.keys()))


def test_finetune_metrics_have_roc_auc(small_hotel_df):
    _, metrics = finetune_cancellation_model(small_hotel_df)
    for name, m in metrics.items():
        assert "roc_auc" in m, f"roc_auc missing for {name}"


def test_evaluate_before_deploy_returns_dict(small_hotel_df):
    models, _ = finetune_cancellation_model(small_hotel_df)
    result = evaluate_before_deploy(models["xgboost"], small_hotel_df)
    assert "roc_auc" in result
    assert "meets_threshold" in result


# ---------------------------------------------------------------------------
# finetune_forecast_model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def revenue_series():
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    rng = np.random.default_rng(1)
    vals = 1000 + rng.normal(0, 100, 200).cumsum()
    return pd.Series(np.abs(vals), index=dates)


def test_finetune_forecast_xgboost(revenue_series):
    model, metrics = finetune_forecast_model(revenue_series, model_type="xgboost")
    assert isinstance(model, tuple)  # (model, feat_cols)
    assert "mae" in metrics


def test_finetune_forecast_prophet(revenue_series):
    model, metrics = finetune_forecast_model(revenue_series, model_type="prophet")
    from prophet import Prophet
    assert isinstance(model, Prophet)
    assert "mae" in metrics
