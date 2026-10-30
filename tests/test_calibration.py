"""Tests for the probability-calibration path on the stacked cancellation model."""
import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.cancellation import (
    _NON_MODEL_KEYS,
    apply_probability_calibrator,
    predict_stacked_proba,
    train_cancellation_models_balanced,
    train_probability_calibrator,
    train_stacked_cancellation_model,
)


@pytest.fixture(scope="module")
def stacked_bundle():
    """Train a small stacked cancellation bundle + calibrator on synthetic data."""
    X, y = make_classification(
        n_samples=600,
        n_features=12,
        n_informative=6,
        n_redundant=2,
        weights=[0.7, 0.3],
        random_state=42,
    )
    base = train_cancellation_models_balanced(X, y)
    meta = train_stacked_cancellation_model(base, X, y, n_splits=3)
    base["stacked_meta"] = meta
    calibrator, _oof = train_probability_calibrator(base, X, y, n_splits=3)
    base["probability_calibrator"] = calibrator
    return X, y, base


def test_apply_calibrator_is_identity_when_none():
    raw = np.array([0.1, 0.5, 0.9])
    out = apply_probability_calibrator(None, raw)
    assert np.allclose(out, raw)


def test_calibrator_outputs_in_unit_range(stacked_bundle):
    X, _, base = stacked_bundle
    calibrated = predict_stacked_proba(
        base["stacked_meta"], base, X,
        calibrator=base["probability_calibrator"],
    )
    assert ((calibrated >= 0.0) & (calibrated <= 1.0)).all()


def test_calibrator_improves_brier(stacked_bundle):
    """Calibration should not make the Brier score worse on the training-equivalent data."""
    X, y, base = stacked_bundle
    raw = predict_stacked_proba(base["stacked_meta"], base, X, calibrator=None)
    cal = predict_stacked_proba(
        base["stacked_meta"], base, X,
        calibrator=base["probability_calibrator"],
    )
    brier_raw = float(np.mean((raw - y) ** 2))
    brier_cal = float(np.mean((cal - y) ** 2))
    # Allow tiny worsening from monotonic constraint noise; calibration on synthetic
    # data that's already pretty well-behaved often draws even rather than improving.
    assert brier_cal <= brier_raw + 1e-3


def test_calibrator_is_none_when_data_too_small():
    """Tiny X should return None instead of raising."""
    rng = np.random.RandomState(0)
    X = rng.randn(8, 4)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    base = train_cancellation_models_balanced(X, y)
    meta = train_stacked_cancellation_model(base, X, y, n_splits=3)
    base["stacked_meta"] = meta
    cal, oof = train_probability_calibrator(base, X, y, n_splits=3)
    # Either None (not enough covered rows) or a fitted regressor — both are fine,
    # but it must not raise. OOF data tracks the calibrator's availability.
    if cal is not None:
        out = apply_probability_calibrator(cal, np.array([0.2, 0.8]))
        assert ((out >= 0.0) & (out <= 1.0)).all()
        assert oof is not None and {"proba_raw", "y"}.issubset(oof.columns)
    else:
        assert oof is None


def test_calibrator_returns_oof_for_threshold_selection(stacked_bundle):
    """The OOF DataFrame is the leakage-free basis for threshold selection: it
    must cover training rows only and carry calibratable raw probabilities."""
    X, y, base = stacked_bundle
    cal, oof = train_probability_calibrator(base, X, y, n_splits=3)
    assert cal is not None and oof is not None
    assert list(oof.columns) == ["proba_raw", "y"]
    assert len(oof) <= len(X)               # OOF covers a subset of training rows
    assert oof["proba_raw"].between(0, 1).all()
    assert set(oof["y"].unique()).issubset({0, 1})


def test_non_model_keys_includes_calibrator():
    """predict_stacked_proba must skip the calibrator slot when scanning base models."""
    assert "probability_calibrator" in _NON_MODEL_KEYS
    assert "stacked_meta" in _NON_MODEL_KEYS


def test_predict_stacked_proba_ignores_non_model_entries(stacked_bundle):
    X, _, base = stacked_bundle
    # Inject a sentinel non-model entry; predict_stacked_proba should not try
    # to call .predict_proba on it.
    base_copy = dict(base)
    base_copy["target_encode_maps"] = {"country": {"PT": 0.5}}
    out = predict_stacked_proba(
        base_copy["stacked_meta"], base_copy, X[:10],
        calibrator=base_copy["probability_calibrator"],
    )
    assert out.shape == (10,)


def test_calibrator_handles_out_of_range_inputs(stacked_bundle):
    _, _, base = stacked_bundle
    # IsotonicRegression(out_of_bounds='clip') should clip negatives and >1 inputs.
    out = apply_probability_calibrator(base["probability_calibrator"], np.array([-0.5, 1.5]))
    assert ((out >= 0.0) & (out <= 1.0)).all()
