import numpy as np
import pytest
from src.evaluation import (
    calibration_metrics,
    classification_report_dict,
    empirical_interval_coverage,
    regression_metrics,
)


def test_regression_perfect_score():
    y = np.array([1.0, 2.0, 3.0])
    result = regression_metrics(y, y)
    assert result["mae"] == pytest.approx(0.0)
    assert result["rmse"] == pytest.approx(0.0)


def test_regression_metric_keys():
    y = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 2.9])
    result = regression_metrics(y, y_pred)
    assert {"mae", "rmse", "mape"} == set(result.keys())


def test_regression_mape_zero_safe():
    # Should not raise ZeroDivisionError when y_true contains zeros
    y = np.array([0.0, 2.0, 4.0])
    y_pred = np.array([0.1, 2.1, 3.9])
    result = regression_metrics(y, y_pred)
    assert result["mape"] >= 0


def test_classification_report_keys():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.4, 0.2, 0.8])
    result = classification_report_dict(y_true, y_pred, y_prob)
    assert {"accuracy", "precision", "recall", "f1", "roc_auc"} == set(result.keys())


def test_classification_perfect():
    y = np.array([0, 1, 0, 1])
    proba = np.array([0.0, 1.0, 0.0, 1.0])
    result = classification_report_dict(y, y, proba)
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["roc_auc"] == pytest.approx(1.0)


def test_calibration_metrics_perfect():
    """When y_prob equals y_true exactly, Brier and ECE both collapse to zero."""
    y = np.array([0, 1, 0, 1, 1, 0])
    p = y.astype(float)
    result = calibration_metrics(y, p, n_bins=5)
    assert result["brier"] == pytest.approx(0.0)
    assert result["ece"] == pytest.approx(0.0)


def test_calibration_metrics_constant_half():
    """All probs = 0.5 with a balanced y: Brier = 0.25, ECE = 0."""
    y = np.array([0, 0, 1, 1])
    p = np.full(4, 0.5)
    result = calibration_metrics(y, p, n_bins=10)
    assert result["brier"] == pytest.approx(0.25)
    assert result["ece"] == pytest.approx(0.0, abs=1e-9)


def test_calibration_metrics_curve_shape():
    rng = np.random.RandomState(0)
    n = 1000
    y = rng.binomial(1, 0.3, size=n)
    p = rng.uniform(0, 1, size=n)
    result = calibration_metrics(y, p, n_bins=10)
    assert "curve" in result
    assert len(result["curve"]) == 10
    # Counts should sum to n
    assert int(result["curve"]["count"].sum()) == n


def test_empirical_interval_coverage_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    lo = y - 0.5
    hi = y + 0.5
    result = empirical_interval_coverage(y, lo, hi)
    assert result["coverage"] == pytest.approx(1.0)
    assert result["mean_width"] == pytest.approx(1.0)
    assert result["n"] == 4


def test_empirical_interval_coverage_none():
    y = np.array([1.0, 2.0])
    lo = np.array([10.0, 20.0])
    hi = np.array([11.0, 21.0])
    result = empirical_interval_coverage(y, lo, hi)
    assert result["coverage"] == pytest.approx(0.0)


def test_empirical_interval_coverage_partial():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    lo = np.array([0.0, 0.0, 5.0, 5.0])
    hi = np.array([5.0, 5.0, 10.0, 10.0])
    result = empirical_interval_coverage(y, lo, hi)
    assert result["coverage"] == pytest.approx(0.5)
