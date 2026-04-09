import numpy as np
import pytest
from src.evaluation import regression_metrics, classification_report_dict


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
