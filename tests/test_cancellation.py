import numpy as np
import pytest
from sklearn.datasets import make_classification
from src.cancellation import train_cancellation_models, predict_cancellation_proba


@pytest.fixture(scope="module")
def synth():
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    return X[:800], X[800:], y[:800], y[800:]


def test_train_returns_three_models(synth):
    X_tr, _, y_tr, _ = synth
    models = train_cancellation_models(X_tr, y_tr)
    assert set(models.keys()) == {"logistic", "random_forest", "xgboost"}


def test_predict_proba_shape(synth):
    X_tr, X_te, y_tr, _ = synth
    models = train_cancellation_models(X_tr, y_tr)
    for name, model in models.items():
        proba = predict_cancellation_proba(model, X_te)
        assert proba.shape == (len(X_te),), f"Wrong shape for {name}"


def test_predict_proba_in_range(synth):
    X_tr, X_te, y_tr, _ = synth
    models = train_cancellation_models(X_tr, y_tr)
    for name, model in models.items():
        proba = predict_cancellation_proba(model, X_te)
        assert ((proba >= 0) & (proba <= 1)).all(), f"Out of range for {name}"
