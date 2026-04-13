import numpy as np
import pytest
from sklearn.datasets import make_classification
import pandas as pd
from src.cancellation import (
    predict_booking_survival_profile,
    predict_cancellation_proba,
    train_cancellation_models,
    train_cancellation_survival_model,
)


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


def test_survival_profile_is_monotonic():
    df = pd.DataFrame(
        {
            "arrival_date": pd.to_datetime(["2024-02-10", "2024-02-10", "2024-02-15", "2024-02-20"]),
            "lead_time": [20, 10, 15, 30],
            "is_canceled": [1, 0, 1, 0],
            "reservation_status_date": pd.to_datetime(["2024-02-05", "2024-02-11", "2024-02-10", "2024-02-20"]),
            "adr": [100.0, 120.0, 95.0, 140.0],
            "total_nights": [2, 3, 1, 4],
            "deposit_type": ["No Deposit", "No Deposit", "Non Refund", "Refundable"],
            "market_segment": ["Online TA", "Direct", "Online TA", "Corporate"],
        }
    )
    bundle = train_cancellation_survival_model(df)
    profile = predict_booking_survival_profile(
        bundle,
        {
            "lead_time": 20,
            "adr": 110.0,
            "total_nights": 2,
            "deposit_type": "No Deposit",
            "market_segment": "Online TA",
        },
    )
    probs = profile["survival_probability"].tolist()
    assert all(0 <= p <= 1 for p in probs)
    assert probs == sorted(probs, reverse=True)
