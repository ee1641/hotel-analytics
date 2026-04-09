"""
Tests for SHAP-based feature selection in src/cancellation.py.
"""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from src.cancellation import (
    train_cancellation_models_balanced,
    get_shap_feature_importance,
    select_top_features,
)


@pytest.fixture(scope="module")
def trained_xgb_and_data():
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    feature_names = [f"feat_{i}" for i in range(10)]
    models = train_cancellation_models_balanced(X[:400], y[:400])
    return models["xgboost"], X[400:], feature_names


def test_shap_returns_correct_columns(trained_xgb_and_data):
    model, X_test, names = trained_xgb_and_data
    df = get_shap_feature_importance(model, X_test, names)
    assert "feature" in df.columns
    assert "mean_abs_shap" in df.columns


def test_shap_returns_all_features(trained_xgb_and_data):
    model, X_test, names = trained_xgb_and_data
    df = get_shap_feature_importance(model, X_test, names)
    assert len(df) == len(names)
    assert set(df["feature"].tolist()) == set(names)


def test_shap_sorted_descending(trained_xgb_and_data):
    model, X_test, names = trained_xgb_and_data
    df = get_shap_feature_importance(model, X_test, names)
    vals = df["mean_abs_shap"].tolist()
    assert vals == sorted(vals, reverse=True)


def test_shap_values_non_negative(trained_xgb_and_data):
    model, X_test, names = trained_xgb_and_data
    df = get_shap_feature_importance(model, X_test, names)
    assert (df["mean_abs_shap"] >= 0).all()


def test_select_top_features_subset(trained_xgb_and_data):
    model, X_test, names = trained_xgb_and_data
    df = get_shap_feature_importance(model, X_test, names)
    selected = select_top_features(df, threshold=0.001)
    assert len(selected) <= len(names)
    assert all(f in names for f in selected)


def test_select_top_features_zero_threshold(trained_xgb_and_data):
    model, X_test, names = trained_xgb_and_data
    df = get_shap_feature_importance(model, X_test, names)
    selected = select_top_features(df, threshold=0.0)
    assert len(selected) == len(names)


def test_select_top_features_high_threshold(trained_xgb_and_data):
    model, X_test, names = trained_xgb_and_data
    df = get_shap_feature_importance(model, X_test, names)
    selected = select_top_features(df, threshold=999.0)
    assert selected == []
