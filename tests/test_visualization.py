import os
import numpy as np
import pandas as pd
import pytest
from src.visualization import (
    plot_revenue_over_time,
    plot_forecast_vs_actual,
    plot_feature_importance,
    plot_cancellation_roc,
)

OUTPUT_DIR = "outputs"


@pytest.fixture(scope="module")
def daily_df():
    dates = pd.date_range("2022-01-01", periods=100, freq="D")
    return pd.DataFrame({
        "daily_revenue": np.random.rand(100) * 10_000,
        "daily_bookings": np.random.randint(10, 100, 100),
    }, index=dates)


def test_revenue_over_time_creates_file(daily_df):
    plot_revenue_over_time(daily_df, output_dir=OUTPUT_DIR)
    assert os.path.exists(f"{OUTPUT_DIR}/revenue_over_time.png")


def test_forecast_vs_actual_creates_file():
    dates = pd.date_range("2022-01-01", periods=50, freq="D")
    actual = pd.Series(np.random.rand(50) * 1000, index=dates)
    forecast = pd.Series(np.random.rand(50) * 1000, index=dates)
    plot_forecast_vs_actual(actual, forecast, title="Test", output_dir=OUTPUT_DIR)
    assert os.path.exists(f"{OUTPUT_DIR}/Test_forecast.png")


def test_feature_importance_creates_file():
    from xgboost import XGBClassifier
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    model = XGBClassifier(verbosity=0).fit(X, y)
    plot_feature_importance(model, [f"f{i}" for i in range(5)], output_dir=OUTPUT_DIR)
    assert os.path.exists(f"{OUTPUT_DIR}/feature_importance.png")


def test_cancellation_roc_creates_file():
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=300, n_features=10, random_state=42)
    model = LogisticRegression().fit(X[:240], y[:240])
    plot_cancellation_roc({"logistic": model}, X[240:], y[240:], output_dir=OUTPUT_DIR)
    assert os.path.exists(f"{OUTPUT_DIR}/cancellation_roc.png")
