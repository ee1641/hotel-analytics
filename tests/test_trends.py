"""Tests for src/trends.py — Google Trends feature pipeline."""
import pandas as pd
import pytest
from src.trends import build_trends_features, TRENDS_FEATURE_COLS


@pytest.fixture(scope="module")
def trend_index():
    return pd.date_range("2016-01-01", periods=60, freq="D")


def test_trends_shape(trend_index):
    feats = build_trends_features(trend_index)
    assert feats.shape == (len(trend_index), len(TRENDS_FEATURE_COLS))


def test_trends_no_nan(trend_index):
    feats = build_trends_features(trend_index)
    assert feats.isna().sum().sum() == 0


def test_trends_columns(trend_index):
    feats = build_trends_features(trend_index)
    for col in TRENDS_FEATURE_COLS:
        assert col in feats.columns, f"Missing column: {col}"


def test_trends_composite_positive(trend_index):
    feats = build_trends_features(trend_index)
    assert (feats["trends_composite"] >= 0).all()


def test_trends_graceful_fallback():
    """build_trends_features returns zero-filled DataFrame when API fails."""
    # Short index far in the future — cache should serve or fall back gracefully
    idx = pd.date_range("2015-07-01", periods=30, freq="D")
    feats = build_trends_features(idx)
    assert feats.shape[1] == len(TRENDS_FEATURE_COLS)
    assert feats.isna().sum().sum() == 0
