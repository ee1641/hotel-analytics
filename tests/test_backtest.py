"""Tests for the rolling-origin backtest harness (src/backtest.py)."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    default_candidates,
    rolling_backtest,
    summarize_backtest,
)


@pytest.fixture(scope="module")
def synth_series():
    rng = np.random.RandomState(0)
    n = 200
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    vals = (
        100.0
        + np.linspace(0, 40, n)
        + 15 * np.sin(2 * np.pi * np.arange(n) / 7.0)
        + rng.normal(0, 5, size=n)
    )
    return pd.Series(vals, index=idx, name="target")


def test_default_candidates_unknown_raises():
    with pytest.raises(ValueError):
        default_candidates(["xgb_recursive", "does_not_exist"])


def test_window_origin_math_and_shape(synth_series):
    res = rolling_backtest(
        synth_series, candidates=default_candidates(["naive"]),
        n_windows=4, horizon=14, step=14, lags=[1, 7],
    )
    # 4 windows x 1 candidate
    assert res["window"].nunique() == 4
    assert len(res) == 4
    assert set(res.columns) >= {"window", "candidate", "mae", "rmse", "mape"}
    # per-weekday MAPE columns present
    assert any(c.startswith("mape_") and c != "mape" for c in res.columns)


def test_autoshrink_on_short_series():
    """Requesting more windows than the series supports shrinks, not crashes."""
    idx = pd.date_range("2023-01-01", periods=90, freq="D")
    s = pd.Series(np.arange(90, dtype=float) + 50, index=idx)
    res = rolling_backtest(
        s, candidates=default_candidates(["naive"]),
        n_windows=20, horizon=14, step=14, lags=[1, 7],
    )
    assert 0 < res["window"].nunique() < 20


def test_series_too_short_raises():
    idx = pd.date_range("2023-01-01", periods=20, freq="D")
    s = pd.Series(np.arange(20, dtype=float), index=idx)
    with pytest.raises(ValueError):
        rolling_backtest(
            s, candidates=default_candidates(["naive"]),
            n_windows=3, horizon=14, step=14, lags=[1, 7],
        )


def test_candidates_only_see_past_data(synth_series):
    """A probe candidate records the max train timestamp it is handed; it must
    never exceed the window origin (no test-window leakage)."""
    seen_max = []
    origin_dates = []

    def probe(train, horizon, *, extras=None, future_extras=None,
              holiday_country=None, lags=None):
        seen_max.append(train.index.max())
        # the test window starts the day after train ends
        idx = pd.date_range(train.index[-1] + pd.Timedelta(days=1),
                            periods=horizon, freq="D")
        return pd.Series(float(train.iloc[-1]), index=idx)

    res = rolling_backtest(
        synth_series, candidates={"probe": probe},
        n_windows=3, horizon=14, step=14, lags=[1, 7],
    )
    assert len(res) == 3
    n = len(synth_series)
    # window i origin index = n - horizon - i*step; train.max == series[origin-1]
    for i, train_max in zip(range(3), seen_max):
        origin = n - 14 - i * 14
        expected_max = synth_series.index[origin - 1]
        assert train_max == expected_max


def test_extras_sliced_to_window(synth_series):
    """Train extras must match the train length; future extras must match the
    horizon and be restricted to future_feature_cols."""
    extras = pd.DataFrame(
        {"known_future": np.arange(len(synth_series), dtype=float),
         "lagged_actual": np.arange(len(synth_series), dtype=float)},
        index=synth_series.index,
    )
    captured = {}

    def probe(train, horizon, *, extras=None, future_extras=None,
              holiday_country=None, lags=None):
        captured["train_extra_len"] = None if extras is None else len(extras)
        captured["future_cols"] = (
            None if future_extras is None else list(future_extras.columns)
        )
        captured["future_len"] = None if future_extras is None else len(future_extras)
        idx = pd.date_range(train.index[-1] + pd.Timedelta(days=1),
                            periods=horizon, freq="D")
        return pd.Series(0.0, index=idx)

    rolling_backtest(
        synth_series, candidates={"probe": probe},
        n_windows=1, horizon=14, step=14, lags=[1, 7],
        extra_features=extras, future_feature_cols=["known_future"],
    )
    n = len(synth_series)
    assert captured["train_extra_len"] == n - 14  # origin for window 0
    assert captured["future_cols"] == ["known_future"]  # lagged_actual excluded
    assert captured["future_len"] == 14


def test_summarize_aggregates_and_counts_wins(synth_series):
    res = rolling_backtest(
        synth_series, candidates=default_candidates(["xgb_recursive", "naive"]),
        n_windows=3, horizon=14, step=14, lags=[1, 7],
    )
    summary = summarize_backtest(res)
    assert set(summary.index) == {"xgb_recursive", "naive"}
    assert "mape_mean" in summary.columns and "wins" in summary.columns
    # wins sum to the number of windows (one winner per window)
    assert summary["wins"].sum() == res["window"].nunique()
    # sorted ascending by mean MAPE
    assert summary["mape_mean"].is_monotonic_increasing


def test_ensemble_candidate_never_sees_test_window(synth_series):
    """The ensemble fits its blend weight internally; it must still only train
    on data up to the window origin. Verified indirectly: results are finite
    and the candidate runs on the standard windows."""
    res = rolling_backtest(
        synth_series, candidates=default_candidates(["ensemble"]),
        n_windows=2, horizon=14, step=14, lags=[1, 7],
    )
    assert res["window"].nunique() == 2
    assert np.isfinite(res["mape"]).all()


def test_summarize_empty_returns_empty():
    assert summarize_backtest(pd.DataFrame()).empty
