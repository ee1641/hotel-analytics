"""Tests for src/monitoring.py — metrics history, drift detection, retrain orchestration."""
import json
from pathlib import Path

import pytest

from src.data_context import DatasetContext, build_demo_dataset_context
from src.monitoring import (
    append_snapshot,
    build_metrics_snapshot,
    detect_drift,
    history_path_for,
    load_history,
    run_retrain,
)


def _snapshot(**overrides):
    base = {
        "timestamp": "2026-06-01T00:00:00+00:00",
        "dataset_id": "demo",
        "demand_mape": 12.0,
        "revenue_mape": 14.0,
        "cancel_accuracy": 0.78,
        "cancel_brier": 0.15,
        "revenue_coverage": 0.90,
        "demand_coverage": 0.90,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# History file I/O
# ---------------------------------------------------------------------------

def test_append_and_load_history_roundtrip(tmp_path):
    path = tmp_path / "metrics_history.jsonl"
    append_snapshot(_snapshot(demand_mape=11.0), path)
    append_snapshot(_snapshot(demand_mape=12.5), path)
    history = load_history(path)
    assert len(history) == 2
    assert history[0]["demand_mape"] == 11.0
    assert history[1]["demand_mape"] == 12.5


def test_load_history_skips_malformed_lines(tmp_path):
    path = tmp_path / "metrics_history.jsonl"
    path.write_text(
        json.dumps(_snapshot()) + "\n"
        + "{not valid json\n"
        + "\n"
        + json.dumps(_snapshot(demand_mape=13.0)) + "\n",
        encoding="utf-8",
    )
    history = load_history(path)
    assert len(history) == 2


def test_load_history_missing_file_returns_empty(tmp_path):
    assert load_history(tmp_path / "nope.jsonl") == []


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def test_detect_drift_stable_history_no_findings():
    previous = [_snapshot() for _ in range(5)]
    findings = detect_drift(previous, _snapshot())
    assert findings == []


def test_detect_drift_flags_mape_warning_and_critical():
    previous = [_snapshot(demand_mape=12.0) for _ in range(5)]

    warn = detect_drift(previous, _snapshot(demand_mape=15.0))
    warn_metrics = {f.metric: f.severity for f in warn}
    assert warn_metrics.get("demand_mape") == "warning"

    crit = detect_drift(previous, _snapshot(demand_mape=18.0))
    crit_metrics = {f.metric: f.severity for f in crit}
    assert crit_metrics.get("demand_mape") == "critical"


def test_detect_drift_absolute_floor_blocks_tiny_base_noise():
    """22% relative rise on a small base (<2pp absolute) must not flag."""
    previous = [_snapshot(demand_mape=4.0) for _ in range(5)]
    findings = detect_drift(previous, _snapshot(demand_mape=4.9))
    assert not any(f.metric == "demand_mape" for f in findings)


def test_detect_drift_accuracy_drop():
    previous = [_snapshot(cancel_accuracy=0.78) for _ in range(5)]
    warn = detect_drift(previous, _snapshot(cancel_accuracy=0.74))
    assert any(f.metric == "cancel_accuracy" and f.severity == "warning" for f in warn)
    crit = detect_drift(previous, _snapshot(cancel_accuracy=0.70))
    assert any(f.metric == "cancel_accuracy" and f.severity == "critical" for f in crit)


def test_detect_drift_coverage_floors():
    previous = [_snapshot() for _ in range(5)]
    warn = detect_drift(previous, _snapshot(revenue_coverage=0.84))
    assert any(f.metric == "revenue_coverage" and f.severity == "warning" for f in warn)
    crit = detect_drift(previous, _snapshot(revenue_coverage=0.78))
    assert any(f.metric == "revenue_coverage" and f.severity == "critical" for f in crit)


def test_detect_drift_insufficient_history_absolute_floors_only():
    previous = [_snapshot(demand_mape=12.0)]  # < min_history of 3

    # Would be critical vs baseline, but with thin history only the ceiling applies.
    quiet = detect_drift(previous, _snapshot(demand_mape=20.0))
    assert not any(f.metric == "demand_mape" for f in quiet)

    ceiling = detect_drift(previous, _snapshot(demand_mape=30.0))
    assert any(f.metric == "demand_mape" for f in ceiling)

    # Coverage floors are absolute and always apply.
    cov = detect_drift(previous, _snapshot(demand_coverage=0.78))
    assert any(f.metric == "demand_coverage" and f.severity == "critical" for f in cov)


def test_detect_drift_per_bin_warning_only():
    previous = [_snapshot() for _ in range(5)]
    latest = _snapshot(
        revenue_coverage_by_bin={"15-30": {"coverage": 0.5, "mean_width": 100.0, "n": 16}},
    )
    findings = detect_drift(previous, latest)
    bin_findings = [f for f in findings if "by_bin" in f.metric]
    assert bin_findings and all(f.severity == "warning" for f in bin_findings)


def test_detect_drift_ignores_none_metrics():
    previous = [_snapshot() for _ in range(5)]
    findings = detect_drift(previous, _snapshot(demand_mape=None, revenue_coverage=None))
    assert not any(f.metric in ("demand_mape", "revenue_coverage") for f in findings)


# ---------------------------------------------------------------------------
# Snapshot construction
# ---------------------------------------------------------------------------

def _fake_results():
    return {
        "eval_metrics": {
            "demand_xgb": {"mape": 12.4, "mae": 10.1, "rmse": 14.0},
            "revenue_xgb": {"mape": 18.0, "mae": 4000.0, "rmse": 5000.0},
            "revenue_conformal_coverage": {
                "coverage": 0.93, "mean_width": 9000.0, "n": 30,
                "alpha": 0.10, "nominal_coverage": 0.90,
                "by_bin": {"1-7": {"coverage": 1.0, "mean_width": 7000.0, "n": 7}},
            },
            "demand_conformal_coverage": {
                "coverage": 0.87, "mean_width": 55.0, "n": 30,
                "alpha": 0.10, "nominal_coverage": 0.90,
                "by_bin": {"1-7": {"coverage": 0.86, "mean_width": 40.0, "n": 7}},
            },
        },
        "cancel_models": {
            "holdout_metrics": {
                "accuracy": 0.78, "brier": 0.151, "ece": 0.021,
                "roc_auc": 0.854, "threshold": 0.60, "n_test": 1000,
            },
        },
    }


def test_build_metrics_snapshot_flattens_results():
    snap = build_metrics_snapshot(_fake_results(), "demo")
    assert snap["dataset_id"] == "demo"
    assert snap["demand_mape"] == 12.4
    assert snap["revenue_mape"] == 18.0
    assert snap["cancel_accuracy"] == 0.78
    assert snap["cancel_brier"] == 0.151
    assert snap["revenue_coverage"] == 0.93
    assert snap["revenue_coverage_by_bin"]["1-7"]["coverage"] == 1.0
    assert "timestamp" in snap and "versions" in snap
    json.dumps(snap)  # must be JSON-serializable


def test_build_metrics_snapshot_tolerates_missing_blocks():
    snap = build_metrics_snapshot({}, "demo")
    assert snap["demand_mape"] is None
    assert snap["cancel_accuracy"] is None
    assert snap["revenue_coverage_by_bin"] is None
    json.dumps(snap)


# ---------------------------------------------------------------------------
# Retrain orchestration (stubbed training — the real path takes minutes)
# ---------------------------------------------------------------------------

def test_run_retrain_smoke_with_stub_train_fn(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "stale_artifact.pkl").write_bytes(b"old")
    history_path = tmp_path / "metrics_history.jsonl"

    context = DatasetContext(
        dataset_id="test_ds",
        data_path=str(tmp_path / "bookings.csv"),
        models_dir=str(models_dir),
        source_label="Test",
    )

    calls = {}

    def stub_train(dataset_context=None):
        calls["context"] = dataset_context
        return _fake_results()

    snapshot, findings = run_retrain(
        context, force=True, history_path=history_path, train_fn=stub_train,
    )

    assert calls["context"] is context
    # Force cleared the stale artifact; the history file survived outside models_dir.
    assert list(models_dir.iterdir()) == []
    history = load_history(history_path)
    assert len(history) == 1
    assert history[0]["demand_mape"] == 12.4
    assert isinstance(snapshot["drift"], list)
    assert findings == []  # first run: no baseline, all floors satisfied


def test_run_retrain_no_force_keeps_models(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    keep = models_dir / "artifact.pkl"
    keep.write_bytes(b"keep me")

    context = DatasetContext(
        dataset_id="test_ds", data_path="x.csv",
        models_dir=str(models_dir), source_label="Test",
    )
    run_retrain(
        context, force=False,
        history_path=tmp_path / "h.jsonl",
        train_fn=lambda dataset_context=None: _fake_results(),
    )
    assert keep.exists()


# ---------------------------------------------------------------------------
# History path resolution
# ---------------------------------------------------------------------------

def test_history_path_outside_models_dir():
    demo = build_demo_dataset_context()
    demo_path = history_path_for(demo)
    assert Path(demo.models_dir).resolve() not in demo_path.resolve().parents

    hotel = DatasetContext(
        dataset_id="hotel_abc_123",
        data_path="data/hotels/abc/bookings.csv",
        models_dir="data/hotels/abc/models",
        source_label="Hotel Upload (abc)",
    )
    hotel_path = history_path_for(hotel)
    assert hotel_path == Path("data/hotels/abc/metrics_history.jsonl")
    assert Path(hotel.models_dir).resolve() not in hotel_path.resolve().parents
