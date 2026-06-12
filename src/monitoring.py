"""
Model retraining orchestration and drift monitoring.

Every retrain (scripts/retrain.py) appends a metrics snapshot — forecast MAPE,
cancellation accuracy/Brier/ECE, conformal interval coverage — to a JSONL
history file. ``detect_drift`` compares the latest snapshot against the median
of recent ones and emits findings that surface as ``model_drift`` alerts in
the dashboard and API (src/alerts.py).

The history file lives *beside* the models directory, never inside it:
``run_retrain(force=True)`` clears the models dir to invalidate the cache, and
the metric trail must survive that.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from src.data_context import DatasetContext, clear_models_dir
from src.model_specs import (
    CANCELLATION_MODEL_VERSION,
    CONFORMAL_INTERVAL_VERSION,
    EVAL_METRICS_VERSION,
    FORECAST_MODEL_VERSION,
)

log = logging.getLogger(__name__)

# Thresholds calibrated against the demo dataset's holdout noise:
# - MAPE sits at 12-13% and a 30-day holdout swings ~1-2pp run to run, so the
#   relative trigger carries an absolute floor to suppress flapping.
# - Cancellation holdout has ~24k rows (binomial SE ~0.27pp); a 3pp drop is
#   ~11 standard errors — unambiguous.
# - Coverage on a 30-day window has SE ~5.5pp at the 90% nominal level, so
#   -5pp is ~1 SE (warn) and -10pp ~2 SE (critical). Per-bin coverage rests on
#   7-16 points and only ever warns, below a looser floor.
DRIFT_CONFIG = {
    "baseline_window": 5,
    "min_history": 3,
    "mape_warn_rel": 1.20,
    "mape_warn_abs_pp": 2.0,
    "mape_crit_rel": 1.40,
    "mape_crit_abs_pp": 4.0,
    "mape_absolute_ceiling": 25.0,
    "accuracy_warn_drop": 0.03,
    "accuracy_crit_drop": 0.06,
    "brier_warn_rel": 1.15,
    "brier_crit_rel": 1.30,
    "coverage_warn_floor": 0.85,
    "coverage_crit_floor": 0.80,
    "bin_coverage_floor": 0.75,
}


@dataclass
class DriftFinding:
    metric: str
    current: float
    baseline: float | None
    severity: str  # "warning" | "critical"
    message: str


def history_path_for(context: DatasetContext) -> Path:
    """Metrics history location for a dataset context — always outside models_dir."""
    if context.dataset_id == "demo":
        return Path("data/metrics_history.jsonl")
    return Path(context.models_dir).parent / "metrics_history.jsonl"


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _coverage_bins(coverage_block: dict | None) -> dict | None:
    if not isinstance(coverage_block, dict):
        return None
    by_bin = coverage_block.get("by_bin")
    if not isinstance(by_bin, dict):
        return None
    return {
        label: {
            "coverage": _safe_float(stats.get("coverage")),
            "mean_width": _safe_float(stats.get("mean_width")),
            "n": int(stats.get("n", 0)),
        }
        for label, stats in by_bin.items()
        if isinstance(stats, dict)
    }


def build_metrics_snapshot(results: dict, dataset_id: str) -> dict:
    """Flatten a get_or_train_models() result into one JSON-serializable record."""
    eval_metrics = results.get("eval_metrics") or {}
    cancel_models = results.get("cancel_models") or {}
    holdout = cancel_models.get("holdout_metrics") if isinstance(cancel_models, dict) else None
    holdout = holdout if isinstance(holdout, dict) else {}
    rev_cov = eval_metrics.get("revenue_conformal_coverage") or {}
    dem_cov = eval_metrics.get("demand_conformal_coverage") or {}

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "versions": {
            "forecast": FORECAST_MODEL_VERSION,
            "cancellation": CANCELLATION_MODEL_VERSION,
            "conformal": CONFORMAL_INTERVAL_VERSION,
            "eval": EVAL_METRICS_VERSION,
        },
        "demand_mape": _safe_float((eval_metrics.get("demand_xgb") or {}).get("mape")),
        "demand_mae": _safe_float((eval_metrics.get("demand_xgb") or {}).get("mae")),
        "revenue_mape": _safe_float((eval_metrics.get("revenue_xgb") or {}).get("mape")),
        "revenue_mae": _safe_float((eval_metrics.get("revenue_xgb") or {}).get("mae")),
        "cancel_accuracy": _safe_float(holdout.get("accuracy")),
        "cancel_brier": _safe_float(holdout.get("brier")),
        "cancel_ece": _safe_float(holdout.get("ece")),
        "cancel_roc_auc": _safe_float(holdout.get("roc_auc")),
        "revenue_coverage": _safe_float(rev_cov.get("coverage")),
        "revenue_mean_width": _safe_float(rev_cov.get("mean_width")),
        "revenue_coverage_by_bin": _coverage_bins(rev_cov),
        "demand_coverage": _safe_float(dem_cov.get("coverage")),
        "demand_mean_width": _safe_float(dem_cov.get("mean_width")),
        "demand_coverage_by_bin": _coverage_bins(dem_cov),
    }


def append_snapshot(snapshot: dict, history_path: str | Path) -> None:
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(snapshot) + "\n")


def load_history(history_path: str | Path) -> list[dict]:
    path = Path(history_path)
    if not path.exists():
        return []
    snapshots = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            log.warning("Skipping malformed line %d in %s", i + 1, path)
            continue
        if isinstance(record, dict):
            snapshots.append(record)
    return snapshots


def _baseline(previous: list[dict], metric: str, window: int) -> float | None:
    values = [
        v for s in previous[-window:]
        if (v := _safe_float(s.get(metric))) is not None
    ]
    return float(median(values)) if values else None


def detect_drift(previous: list[dict], latest: dict, config: dict | None = None) -> list[DriftFinding]:
    """
    Compare the latest snapshot against the trailing-median baseline.

    With fewer than ``min_history`` previous snapshots only absolute floors
    apply (coverage floors + MAPE ceiling) — relative comparisons against one
    or two runs would just amplify sampling noise.
    """
    cfg = {**DRIFT_CONFIG, **(config or {})}
    previous = [s for s in previous if isinstance(s, dict)]
    findings: list[DriftFinding] = []

    def add(metric, current, baseline, severity, message):
        findings.append(DriftFinding(
            metric=metric, current=float(current),
            baseline=float(baseline) if baseline is not None else None,
            severity=severity, message=message,
        ))

    has_baseline = len(previous) >= cfg["min_history"]

    # --- MAPE: relative + absolute-floor against baseline; absolute ceiling always
    for metric in ("demand_mape", "revenue_mape"):
        cur = _safe_float(latest.get(metric))
        if cur is None:
            continue
        if has_baseline:
            base = _baseline(previous, metric, cfg["baseline_window"])
            if base is not None:
                delta = cur - base
                if cur > base * cfg["mape_crit_rel"] and delta > cfg["mape_crit_abs_pp"]:
                    add(metric, cur, base, "critical",
                        f"{metric} rose to {cur:.1f}% (baseline {base:.1f}%, +{delta:.1f}pp)")
                    continue
                if cur > base * cfg["mape_warn_rel"] and delta > cfg["mape_warn_abs_pp"]:
                    add(metric, cur, base, "warning",
                        f"{metric} rose to {cur:.1f}% (baseline {base:.1f}%, +{delta:.1f}pp)")
                    continue
        if cur > cfg["mape_absolute_ceiling"]:
            add(metric, cur, None, "warning",
                f"{metric} at {cur:.1f}% exceeds the {cfg['mape_absolute_ceiling']:.0f}% absolute ceiling")

    # --- Cancellation accuracy / Brier: baseline-relative only
    if has_baseline:
        cur = _safe_float(latest.get("cancel_accuracy"))
        base = _baseline(previous, "cancel_accuracy", cfg["baseline_window"])
        if cur is not None and base is not None:
            drop = base - cur
            if drop > cfg["accuracy_crit_drop"]:
                add("cancel_accuracy", cur, base, "critical",
                    f"cancel_accuracy fell to {cur:.3f} (baseline {base:.3f}, -{drop * 100:.1f}pp)")
            elif drop > cfg["accuracy_warn_drop"]:
                add("cancel_accuracy", cur, base, "warning",
                    f"cancel_accuracy fell to {cur:.3f} (baseline {base:.3f}, -{drop * 100:.1f}pp)")

        cur = _safe_float(latest.get("cancel_brier"))
        base = _baseline(previous, "cancel_brier", cfg["baseline_window"])
        if cur is not None and base is not None and base > 0:
            if cur > base * cfg["brier_crit_rel"]:
                add("cancel_brier", cur, base, "critical",
                    f"cancel_brier worsened to {cur:.4f} (baseline {base:.4f})")
            elif cur > base * cfg["brier_warn_rel"]:
                add("cancel_brier", cur, base, "warning",
                    f"cancel_brier worsened to {cur:.4f} (baseline {base:.4f})")

    # --- Conformal coverage: absolute floors, no baseline needed
    for metric in ("revenue_coverage", "demand_coverage"):
        cur = _safe_float(latest.get(metric))
        if cur is None:
            continue
        if cur < cfg["coverage_crit_floor"]:
            add(metric, cur, None, "critical",
                f"{metric} at {cur:.0%} is below the {cfg['coverage_crit_floor']:.0%} critical floor "
                f"(nominal 90%)")
        elif cur < cfg["coverage_warn_floor"]:
            add(metric, cur, None, "warning",
                f"{metric} at {cur:.0%} is below the {cfg['coverage_warn_floor']:.0%} warning floor "
                f"(nominal 90%)")

    # --- Per-bin coverage: tiny samples (7-16 points) → warning only, looser floor
    for metric in ("revenue_coverage_by_bin", "demand_coverage_by_bin"):
        bins = latest.get(metric)
        if not isinstance(bins, dict):
            continue
        for label, stats in bins.items():
            cov = _safe_float(stats.get("coverage")) if isinstance(stats, dict) else None
            if cov is not None and cov < cfg["bin_coverage_floor"]:
                add(f"{metric}[{label}]", cov, None, "warning",
                    f"{metric.split('_')[0]} coverage in horizon bin {label} at {cov:.0%} "
                    f"is below the {cfg['bin_coverage_floor']:.0%} floor")

    return findings


def run_retrain(
    context: DatasetContext,
    *,
    force: bool = True,
    history_path: str | Path | None = None,
    train_fn=None,
    drift_config: dict | None = None,
) -> tuple[dict, list[DriftFinding]]:
    """
    Retrain models for a dataset context, snapshot metrics, detect drift.

    ``force=True`` clears the models dir so every artifact retrains.
    get_or_train_models saves each artifact as it completes, so a mid-run
    crash resumes from the last saved artifact on the next invocation.

    ``train_fn`` is injectable for tests; defaults to
    ``src.services.get_or_train_models``.

    Returns (snapshot, findings). The snapshot — including its ``drift``
    list — has already been appended to the history file.
    """
    if train_fn is None:
        from src.services import get_or_train_models as train_fn  # deferred heavy import

    path = Path(history_path) if history_path is not None else history_path_for(context)

    if force:
        clear_models_dir(context.models_dir)

    results = train_fn(dataset_context=context)
    snapshot = build_metrics_snapshot(results, context.dataset_id)
    findings = detect_drift(load_history(path), snapshot, drift_config)
    snapshot["drift"] = [asdict(f) for f in findings]
    append_snapshot(snapshot, path)
    return snapshot, findings
