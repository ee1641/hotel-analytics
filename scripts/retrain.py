"""
Retrain models, snapshot metrics, and detect drift.

Run from the repository root:
    python scripts/retrain.py                     # demo dataset, force retrain
    python scripts/retrain.py --hotel-id abc123   # one registered hotel
    python scripts/retrain.py --no-force          # snapshot current cached metrics only

Exit codes (for Task Scheduler / CI gating):
    0  healthy — retrained, no drift
    1  drift detected — see findings in output and metrics_history.jsonl
    2  execution error

Scheduling recipes: see docs/retraining.md.
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.data_context import (
    DatasetContext,
    build_demo_dataset_context,
    fingerprint_file,
    resolve_hotel_dataset_context,
)
from src.monitoring import run_retrain


def _context_from_args(args) -> DatasetContext:
    if args.hotel_id:
        return resolve_hotel_dataset_context(args.hotel_id)
    if args.data:
        data_path = Path(args.data)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        demo = build_demo_dataset_context()
        if data_path.resolve() == Path(demo.data_path).resolve():
            return demo
        dataset_id = f"adhoc_{fingerprint_file(data_path)}"
        models_dir = Path("data/adhoc") / dataset_id / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        return DatasetContext(
            dataset_id=dataset_id,
            data_path=str(data_path),
            models_dir=str(models_dir),
            source_label=f"Ad-hoc CSV ({data_path.name})",
        )
    return build_demo_dataset_context()


def _fmt(value, spec=".2f", suffix="") -> str:
    if value is None:
        return "n/a"
    return f"{value:{spec}}{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data", default=None,
                        help="Path to a bookings CSV (default: built-in demo dataset)")
    source.add_argument("--hotel-id", default=None,
                        help="Registered hotel id (data/hotels/<id>/)")
    parser.add_argument("--no-force", action="store_true",
                        help="Reuse cached models instead of clearing them; just snapshot metrics")
    parser.add_argument("--history-path", default=None,
                        help="Override the metrics history JSONL location")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    context = _context_from_args(args)
    print(f"Dataset : {context.source_label} ({context.dataset_id})")
    print(f"Models  : {context.models_dir}")
    print(f"Force   : {not args.no_force}")

    snapshot, findings = run_retrain(
        context,
        force=not args.no_force,
        history_path=args.history_path,
    )

    print("\nMetrics snapshot")
    print(f"  demand MAPE      : {_fmt(snapshot['demand_mape'], '.1f', '%')}")
    print(f"  revenue MAPE     : {_fmt(snapshot['revenue_mape'], '.1f', '%')}")
    print(f"  cancel accuracy  : {_fmt(snapshot['cancel_accuracy'], '.3f')}")
    print(f"  cancel Brier     : {_fmt(snapshot['cancel_brier'], '.4f')}")
    print(f"  cancel ECE       : {_fmt(snapshot['cancel_ece'], '.4f')}")
    print(f"  revenue coverage : {_fmt(snapshot['revenue_coverage'], '.0%')}")
    print(f"  demand coverage  : {_fmt(snapshot['demand_coverage'], '.0%')}")

    if findings:
        print(f"\nDRIFT DETECTED — {len(findings)} finding(s):")
        for f in findings:
            print(f"  [{f.severity.upper():<8s}] {f.message}")
        return 1

    print("\nNo drift detected.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # exit code 2 for schedulers to distinguish crash from drift
        logging.getLogger(__name__).error("Retrain failed: %s", exc, exc_info=True)
        raise SystemExit(2)
