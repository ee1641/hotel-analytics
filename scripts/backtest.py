"""
Rolling-origin backtest CLI — the arbiter for forecast-model claims.

Evaluates every candidate model across K rolling windows so accuracy claims
rest on a distribution, not the single (easy) final holdout. Use this to
decide whether ensemble / direct / new-feature variants actually beat the
production xgb_recursive model before promoting them.

Run from the repository root:
    python scripts/backtest.py --series demand  --windows 12 --horizon 30
    python scripts/backtest.py --series revenue --windows 12 --horizon 30
    python scripts/backtest.py --series demand --candidates xgb_recursive,ensemble
    python scripts/backtest.py --series demand --no-extras   # ablate exogenous feats

Writes outputs/backtest_<series>.csv (one row per window x candidate) and
prints the per-candidate summary table. Slow: direct multi-step trains one
model per horizon per window.
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.backtest import default_candidates, rolling_backtest, summarize_backtest
from src.config import APP_CONFIG
from src.services import (
    DATA_PATH,
    DEMAND_FUTURE_FEATURE_COLS,
    build_demand_extra_features,
    load_and_prepare_data,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--series", choices=["demand", "revenue"], default="demand")
    parser.add_argument("--data", default=DATA_PATH, help="Bookings CSV (default: demo)")
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--step", type=int, default=30)
    parser.add_argument(
        "--candidates", default=None,
        help="Comma-separated subset of: xgb_recursive,xgb_direct,prophet,naive,ensemble",
    )
    parser.add_argument(
        "--no-extras", action="store_true",
        help="Ablate exogenous demand features (trends/OTB/pace).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    df, daily_df = load_and_prepare_data(args.data)
    if args.series == "demand":
        series = daily_df["daily_bookings"]
        extras = None if args.no_extras else build_demand_extra_features(df, daily_df)
        future_cols = None if args.no_extras else DEMAND_FUTURE_FEATURE_COLS
    else:
        series = daily_df["daily_revenue"]
        extras, future_cols = None, None  # revenue is trained without extras in production

    names = args.candidates.split(",") if args.candidates else None
    candidates = default_candidates(names)

    print(f"Series   : {args.series} ({len(series)} daily points)")
    print(f"Windows  : {args.windows} requested, horizon={args.horizon}, step={args.step}")
    print(f"Candidates: {', '.join(candidates)}")
    print(f"Extras   : {'none' if extras is None else f'{extras.shape[1]} cols'}\n")

    results = rolling_backtest(
        series,
        candidates=candidates,
        n_windows=args.windows,
        horizon=args.horizon,
        step=args.step,
        extra_features=extras,
        future_feature_cols=future_cols,
        holiday_country=APP_CONFIG.get("holiday_country", "PT"),
    )

    if results.empty:
        print("No backtest results produced (series too short?).")
        return 2

    summary = summarize_backtest(results)
    actual_windows = results["window"].nunique()
    print(f"Completed {actual_windows} window(s).\n")
    print("Per-candidate summary (sorted by mean MAPE, lower is better):")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary.to_string())

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"backtest_{args.series}.csv"
    results.to_csv(out_path, index=False)
    print(f"\nPer-window detail written to {out_path}")

    best = summary.index[0]
    print(f"\nLowest mean MAPE: {best} ({summary.loc[best, 'mape_mean']:.2f}%, "
          f"{summary.loc[best, 'wins']} window win(s))")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        logging.getLogger(__name__).error("Backtest failed: %s", exc, exc_info=True)
        raise SystemExit(2)
