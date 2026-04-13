import numpy as np
import pandas as pd

from src.revenue_management import (
    build_pickup_report,
    build_segment_mix_report,
    fit_price_elasticity_models,
    recommend_dynamic_prices,
)


def test_pickup_report_counts_active_bookings():
    df = pd.DataFrame(
        [
            {
                "arrival_date": "2024-02-10",
                "lead_time": 10,
                "is_canceled": 0,
                "reservation_status_date": "2024-02-12",
                "adr": 100.0,
                "market_segment": "Direct",
            },
            {
                "arrival_date": "2024-02-10",
                "lead_time": 5,
                "is_canceled": 1,
                "reservation_status_date": "2024-02-08",
                "adr": 120.0,
                "market_segment": "Online TA",
            },
            {
                "arrival_date": "2024-02-11",
                "lead_time": 10,
                "is_canceled": 1,
                "reservation_status_date": "2024-02-05",
                "adr": 90.0,
                "market_segment": "Direct",
            },
        ]
    )
    df["arrival_date"] = pd.to_datetime(df["arrival_date"])
    df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"])

    report = build_pickup_report(df, snapshot_date=pd.Timestamp("2024-02-06"), periods=5)

    row = report.loc[report["date"] == pd.Timestamp("2024-02-10")].iloc[0]
    assert row["otb_today"] == 2
    assert row["pickup_7d"] >= 0


def test_segment_mix_report_returns_share_columns():
    df = pd.DataFrame(
        [
            {
                "arrival_date": "2024-02-10",
                "lead_time": 10,
                "is_canceled": 0,
                "reservation_status_date": "2024-02-12",
                "adr": 100.0,
                "market_segment": "Direct",
            },
            {
                "arrival_date": "2024-02-10",
                "lead_time": 8,
                "is_canceled": 0,
                "reservation_status_date": "2024-02-12",
                "adr": 110.0,
                "market_segment": "Online TA",
            },
        ]
    )
    df["arrival_date"] = pd.to_datetime(df["arrival_date"])
    df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"])

    mix = build_segment_mix_report(df, snapshot_date=pd.Timestamp("2024-02-05"), periods=7)
    assert {"date", "segment", "otb_bookings", "segment_share"} <= set(mix.columns)


def test_pricing_recommendation_raises_adr_when_demand_exceeds_capacity():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    rows = []
    for idx, date in enumerate(dates):
        demand = max(2, int(14 - 0.05 * idx))
        adr = 80 + idx * 0.6
        for _ in range(demand):
            rows.append(
                {
                    "arrival_date": date,
                    "lead_time": int(rng.integers(1, 20)),
                    "is_canceled": 0,
                    "reservation_status_date": date + pd.Timedelta(days=1),
                    "adr": adr,
                    "market_segment": "Direct",
                }
            )
    df = pd.DataFrame(rows)

    models = fit_price_elasticity_models(df)
    pickup = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-04-10")],
            "otb_today": [40],
            "otb_stly_today": [20],
            "pace_vs_stly": [1.0],
            "current_adr": [140.0],
            "pickup_7d": [10],
        }
    )
    forecast_df = pd.DataFrame(
        {
            "ds": [pd.Timestamp("2024-04-10")],
            "yhat": [130.0],
            "yhat_lower": [110.0],
            "yhat_upper": [145.0],
        }
    )
    pricing = recommend_dynamic_prices(forecast_df, pickup, models, capacity=100)
    assert pricing.iloc[0]["recommended_adr"] >= pricing.iloc[0]["current_adr"]
