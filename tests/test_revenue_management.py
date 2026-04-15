import numpy as np
import pandas as pd

from src.revenue_management import (
    build_otb_feature_matrix,
    build_pickup_report,
    build_segment_mix_report,
    fit_price_elasticity_models,
    pickup_report_to_otb_features,
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


# ---------------------------------------------------------------------------
# build_otb_feature_matrix tests
# ---------------------------------------------------------------------------

def _make_otb_df(arrival="2024-03-01", booking_days_before=30, cancel_date=None):
    """Single-booking DataFrame for OTB unit tests."""
    arrival_ts = pd.Timestamp(arrival)
    booking_date = arrival_ts - pd.Timedelta(days=booking_days_before)
    row = {
        "arrival_date": arrival_ts,
        "lead_time": booking_days_before,
        "is_canceled": 1 if cancel_date else 0,
        "reservation_status_date": pd.Timestamp(cancel_date) if cancel_date else arrival_ts,
        "adr": 100.0,
        "market_segment": "Direct",
    }
    return pd.DataFrame([row])


def test_build_otb_matrix_cancellation_after_snapshot_counts():
    """Booking cancelled AFTER the snapshot date still counts as OTB at that snapshot."""
    # Arrival 2024-03-01, booked 30 days before, cancelled on 2024-02-20
    # snapshot for k=7 = arrival - 7 = 2024-02-23; cancel_date 2024-02-20 < snapshot → NOT OTB
    # snapshot for k=14 = 2024-02-16; cancel_date 2024-02-20 > snapshot → IS OTB
    df = _make_otb_df(arrival="2024-03-01", booking_days_before=30,
                      cancel_date="2024-02-20")
    result = build_otb_feature_matrix(df, lead_times=[7, 14])
    arrival = pd.Timestamp("2024-03-01")
    assert result.loc[arrival, "otb_14d"] == 1   # booked & not-yet-cancelled at d-14
    assert result.loc[arrival, "otb_7d"] == 0    # already cancelled at d-7


def test_build_otb_matrix_cancellation_before_snapshot_excluded():
    """Booking cancelled BEFORE the snapshot does NOT count."""
    # Cancel 2024-01-15, k=30 → snapshot = 2024-02-01; cancel_date < snapshot → 0
    df = _make_otb_df(arrival="2024-03-01", booking_days_before=60,
                      cancel_date="2024-01-15")
    result = build_otb_feature_matrix(df, lead_times=[30])
    assert result.loc[pd.Timestamp("2024-03-01"), "otb_30d"] == 0


def test_build_otb_matrix_booking_exactly_on_snapshot():
    """Booking placed exactly on (d - k) is boundary-inclusive and must count."""
    # booking_days_before = k=7, so booking_date = arrival - 7 = snapshot exactly
    df = _make_otb_df(arrival="2024-03-01", booking_days_before=7)
    result = build_otb_feature_matrix(df, lead_times=[7])
    assert result.loc[pd.Timestamp("2024-03-01"), "otb_7d"] == 1


def test_build_otb_matrix_booking_after_snapshot_excluded():
    """Booking placed 3 days before arrival does NOT appear in otb_7d (snapshot = d-7)."""
    df = _make_otb_df(arrival="2024-03-01", booking_days_before=3)
    result = build_otb_feature_matrix(df, lead_times=[7])
    assert result.loc[pd.Timestamp("2024-03-01"), "otb_7d"] == 0


def test_build_otb_matrix_empty_df():
    """Returns empty DataFrame with correct columns for empty input."""
    df = pd.DataFrame(columns=["arrival_date", "lead_time", "is_canceled",
                                "reservation_status_date", "adr"])
    result = build_otb_feature_matrix(df, lead_times=[7, 14, 30, 60])
    assert set(result.columns) == {"otb_7d", "otb_14d", "otb_30d", "otb_60d"}
    assert len(result) == 0


def test_build_otb_matrix_index_no_gaps():
    """Index must have no date gaps (matches aggregate_daily() contract)."""
    rows = []
    for d in [1, 5, 10]:  # arrivals on days 1, 5, 10 — gaps in between
        rows.append({
            "arrival_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=d - 1),
            "lead_time": 30,
            "is_canceled": 0,
            "reservation_status_date": pd.Timestamp("2024-01-31"),
            "adr": 100.0,
            "market_segment": "Direct",
        })
    df = pd.DataFrame(rows)
    result = build_otb_feature_matrix(df, lead_times=[7])
    expected_len = (result.index.max() - result.index.min()).days + 1
    assert len(result) == expected_len


def test_build_otb_matrix_no_future_leakage():
    """
    Leakage guard: a booking placed AFTER the snapshot must not appear in the
    OTB count for that snapshot.

    Concretely: booking placed on 2024-02-25 (5 days before arrival 2024-03-01).
    For k=7, snapshot = 2024-02-23 — booking_date (2024-02-25) > snapshot → 0.
    For k=3, snapshot = 2024-02-27 — booking_date (2024-02-25) <= snapshot → 1.
    """
    df = _make_otb_df(arrival="2024-03-01", booking_days_before=5)
    result = build_otb_feature_matrix(df, lead_times=[3, 7])
    arrival = pd.Timestamp("2024-03-01")
    assert result.loc[arrival, "otb_7d"] == 0   # booking after snapshot → leakage would show 1
    assert result.loc[arrival, "otb_3d"] == 1   # booking before this snapshot → correct


# ---------------------------------------------------------------------------
# pickup_report_to_otb_features tests
# ---------------------------------------------------------------------------

def _make_pickup_report():
    dates = pd.date_range("2024-03-05", periods=3, freq="D")
    return pd.DataFrame({
        "date": dates,
        "otb_today": [10, 12, 8],
        "otb_7d_ago": [6, 7, 5],
        "otb_14d_ago": [3, 4, 2],
        "otb_30d_ago": [1, 2, 0],
    })


def test_pickup_report_to_otb_features_columns():
    report = _make_pickup_report()
    result = pickup_report_to_otb_features(report)
    assert set(result.columns) == {"otb_7d", "otb_14d", "otb_30d", "otb_60d"}


def test_pickup_report_to_otb_features_no_nan():
    report = _make_pickup_report()
    result = pickup_report_to_otb_features(report)
    assert result.isna().sum().sum() == 0


def test_pickup_report_to_otb_features_index_is_dates():
    report = _make_pickup_report()
    result = pickup_report_to_otb_features(report)
    assert pd.api.types.is_datetime64_any_dtype(result.index)
    assert len(result) == len(report)


def test_pickup_report_to_otb_features_uses_30d_ago_for_longer_leads():
    """otb_30d and otb_60d should use otb_30d_ago when available."""
    report = _make_pickup_report()
    result = pickup_report_to_otb_features(report)
    assert list(result["otb_30d"]) == list(report["otb_30d_ago"])
    assert list(result["otb_60d"]) == list(report["otb_30d_ago"])
