"""
Actionable alerts system.

Generates prioritised alerts for hotel operators based on cancellation risk,
demand forecasts, and occupancy levels. Alerts are sorted by severity so
critical issues surface first.
"""
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any
import pandas as pd
import numpy as np

from src.booking_risk_scorer import score_bookings

_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


@dataclass
class Alert:
    type: str
    severity: str          # "info" | "warning" | "critical"
    title: str
    description: str
    recommended_action: str
    affected_date: date
    data: dict = field(default_factory=dict)


def generate_daily_alerts(
    bookings_df: pd.DataFrame,
    forecast_series: pd.Series,
    cancellation_model,
    config: dict,
) -> list[Alert]:
    """
    Generate actionable alerts sorted by severity (critical first).

    Parameters
    ----------
    bookings_df : full bookings DataFrame (same schema as hotel_bookings.csv)
    forecast_series : pd.Series with DatetimeIndex, values = predicted daily demand
    cancellation_model : fitted classifier with predict_proba
    config : optional thresholds dict, all keys optional:
        cancel_risk_threshold (float, default 0.7)
        demand_spike_percentile (int, default 90)
        revenue_dip_percentile (int, default 25)
        occupancy_overbook_threshold (float, default 0.9)
        low_occupancy_threshold (float, default 0.6)
        total_rooms (int, default 100)

    Returns
    -------
    list[Alert] sorted critical → warning → info
    """
    alerts: list[Alert] = []
    today = datetime.today().date()
    cfg = {
        "cancel_risk_threshold": 0.7,
        "demand_spike_percentile": 90,
        "revenue_dip_percentile": 25,
        "occupancy_overbook_threshold": 0.9,
        "low_occupancy_threshold": 0.6,
        "total_rooms": 100,
        **config,
    }

    # ------------------------------------------------------------------
    # 1. High cancellation risk
    # ------------------------------------------------------------------
    try:
        arriving_soon = bookings_df[
            (bookings_df["is_canceled"] == 0) &
            (bookings_df["arrival_date"].dt.date <= today + timedelta(days=7))
        ]
        if len(arriving_soon) > 0:
            scored = score_bookings(arriving_soon, cancellation_model)
            high_risk = scored[scored["cancel_probability"] > cfg["cancel_risk_threshold"]]
            if len(high_risk) > 0:
                alerts.append(Alert(
                    type="high_cancellation_risk",
                    severity="critical",
                    title=f"{len(high_risk)} High-Risk Booking(s) Arriving Within 7 Days",
                    description=(
                        f"{len(high_risk)} booking(s) have >{cfg['cancel_risk_threshold']:.0%} "
                        f"cancellation probability and arrive within the next 7 days."
                    ),
                    recommended_action=(
                        "Review these bookings and consider sending retention offers "
                        "or requesting deposit confirmation."
                    ),
                    affected_date=today,
                    data={
                        "count": len(high_risk),
                        "avg_probability": round(float(high_risk["cancel_probability"].mean()), 3),
                    },
                ))
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 2. Demand spike — next 14 days exceeds historical 90th percentile
    # ------------------------------------------------------------------
    try:
        threshold_90 = float(forecast_series.quantile(cfg["demand_spike_percentile"] / 100))
        next_14_end = pd.Timestamp(today + timedelta(days=14))
        next_14 = forecast_series[forecast_series.index <= next_14_end]
        spike_days = next_14[next_14 > threshold_90]
        if len(spike_days) > 0:
            alerts.append(Alert(
                type="demand_spike",
                severity="warning",
                title=f"Demand Spike: {len(spike_days)} Day(s) Above 90th Percentile",
                description=(
                    f"Forecasted demand exceeds the historical 90th percentile "
                    f"({threshold_90:.0f} bookings/day) on {len(spike_days)} upcoming day(s)."
                ),
                recommended_action=(
                    "Consider dynamic pricing increases and ensure adequate staffing."
                ),
                affected_date=spike_days.index[0].date(),
                data={"spike_days": len(spike_days), "threshold": round(threshold_90, 1)},
            ))
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 3. Revenue dip — next 7 days below historical 25th percentile
    # ------------------------------------------------------------------
    try:
        threshold_25 = float(forecast_series.quantile(cfg["revenue_dip_percentile"] / 100))
        next_7_end = pd.Timestamp(today + timedelta(days=7))
        next_7 = forecast_series[forecast_series.index <= next_7_end]
        dip_days = next_7[next_7 < threshold_25]
        if len(dip_days) > 0:
            alerts.append(Alert(
                type="revenue_dip",
                severity="warning",
                title=f"Revenue Dip: {len(dip_days)} Low-Demand Day(s) This Week",
                description=(
                    f"Forecasted demand is below the 25th percentile on "
                    f"{len(dip_days)} day(s) in the next 7 days."
                ),
                recommended_action=(
                    "Consider promotional pricing or targeted marketing to drive bookings."
                ),
                affected_date=dip_days.index[0].date(),
                data={"dip_days": len(dip_days), "threshold": round(threshold_25, 1)},
            ))
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 4. Low occupancy warning — next 30 days below 60% capacity
    # ------------------------------------------------------------------
    try:
        capacity = cfg["total_rooms"]
        low_threshold = capacity * cfg["low_occupancy_threshold"]
        next_30_end = pd.Timestamp(today + timedelta(days=30))
        next_30 = forecast_series[forecast_series.index <= next_30_end]
        low_days = next_30[next_30 < low_threshold]
        if len(low_days) > 0:
            alerts.append(Alert(
                type="low_occupancy",
                severity="info",
                title=f"Low Occupancy: {len(low_days)} Day(s) Below {cfg['low_occupancy_threshold']:.0%} Capacity",
                description=(
                    f"{len(low_days)} day(s) in the next 30 days are forecast below "
                    f"{cfg['low_occupancy_threshold']:.0%} occupancy "
                    f"({low_threshold:.0f} of {capacity} rooms)."
                ),
                recommended_action=(
                    "Run promotions or adjust rates to stimulate demand for these dates."
                ),
                affected_date=low_days.index[0].date(),
                data={"low_days": len(low_days), "capacity": capacity},
            ))
    except Exception:
        pass

    return sorted(alerts, key=lambda a: _SEVERITY_ORDER.get(a.severity, 99))
