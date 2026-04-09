"""
Overbooking recommendation engine.

Uses the Newsvendor critical ratio to find the optimal number of rooms
to overbook for a given arrival date, balancing the cost of walking a
guest against the cost of an empty room from a cancellation.
"""
import numpy as np
import pandas as pd
from datetime import date
from scipy import stats
from src.booking_risk_scorer import score_bookings


def calculate_overbooking_recommendation(
    bookings_df: pd.DataFrame,
    cancellation_model,
    target_date: date,
    total_rooms: int,
    overbooking_cost: float = 150.0,
    empty_room_cost: float = 100.0,
    confidence_level: float = 0.80,
) -> dict:
    """
    Recommend how many rooms to overbook for target_date.

    Uses the Newsvendor critical ratio to find the optimal overbooking count:
      threshold = empty_room_cost / (empty_room_cost + overbooking_cost)
    Then uses a Poisson distribution of expected cancellations to find the
    overbooking level where P(cancellations >= overbook_count) = threshold.

    Parameters
    ----------
    bookings_df : DataFrame with arrival_date, is_canceled, adr columns
    cancellation_model : fitted sklearn-compatible classifier with predict_proba
    target_date : the arrival date to analyse
    total_rooms : hotel capacity
    overbooking_cost : cost of walking a guest (relocation + compensation)
    empty_room_cost : lost revenue from an empty room
    confidence_level : width of the confidence interval (0–1)

    Returns
    -------
    dict with keys:
        expected_cancellations (float)
        recommended_overbooks (int)
        confidence_interval (tuple[float, float])
        risk_level ("Low" | "Medium" | "High")
        expected_revenue_impact (float)
    """
    # Filter to target_date non-cancelled bookings
    day_bookings = bookings_df[
        (bookings_df["arrival_date"].dt.date == target_date) &
        (bookings_df["is_canceled"] == 0)
    ]
    if len(day_bookings) == 0:
        return {
            "expected_cancellations": 0.0,
            "recommended_overbooks": 0,
            "confidence_interval": (0.0, 0.0),
            "risk_level": "Low",
            "expected_revenue_impact": 0.0,
        }

    # Get cancel probabilities for each booking
    scored = score_bookings(day_bookings, cancellation_model)
    probs = scored["cancel_probability"].values

    # Expected cancellations (mean of Poisson = sum of individual probabilities)
    lam = float(probs.sum())

    if lam == 0.0:
        return {
            "expected_cancellations": 0.0,
            "recommended_overbooks": 0,
            "confidence_interval": (0.0, 0.0),
            "risk_level": "Low",
            "expected_revenue_impact": 0.0,
        }

    # Critical ratio threshold (Newsvendor model)
    threshold = empty_room_cost / (empty_room_cost + overbooking_cost)

    # Find recommended overbooks: smallest k where P(Poisson(lam) >= k) <= threshold
    recommended = 0
    for k in range(1, max(int(lam * 4) + 1, 10)):
        if 1 - stats.poisson.cdf(k - 1, lam) <= threshold:
            recommended = k
            break

    # Confidence interval at confidence_level
    alpha = (1 - confidence_level) / 2
    lower = float(stats.poisson.ppf(alpha, lam))
    upper = float(stats.poisson.ppf(1 - alpha, lam))

    # Risk level
    if recommended == 0:
        risk_level = "Low"
    elif recommended <= 2:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # Revenue impact: expected filled rooms * avg_adr minus expected walk costs
    avg_adr = float(day_bookings["adr"].mean()) if "adr" in day_bookings.columns else empty_room_cost
    revenue_impact = (recommended * avg_adr) - (lam * overbooking_cost)

    return {
        "expected_cancellations": round(lam, 2),
        "recommended_overbooks": recommended,
        "confidence_interval": (lower, upper),
        "risk_level": risk_level,
        "expected_revenue_impact": round(revenue_impact, 2),
    }
