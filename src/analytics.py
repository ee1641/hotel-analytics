"""
Staffing recommendations and profit/tax estimation.
"""
import math
import pandas as pd
from src.services import APP_CONFIG


def estimate_staffing(
    predicted_bookings: float,
    guests_per_staff: int = APP_CONFIG["guests_per_staff_default"],
) -> int:
    """
    Return number of staff needed: ceil(bookings / guests_per_staff).
    Uses ceiling so we never under-staff.
    """
    return math.ceil(predicted_bookings / guests_per_staff)


def estimate_profit(
    revenue,
    tax_rate: float = APP_CONFIG["tax_rate"],
    cost_ratio: float = APP_CONFIG["cost_ratio"],
) -> dict:
    """
    Compute tax, operational cost, and net profit from revenue.

    Args:
        revenue    : scalar or pd.Series
        tax_rate   : fraction of revenue paid as tax       (default from APP_CONFIG)
        cost_ratio : fraction of revenue as operating cost (default from APP_CONFIG)

    Returns:
        dict with keys: tax, operational_cost, profit
    """
    tax = revenue * tax_rate
    operational_cost = revenue * cost_ratio
    profit = revenue - operational_cost - tax
    return {"tax": tax, "operational_cost": operational_cost, "profit": profit}
