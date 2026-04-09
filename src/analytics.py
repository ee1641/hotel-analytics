"""
Staffing recommendations and profit/tax estimation.
"""
import math
import pandas as pd


def estimate_staffing(predicted_bookings: float, guests_per_staff: int = 10) -> int:
    """
    Return number of staff needed: ceil(bookings / guests_per_staff).
    Uses ceiling so we never under-staff.
    """
    return math.ceil(predicted_bookings / guests_per_staff)


def estimate_profit(revenue, tax_rate: float = 0.10, cost_ratio: float = 0.60) -> dict:
    """
    Compute tax, operational cost, and net profit from revenue.

    Args:
        revenue    : scalar or pd.Series
        tax_rate   : fraction of revenue paid as tax       (default 10%)
        cost_ratio : fraction of revenue as operating cost (default 60%)

    Returns:
        dict with keys: tax, operational_cost, profit
    """
    tax = revenue * tax_rate
    operational_cost = revenue * cost_ratio
    profit = revenue - operational_cost - tax
    return {"tax": tax, "operational_cost": operational_cost, "profit": profit}
