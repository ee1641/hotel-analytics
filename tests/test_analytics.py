import pandas as pd
import pytest
from src.analytics import estimate_staffing, estimate_profit


def test_staffing_exact():
    assert estimate_staffing(predicted_bookings=100, guests_per_staff=10) == 10


def test_staffing_ceil():
    assert estimate_staffing(predicted_bookings=101, guests_per_staff=10) == 11


def test_staffing_fractional_input():
    # Forecast values are floats
    assert estimate_staffing(predicted_bookings=95.7, guests_per_staff=10) == 10


def test_profit_keys():
    result = estimate_profit(1000.0, tax_rate=0.10, cost_ratio=0.60)
    assert {"tax", "operational_cost", "profit"} == set(result.keys())


def test_profit_scalar_values():
    result = estimate_profit(1000.0, tax_rate=0.10, cost_ratio=0.60)
    assert abs(result["tax"] - 100.0) < 1e-6
    assert abs(result["operational_cost"] - 600.0) < 1e-6
    assert abs(result["profit"] - 300.0) < 1e-6


def test_profit_series_input():
    revenue = pd.Series([1000.0, 2000.0])
    result = estimate_profit(revenue, tax_rate=0.10, cost_ratio=0.60)
    assert len(result["profit"]) == 2
    assert abs(result["profit"].iloc[0] - 300.0) < 1e-6
    assert abs(result["profit"].iloc[1] - 600.0) < 1e-6
