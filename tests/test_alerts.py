"""Tests for src/alerts.py and src/email_alerts.py"""
import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

from src.alerts import Alert, generate_daily_alerts, _SEVERITY_ORDER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bookings(n: int = 20) -> pd.DataFrame:
    today = pd.Timestamp.today().normalize()
    return pd.DataFrame({
        "arrival_date": [today + pd.Timedelta(days=i % 5) for i in range(n)],
        "is_canceled":  [0] * n,
        "adr":          [100.0] * n,
        "lead_time":    [10] * n,
        "adults":       [2] * n,
    })


def _make_forecast(periods: int = 60, value: float = 50.0) -> pd.Series:
    idx = pd.date_range(start=pd.Timestamp.today(), periods=periods, freq="D")
    return pd.Series([value] * periods, index=idx)


def _mock_model():
    m = MagicMock()
    m.predict_proba = MagicMock(
        side_effect=lambda X: np.column_stack([
            np.full(len(X), 0.2),
            np.full(len(X), 0.8),  # high cancel proba to trigger critical alert
        ])
    )
    return m


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------

def test_alert_default_data_is_empty_dict():
    a = Alert(
        type="test", severity="info", title="T",
        description="D", recommended_action="A", affected_date=date.today(),
    )
    assert a.data == {}


def test_alert_stores_data():
    a = Alert(
        type="t", severity="critical", title="T",
        description="D", recommended_action="A",
        affected_date=date.today(), data={"count": 5},
    )
    assert a.data["count"] == 5


# ---------------------------------------------------------------------------
# Severity ordering
# ---------------------------------------------------------------------------

def test_severity_order_critical_before_warning():
    assert _SEVERITY_ORDER["critical"] < _SEVERITY_ORDER["warning"]


def test_severity_order_warning_before_info():
    assert _SEVERITY_ORDER["warning"] < _SEVERITY_ORDER["info"]


def test_alerts_sorted_by_severity():
    df = _make_bookings()
    forecast = _make_forecast()
    with patch("src.alerts.score_bookings") as mock_score:
        mock_score.return_value = pd.DataFrame({"cancel_probability": [0.9] * 5})
        alerts = generate_daily_alerts(df, forecast, _mock_model(), {})
    if len(alerts) > 1:
        orders = [_SEVERITY_ORDER.get(a.severity, 99) for a in alerts]
        assert orders == sorted(orders)


# ---------------------------------------------------------------------------
# Alert trigger conditions
# ---------------------------------------------------------------------------

def test_low_occupancy_alert_triggers_when_forecast_low():
    df = _make_bookings()
    forecast = _make_forecast(value=5.0)  # far below 60% of 100 rooms
    with patch("src.alerts.score_bookings") as mock_score:
        mock_score.return_value = pd.DataFrame({"cancel_probability": [0.1] * 5})
        alerts = generate_daily_alerts(df, forecast, MagicMock(), {"total_rooms": 100})
    types = [a.type for a in alerts]
    assert "low_occupancy" in types


def test_demand_spike_alert_triggers():
    # Spike on day 5-7 (within the 14-day window), baseline low
    idx = pd.date_range(start=pd.Timestamp.today(), periods=60, freq="D")
    vals = [10.0] * 60
    vals[5] = 1000.0
    vals[6] = 1000.0
    vals[7] = 1000.0
    forecast = pd.Series(vals, index=idx)
    df = _make_bookings()
    with patch("src.alerts.score_bookings") as mock_score:
        mock_score.return_value = pd.DataFrame({"cancel_probability": [0.1] * 5})
        alerts = generate_daily_alerts(df, forecast, MagicMock(), {})
    types = [a.type for a in alerts]
    assert "demand_spike" in types


def test_no_critical_when_all_low_risk():
    df = _make_bookings()
    forecast = _make_forecast(value=70.0)
    model = MagicMock()
    model.predict_proba = MagicMock(
        side_effect=lambda X: np.column_stack([
            np.full(len(X), 0.95),
            np.full(len(X), 0.05),  # very low cancel prob
        ])
    )
    with patch("src.alerts.score_bookings") as mock_score:
        mock_score.return_value = pd.DataFrame({"cancel_probability": [0.05] * len(df)})
        alerts = generate_daily_alerts(df, forecast, model, {
            "total_rooms": 100,
            "low_occupancy_threshold": 0.3,  # 30% threshold → 70 bookings won't trigger
        })
    critical = [a for a in alerts if a.severity == "critical"]
    assert len(critical) == 0


# ---------------------------------------------------------------------------
# Email template
# ---------------------------------------------------------------------------

def test_format_alert_email_returns_html():
    from src.email_alerts import format_alert_email
    alerts = [
        Alert(type="t", severity="critical", title="Test Alert",
              description="Desc", recommended_action="Act", affected_date=date.today()),
    ]
    html = format_alert_email(alerts)
    assert "<html" in html
    assert "Test Alert" in html
    assert "critical" in html.lower()


def test_format_alert_email_empty_list():
    from src.email_alerts import format_alert_email
    html = format_alert_email([])
    assert "<html" in html
    assert "No alerts" in html


def test_format_alert_email_all_severities():
    from src.email_alerts import format_alert_email
    alerts = [
        Alert(type="a", severity="critical", title="C", description="D", recommended_action="A", affected_date=date.today()),
        Alert(type="b", severity="warning",  title="W", description="D", recommended_action="A", affected_date=date.today()),
        Alert(type="c", severity="info",     title="I", description="D", recommended_action="A", affected_date=date.today()),
    ]
    html = format_alert_email(alerts)
    assert "critical" in html.lower()
    assert "warning"  in html.lower()
    assert "info"     in html.lower()
