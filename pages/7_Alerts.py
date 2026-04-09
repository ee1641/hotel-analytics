"""
Page 7 — Actionable Alerts
Severity-sorted alert cards with dismiss, filter, and config controls.
"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import load_and_prepare_data, get_or_train_models
from src.alerts import generate_daily_alerts
from src.forecasting import forecast_xgb

st.set_page_config(page_title="Alerts", page_icon="🔔", layout="wide")
st.title("🔔 Actionable Alerts")
st.markdown("Real-time alerts generated from your cancellation model and demand forecasts.")

try:
    with st.spinner("Loading models..."):
        df, daily_df = load_and_prepare_data()
        models = get_or_train_models()
except Exception as e:
    st.error(f"Failed to load data or models: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar config
# ---------------------------------------------------------------------------
st.sidebar.header("Alert Configuration")
total_rooms = st.sidebar.number_input(
    "Hotel Capacity (rooms)", min_value=10, max_value=5000, value=100
)
cancel_threshold = st.sidebar.slider(
    "Cancellation Risk Threshold", min_value=0.50, max_value=0.95,
    value=0.70, step=0.05,
)
low_occ_threshold = st.sidebar.slider(
    "Low Occupancy Threshold", min_value=0.30, max_value=0.80,
    value=0.60, step=0.05,
)

config = {
    "cancel_risk_threshold": cancel_threshold,
    "low_occupancy_threshold": low_occ_threshold,
    "total_rooms": total_rooms,
}

# ---------------------------------------------------------------------------
# Generate alerts
# ---------------------------------------------------------------------------
cancel_model = models["cancel_models"]["xgboost"]
xgb_model, feat_cols = models["xgb_demand"]
demand_forecast = forecast_xgb(xgb_model, feat_cols, daily_df["daily_bookings"], periods=30)

try:
    alerts = generate_daily_alerts(df, demand_forecast, cancel_model, config)
except Exception as e:
    st.error(f"Could not generate alerts: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Dismiss state
# ---------------------------------------------------------------------------
if "dismissed_alert_ids" not in st.session_state:
    st.session_state["dismissed_alert_ids"] = set()

active_alerts = [
    a for i, a in enumerate(alerts)
    if i not in st.session_state["dismissed_alert_ids"]
]

# ---------------------------------------------------------------------------
# Filter controls
# ---------------------------------------------------------------------------
filter_col1, filter_col2 = st.columns(2)
with filter_col1:
    severity_filter = st.multiselect(
        "Filter by severity",
        options=["critical", "warning", "info"],
        default=["critical", "warning", "info"],
    )
with filter_col2:
    all_types = sorted({a.type for a in active_alerts}) if active_alerts else []
    type_filter = st.multiselect(
        "Filter by type",
        options=all_types,
        default=all_types,
    )

filtered = [
    (i, a) for i, a in enumerate(alerts)
    if a.severity in severity_filter
    and a.type in type_filter
    and i not in st.session_state["dismissed_alert_ids"]
]

# ---------------------------------------------------------------------------
# Render alert cards
# ---------------------------------------------------------------------------
_COLORS = {
    "critical": "#dc3545",
    "warning":  "#ffc107",
    "info":     "#0dcaf0",
}
_BG = {
    "critical": "#fff5f5",
    "warning":  "#fffbf0",
    "info":     "#f0faff",
}

if not filtered:
    st.success("No active alerts matching your filters. Everything looks good!")
else:
    st.subheader(f"{len(filtered)} Active Alert(s)")
    for orig_idx, alert in filtered:
        color = _COLORS.get(alert.severity, "#6c757d")
        bg    = _BG.get(alert.severity, "#fafafa")
        card_col, btn_col = st.columns([11, 1])
        with card_col:
            st.markdown(
                f"""<div style="border-left:4px solid {color};padding:14px 18px;
                    margin-bottom:10px;background:{bg};border-radius:0 6px 6px 0;">
                  <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                    <span style="background:{color};color:white;padding:2px 10px;border-radius:4px;
                                 font-size:11px;font-weight:bold;text-transform:uppercase;">
                      {alert.severity}
                    </span>
                    <strong style="font-size:15px;">{alert.title}</strong>
                  </div>
                  <p style="margin:0 0 6px;color:#555;font-size:13px;">{alert.description}</p>
                  <p style="margin:0;color:#1a6c3c;font-size:13px;">
                    ✅ <em>{alert.recommended_action}</em>
                  </p>
                </div>""",
                unsafe_allow_html=True,
            )
        with btn_col:
            if st.button("✕", key=f"dismiss_{orig_idx}", help="Dismiss this alert"):
                st.session_state["dismissed_alert_ids"].add(orig_idx)
                st.rerun()

# ---------------------------------------------------------------------------
# Email preview expander
# ---------------------------------------------------------------------------
with st.expander("Preview Email Summary"):
    from src.email_alerts import format_alert_email
    html = format_alert_email([a for _, a in filtered])
    st.components.v1.html(html, height=400, scrolling=True)
