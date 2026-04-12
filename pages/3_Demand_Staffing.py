"""
Page 3 — Demand & Staffing
30-day booking forecast + staffing recommendations + download.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import load_and_prepare_data, get_or_train_models
from src.forecasting import forecast_xgb, naive_baseline_forecast
from src.analytics import estimate_staffing
from src.time_series import expand_stay_nights, aggregate_stay_nights

st.set_page_config(page_title="Demand & Staffing", page_icon="👥", layout="wide")
st.title("👥 Demand & Staffing Planner")
st.caption(
    "Demand figures count bookings by **arrival date**. "
    "Staffing recommendations are estimates based on expected arrivals, not total room-nights occupied."
)

if st.session_state.get("defaulted_columns"):
    st.info(
        "Note: some columns were not in your uploaded dataset and use default assumptions: "
        + ", ".join(f"**{c.split(' —')[0]}**" for c in st.session_state["defaulted_columns"])
        + ". Results may not reflect real values."
    )

try:
    df, daily_df = load_and_prepare_data()
    models = get_or_train_models()
except Exception as e:
    st.error(f"Failed to load data or models: {e}")
    st.stop()

xgb_model, feat_cols = models["xgb_demand"]

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
guests_per_staff = st.sidebar.slider(
    "Guests per Staff Member", min_value=5, max_value=30, value=10, step=1
)

# ---------------------------------------------------------------------------
# Forecasts
# ---------------------------------------------------------------------------
demand_series = daily_df["daily_bookings"]
forecast_30   = forecast_xgb(xgb_model, feat_cols, demand_series, periods=30)
baseline_30   = naive_baseline_forecast(demand_series, periods=30)

staff_needed = forecast_30.apply(
    lambda b: estimate_staffing(b, guests_per_staff=guests_per_staff)
)

# ---------------------------------------------------------------------------
# Summary KPIs
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Avg Predicted Bookings/Day", f"{forecast_30.mean():.1f}")
col2.metric("Peak Predicted Bookings",    f"{forecast_30.max():.1f}")
col3.metric("Avg Staff Needed/Day",       f"{staff_needed.mean():.1f}")

st.divider()

# ---------------------------------------------------------------------------
# Chart — bookings + staff on dual axis
# ---------------------------------------------------------------------------
st.subheader("30-Day Booking Forecast & Staffing Requirements")

fig = go.Figure()
fig.add_trace(go.Bar(
    x=forecast_30.index, y=forecast_30.values,
    name="Predicted Bookings", marker_color="steelblue", opacity=0.8,
    yaxis="y1",
))
fig.add_trace(go.Scatter(
    x=baseline_30.index, y=baseline_30.values,
    name="Naive Baseline", line=dict(color="gray", dash="dot", width=1.5),
    yaxis="y1",
))
fig.add_trace(go.Scatter(
    x=staff_needed.index, y=staff_needed.values,
    name="Staff Needed", line=dict(color="tomato", width=2.5),
    yaxis="y2", mode="lines+markers", marker=dict(size=6),
))
fig.update_layout(
    hovermode="x unified",
    xaxis_title="Date",
    yaxis=dict(title="Bookings", side="left"),
    yaxis2=dict(title="Staff Needed", side="right", overlaying="y", showgrid=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=430,
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Table + download
# ---------------------------------------------------------------------------
st.subheader("Day-by-Day Forecast")
forecast_df = pd.DataFrame({
    "Date":               forecast_30.index.date,
    "Predicted Bookings": forecast_30.values.round(1),
    "Naive Baseline":     baseline_30.values.round(1),
    "Staff Needed":       staff_needed.values,
})
st.dataframe(forecast_df, use_container_width=True)

csv_data = forecast_df.to_csv(index=False)
st.download_button(
    label="Download Staffing Plan (CSV)",
    data=csv_data,
    file_name="staffing_forecast.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------------
# Insight
# ---------------------------------------------------------------------------
peak_day   = forecast_30.idxmax().date()
peak_staff = int(staff_needed.max())
st.info(
    f"Peak demand is expected on **{peak_day}** — plan for at least **{peak_staff} staff members** that day."
)

with st.expander("Model Info"):
    m = models["eval_metrics"].get("demand_xgb", {})
    st.markdown(
        f"**Model:** XGBoost (lag + rolling + calendar features)  \n"
        f"**MAE (held-out 20%):** {m.get('mae', 'N/A'):,.1f} bookings  \n"
        f"**MAPE:** {m.get('mape', 'N/A'):.1f}%"
    )

# ---------------------------------------------------------------------------
# Stay-night occupancy view
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Occupied Rooms by Stay Night")
st.caption(
    "Each non-cancelled booking is expanded across all its nights of stay. "
    "Occupied rooms = actual room-nights in use, not just arrival counts."
)

try:
    stay_df = expand_stay_nights(df)
    occ_df = aggregate_stay_nights(stay_df)
    occ_hist = occ_df.iloc[-90:] if len(occ_df) >= 90 else occ_df

    occ_col1, occ_col2 = st.columns(2)
    occ_col1.metric("Avg Occupied Rooms/Night (hist.)", f"{occ_hist['occupied_rooms'].mean():.1f}")
    occ_col2.metric("Peak Occupied Rooms", f"{int(occ_hist['occupied_rooms'].max())}")

    fig_occ = go.Figure()
    fig_occ.add_trace(go.Bar(
        x=occ_hist.index,
        y=occ_hist["occupied_rooms"].values,
        name="Occupied Rooms",
        marker_color="mediumseagreen",
        opacity=0.8,
    ))
    fig_occ.update_layout(
        xaxis_title="Date",
        yaxis_title="Occupied Rooms",
        height=380,
        hovermode="x unified",
    )
    st.plotly_chart(fig_occ, use_container_width=True)

    st.markdown("**Staffing based on occupied rooms (historical)**")
    occ_staff = occ_hist["occupied_rooms"].apply(
        lambda r: estimate_staffing(r, guests_per_staff=guests_per_staff)
    )
    st.dataframe(
        pd.DataFrame({
            "Date": occ_hist.index.date,
            "Occupied Rooms": occ_hist["occupied_rooms"].values.astype(int),
            "Staff Needed (occupied)": occ_staff.values,
        }),
        use_container_width=True,
    )
except Exception as exc:
    st.warning(f"Could not compute stay-night occupancy: {exc}")
