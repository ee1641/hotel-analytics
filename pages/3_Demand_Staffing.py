"""
Page 3 - Demand & Staffing
Demand forecast, staffing requirements, and segment-level breakdown.
"""
import warnings

warnings.filterwarnings("ignore")

import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import get_or_train_models, load_and_prepare_data
from src.analytics import estimate_staffing
from src.config import APP_CONFIG
from src.conformal import forecast_conformal
from src.forecasting import forecast_xgb, forecast_xgb_with_intervals, naive_baseline_forecast
from src.model_specs import FORECAST_LAGS
from src.revenue_management import (
    build_segment_mix_report,
    forecast_segment_demand,
    train_segment_demand_models,
)
from src.time_series import aggregate_stay_nights, expand_stay_nights

st.set_page_config(page_title="Demand & Staffing", page_icon="👥", layout="wide")
st.title("👥 Demand & Staffing Planner")
st.caption(
    "Demand counts bookings by arrival date. The staffing plan is arrival-driven, while the segment view shows where that demand is coming from."
)

if st.session_state.get("defaulted_columns"):
    st.info(
        "Note: some columns were not in your uploaded dataset and use default assumptions: "
        + ", ".join(f"**{c.split(' —')[0]}**" for c in st.session_state["defaulted_columns"])
        + ". Results may not reflect real values."
    )

with st.spinner("Training models… this may take a few minutes on first run."):
    try:
        df, daily_df = load_and_prepare_data()
        models = get_or_train_models()
    except Exception as e:
        st.error(f"Failed to load data or models: {e}")
        st.stop()

daily_df = models.get("daily_revenue_views", daily_df)
xgb_model, feat_cols = models["xgb_demand"]

st.sidebar.header("Planner Controls")
guests_per_staff = st.sidebar.slider(
    "Guests per Staff Member",
    min_value=5,
    max_value=30,
    value=APP_CONFIG["guests_per_staff_default"],
    step=1,
)
total_rooms = st.sidebar.number_input(
    "Room Capacity",
    min_value=10,
    max_value=5000,
    value=APP_CONFIG.get("default_total_rooms", 100),
)

demand_series = daily_df["daily_bookings"]
conformal_bundle = models.get("xgb_demand_conformal")
interval_models = models.get("xgb_demand_intervals")
if conformal_bundle is not None:
    demand_band_df = forecast_conformal(conformal_bundle, demand_series, periods=30)
    forecast_30 = demand_band_df.set_index("ds")["yhat"]
elif interval_models:
    demand_band_df = forecast_xgb_with_intervals(interval_models, demand_series, periods=30)
    forecast_30 = demand_band_df.set_index("ds")["yhat"]
else:
    demand_band_df = None
    forecast_30 = forecast_xgb(xgb_model, feat_cols, demand_series, periods=30)
baseline_30 = naive_baseline_forecast(demand_series, periods=30)

staff_needed = forecast_30.apply(lambda b: estimate_staffing(b, guests_per_staff=guests_per_staff))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Predicted Bookings/Day", f"{forecast_30.mean():.1f}")
col2.metric("Peak Predicted Bookings", f"{forecast_30.max():.1f}")
col3.metric("Avg Staff Needed/Day", f"{staff_needed.mean():.1f}")
col4.metric("Peak Occupancy vs Capacity", f"{forecast_30.max() / max(total_rooms, 1):.1%}")

st.divider()
st.subheader("30-Day Booking Forecast & Staffing Requirements")

fig = go.Figure()
if demand_band_df is not None:
    fig.add_trace(
        go.Scatter(
            x=demand_band_df["ds"].tolist() + demand_band_df["ds"].tolist()[::-1],
            y=demand_band_df["yhat_upper"].tolist() + demand_band_df["yhat_lower"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(70,130,180,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Forecast Range",
        )
    )
fig.add_trace(
    go.Bar(
        x=forecast_30.index,
        y=forecast_30.values,
        name="Predicted Bookings",
        marker_color="steelblue",
        opacity=0.8,
        yaxis="y1",
    )
)
fig.add_trace(
    go.Scatter(
        x=baseline_30.index,
        y=baseline_30.values,
        name="Naive Baseline",
        line=dict(color="gray", dash="dot", width=1.5),
        yaxis="y1",
    )
)
fig.add_trace(
    go.Scatter(
        x=staff_needed.index,
        y=staff_needed.values,
        name="Staff Needed",
        line=dict(color="tomato", width=2.5),
        yaxis="y2",
        mode="lines+markers",
        marker=dict(size=6),
    )
)
fig.update_layout(
    hovermode="x unified",
    xaxis_title="Date",
    yaxis=dict(title="Bookings", side="left"),
    yaxis2=dict(title="Staff Needed", side="right", overlaying="y", showgrid=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=430,
)
st.plotly_chart(fig, width='stretch')

forecast_df = pd.DataFrame(
    {
        "Date": forecast_30.index.date,
        "Predicted Bookings": forecast_30.values.round(1),
        "Naive Baseline": baseline_30.values.round(1),
        "Staff Needed": staff_needed.values,
    }
)
if demand_band_df is not None:
    forecast_df["Low"] = demand_band_df["yhat_lower"].round(1).values
    forecast_df["High"] = demand_band_df["yhat_upper"].round(1).values
st.dataframe(forecast_df, width='stretch')

st.divider()
st.subheader("Market Segment Demand Outlook")
st.caption(
    "Separate top-segment forecasts help distinguish whether soft demand is coming from OTA, Direct, Corporate, or group business."
)

@st.cache_resource(show_spinner=False)
def _segment_models(dataframe: pd.DataFrame):
    return train_segment_demand_models(
        dataframe,
        top_n=APP_CONFIG.get("segment_top_n", 4),
        lags=FORECAST_LAGS,
    )


segment_models = _segment_models(df)
segment_forecast_df = forecast_segment_demand(segment_models, periods=30)
segment_mix_df = build_segment_mix_report(
    df,
    periods=30,
    top_n=APP_CONFIG.get("segment_top_n", 4),
)

if segment_forecast_df.empty:
    st.info("Not enough segment history to train separate segment forecasters.")
else:
    seg_summary = (
        segment_forecast_df.groupby("segment")["yhat"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"yhat": "30d_forecast_bookings"})
    )
    st.dataframe(seg_summary, width='stretch')

    seg_chart = go.Figure()
    for segment in seg_summary["segment"]:
        seg_slice = segment_forecast_df[segment_forecast_df["segment"] == segment]
        seg_chart.add_trace(
            go.Scatter(
                x=seg_slice["ds"],
                y=seg_slice["yhat"],
                mode="lines",
                stackgroup="one",
                name=segment,
            )
        )
    seg_chart.update_layout(
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Forecast Bookings",
        height=380,
    )
    st.plotly_chart(seg_chart, width='stretch')

if not segment_mix_df.empty:
    mix_summary = (
        segment_mix_df.groupby("segment")["otb_bookings"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    mix_summary["segment_share"] = mix_summary["otb_bookings"] / mix_summary["otb_bookings"].sum()
    mix_cols = st.columns(min(len(mix_summary), 4) or 1)
    for idx, row in mix_summary.head(4).iterrows():
        mix_cols[idx].metric(row["segment"], f"{row['segment_share']:.1%}")
    st.dataframe(segment_mix_df, width='stretch')

st.divider()
st.subheader("Occupied Rooms by Stay Night")
st.caption(
    "This stay-night view is useful for operations: room-nights in-house can be very different from arrival counts."
)

try:
    stay_df = expand_stay_nights(df)
    occ_df = aggregate_stay_nights(stay_df)
    occ_hist = occ_df.iloc[-90:] if len(occ_df) >= 90 else occ_df

    occ_col1, occ_col2 = st.columns(2)
    occ_col1.metric("Avg Occupied Rooms/Night (hist.)", f"{occ_hist['occupied_rooms'].mean():.1f}")
    occ_col2.metric("Peak Occupied Rooms", f"{int(occ_hist['occupied_rooms'].max())}")

    fig_occ = go.Figure()
    fig_occ.add_trace(
        go.Bar(
            x=occ_hist.index,
            y=occ_hist["occupied_rooms"].values,
            name="Occupied Rooms",
            marker_color="mediumseagreen",
            opacity=0.8,
        )
    )
    fig_occ.update_layout(
        xaxis_title="Date",
        yaxis_title="Occupied Rooms",
        height=360,
        hovermode="x unified",
    )
    st.plotly_chart(fig_occ, width='stretch')
except Exception as exc:
    st.warning(f"Could not compute stay-night occupancy: {exc}")
