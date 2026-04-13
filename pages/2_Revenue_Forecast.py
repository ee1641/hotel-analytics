"""
Page 2 - Revenue and Demand Forecast
Richer forecast features, pickup / pace, pricing, and uncertainty.
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
from src.config import APP_CONFIG
from src.forecasting import (
    forecast_prophet,
    forecast_xgb,
    forecast_xgb_with_intervals,
    naive_baseline_forecast,
)
from src.revenue_management import (
    build_pickup_report,
    default_snapshot_date,
    fit_price_elasticity_models,
    recommend_dynamic_prices,
)
from src.time_series import aggregate_stay_nights, expand_stay_nights

st.set_page_config(page_title="Revenue Forecast", page_icon="📈", layout="wide")
st.title("📈 Revenue and Demand Forecast")

if st.session_state.get("defaulted_columns"):
    st.info(
        "Note: some columns were not in your uploaded dataset and use default assumptions: "
        + ", ".join(f"**{c.split(' —')[0]}**" for c in st.session_state["defaulted_columns"])
        + ". Results may not reflect real values."
    )

try:
    df, daily_df = load_and_prepare_data()
    models = get_or_train_models()
    eval_metrics = models["eval_metrics"]
except Exception as e:
    st.error(f"Failed to load data or models: {e}")
    st.stop()

df = models.get("df_with_expected_revenue", df)
daily_df = models.get("daily_revenue_views", daily_df)

st.sidebar.header("Revenue Management Controls")
total_rooms = st.sidebar.number_input(
    "Room Capacity",
    min_value=10,
    max_value=5000,
    value=APP_CONFIG.get("default_total_rooms", 100),
)
max_snapshot = (df["arrival_date"].max() - pd.Timedelta(days=1)).date()
min_snapshot = df["arrival_date"].min().date()
default_snapshot = default_snapshot_date(
    df,
    periods=APP_CONFIG.get("forecast_horizon_days", 30),
).date()
snapshot_date = pd.Timestamp(
    st.sidebar.date_input(
        "Pickup Snapshot Date",
        value=default_snapshot,
        min_value=min_snapshot,
        max_value=max_snapshot,
    )
)

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    series_choice = st.radio("Series", ["Revenue", "Demand"], horizontal=True)
with col_ctrl2:
    model_choice = st.radio("Forecast Model", ["XGBoost", "Prophet", "Naive Baseline"], horizontal=True)

if series_choice == "Revenue":
    revenue_type = st.radio(
        "Revenue Type",
        ["Booked Revenue", "Realized Revenue", "Expected Realized Revenue"],
        horizontal=True,
    )
    revenue_config = {
        "Booked Revenue": {
            "column": "daily_booked_revenue",
            "label": "Booked Revenue (€)",
            "xgb_key": "xgb_revenue",
            "interval_key": "xgb_revenue_intervals",
            "prophet_key": "prophet_revenue",
            "metrics": ("revenue_booked_xgb", "revenue_booked_prophet", "revenue_booked_naive"),
        },
        "Realized Revenue": {
            "column": "daily_realized_revenue",
            "label": "Realized Revenue (€)",
            "xgb_key": "xgb_realized_revenue",
            "interval_key": "xgb_realized_revenue_intervals",
            "prophet_key": "prophet_realized_revenue",
            "metrics": ("revenue_realized_xgb", "revenue_realized_prophet", "revenue_realized_naive"),
        },
        "Expected Realized Revenue": {
            "column": "daily_expected_realized_revenue",
            "label": "Expected Realized Revenue (€)",
            "xgb_key": "xgb_expected_revenue",
            "interval_key": "xgb_expected_revenue_intervals",
            "prophet_key": "prophet_expected_revenue",
            "metrics": ("revenue_expected_xgb", "revenue_expected_prophet", "revenue_expected_naive"),
        },
    }[revenue_type]

    series = daily_df[revenue_config["column"]]
    y_label = revenue_config["label"]
    xgb_key = revenue_config["xgb_key"]
    interval_key = revenue_config["interval_key"]
    prophet_key = revenue_config["prophet_key"]
    xgb_metric_key, prophet_metric_key, naive_key = revenue_config["metrics"]
    forecast_title = f"30-Day {revenue_type} Forecast"
else:
    series = daily_df["daily_bookings"]
    y_label = "Bookings"
    xgb_key = "xgb_demand"
    interval_key = "xgb_demand_intervals"
    prophet_key = "prophet_demand"
    xgb_metric_key, prophet_metric_key, naive_key = "demand_xgb", "demand_prophet", "demand_naive"
    forecast_title = "30-Day Demand Forecast"

xgb_model, feat_cols = models[xgb_key]
prophet_model = models[prophet_key]

st.subheader("Model Performance")
m_xgb = eval_metrics[xgb_metric_key]
m_prophet = eval_metrics.get(prophet_metric_key, {})
m_naive = eval_metrics[naive_key]

metric_cols = st.columns(3)
with metric_cols[0]:
    st.markdown("**XGBoost**")
    st.metric("MAE", f"{m_xgb['mae']:,.1f}")
    st.metric("MAPE", f"{m_xgb['mape']:.1f}%")
with metric_cols[1]:
    st.markdown("**Prophet**")
    st.metric("MAE", f"{m_prophet.get('mae', 0):,.1f}")
    st.metric("MAPE", f"{m_prophet.get('mape', 0):.1f}%")
with metric_cols[2]:
    st.markdown("**Naive Baseline**")
    st.metric("MAE", f"{m_naive['mae']:,.1f}")
    st.metric("MAPE", f"{m_naive['mape']:.1f}%")

selected_mape = {
    "XGBoost": m_xgb["mape"],
    "Prophet": m_prophet.get("mape", m_naive["mape"]),
    "Naive Baseline": m_naive["mape"],
}[model_choice]
improvement = m_naive["mape"] - selected_mape
if model_choice != "Naive Baseline":
    if improvement > 0:
        st.info(f"{model_choice} beats the seasonal naive baseline by **{improvement:.1f} MAPE points**.")
    else:
        st.warning(f"{model_choice} trails the seasonal naive baseline by **{abs(improvement):.1f} MAPE points**.")

st.divider()

st.subheader(forecast_title)
history_window = series.iloc[-90:]
show_ci = False
band_df = None

if model_choice == "XGBoost":
    interval_models = models.get(interval_key)
    if interval_models:
        band_df = forecast_xgb_with_intervals(interval_models, series, periods=30)
        band_df["ds"] = pd.to_datetime(band_df["ds"])
        forecast_30 = band_df.set_index("ds")["yhat"].rename("forecast")
        show_ci = True
    else:
        forecast_30 = forecast_xgb(xgb_model, feat_cols, series, periods=30)
elif model_choice == "Prophet":
    band_df = forecast_prophet(prophet_model, periods=30)
    band_df["ds"] = pd.to_datetime(band_df["ds"])
    forecast_30 = band_df.set_index("ds")["yhat"].rename("forecast")
    show_ci = True
else:
    forecast_30 = naive_baseline_forecast(series, periods=30)

baseline_30 = naive_baseline_forecast(series, periods=30)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=history_window.index,
        y=history_window.values,
        name="Historical",
        line=dict(color="steelblue", width=2),
    )
)

if show_ci and band_df is not None:
    fig.add_trace(
        go.Scatter(
            x=band_df["ds"].tolist() + band_df["ds"].tolist()[::-1],
            y=band_df["yhat_upper"].tolist() + band_df["yhat_lower"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(255,99,71,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Forecast Range",
            showlegend=True,
        )
    )

fig.add_trace(
    go.Scatter(
        x=forecast_30.index,
        y=forecast_30.values,
        name=f"{model_choice} Forecast",
        line=dict(color="tomato", width=2, dash="dash"),
    )
)
fig.add_trace(
    go.Scatter(
        x=baseline_30.index,
        y=baseline_30.values,
        name="Naive Baseline",
        line=dict(color="gray", width=1, dash="dot"),
    )
)
fig.update_layout(
    hovermode="x unified",
    xaxis_title="Date",
    yaxis_title=y_label,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=430,
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("View 30-Day Forecast Data"):
    forecast_df = pd.DataFrame(
        {
            "Date": forecast_30.index.date,
            f"{model_choice} Forecast": forecast_30.values.round(2),
            "Naive Baseline": baseline_30.values.round(2),
        }
    )
    if show_ci and band_df is not None:
        forecast_df["Low"] = band_df["yhat_lower"].round(2).values
        forecast_df["High"] = band_df["yhat_upper"].round(2).values
    st.dataframe(forecast_df, use_container_width=True)

st.divider()
st.subheader("Pickup / Pace Report")
st.caption(
    "This simulates the revenue manager’s daily OTB view from the selected snapshot date. "
    "For demo data, the snapshot is a historical date near the end of the dataset."
)

pickup_report = build_pickup_report(
    df,
    snapshot_date=snapshot_date,
    periods=30,
    capacity=int(total_rooms),
)

pickup_cols = st.columns(4)
pickup_cols[0].metric("Future Dates Shown", str(len(pickup_report)))
pickup_cols[1].metric("Total OTB", f"{int(pickup_report['otb_today'].sum()):,}")
pickup_cols[2].metric(
    "Avg Pace vs STLY",
    f"{pickup_report['pace_vs_stly'].dropna().mean():.1%}" if pickup_report["pace_vs_stly"].notna().any() else "N/A",
)
pickup_cols[3].metric("7-Day Pickup", f"{int(pickup_report['pickup_7d'].sum()):,}")

pace_fig = go.Figure()
pace_fig.add_trace(
    go.Bar(
        x=pickup_report["date"],
        y=pickup_report["otb_today"],
        name="OTB Today",
        marker_color="darkcyan",
    )
)
pace_fig.add_trace(
    go.Scatter(
        x=pickup_report["date"],
        y=pickup_report["otb_stly_today"],
        name="OTB STLY",
        line=dict(color="darkorange", width=2),
    )
)
pace_fig.update_layout(
    hovermode="x unified",
    xaxis_title="Arrival Date",
    yaxis_title="Bookings On The Books",
    height=360,
)
st.plotly_chart(pace_fig, use_container_width=True)
st.dataframe(
    pickup_report.assign(
        pace_vs_stly=pickup_report["pace_vs_stly"].map(lambda v: None if pd.isna(v) else round(v, 3))
    ),
    use_container_width=True,
)

st.divider()
st.subheader("Directional Pricing Recommendations")
st.caption(
    "These ADR suggestions come from a simple elasticity model on observed ADR vs demand. "
    "They are directional signals, not causal optimization."
)

@st.cache_resource(show_spinner=False)
def _pricing_models(dataframe: pd.DataFrame):
    return fit_price_elasticity_models(dataframe)


elasticity_models = _pricing_models(df)
demand_bands = models.get("xgb_demand_intervals")
if demand_bands:
    demand_forecast_df = forecast_xgb_with_intervals(demand_bands, daily_df["daily_bookings"], periods=30)
else:
    point_forecast = forecast_xgb(models["xgb_demand"][0], models["xgb_demand"][1], daily_df["daily_bookings"], periods=30)
    demand_forecast_df = pd.DataFrame(
        {
            "ds": point_forecast.index,
            "yhat": point_forecast.values,
            "yhat_lower": point_forecast.values,
            "yhat_upper": point_forecast.values,
        }
    )

pricing_df = recommend_dynamic_prices(
    demand_forecast_df,
    pickup_report,
    elasticity_models,
    capacity=int(total_rooms),
)
if pricing_df.empty:
    st.info("Not enough ADR variation to estimate pricing elasticity reliably.")
else:
    pricing_cols = st.columns(3)
    pricing_cols[0].metric("Avg Current ADR", f"€{pricing_df['current_adr'].mean():.0f}")
    pricing_cols[1].metric("Avg Recommended ADR", f"€{pricing_df['recommended_adr'].mean():.0f}")
    pricing_cols[2].metric(
        "Avg ADR Change",
        f"€{pricing_df['adr_change'].mean():+.1f}",
    )
    st.dataframe(
        pricing_df.assign(
            pace_vs_stly=pricing_df["pace_vs_stly"].map(lambda v: None if pd.isna(v) else round(v, 3))
        ),
        use_container_width=True,
    )

st.divider()
st.subheader("Revenue by Stay Night (Historical)")
st.caption(
    "Revenue is spread across all occupied nights, not only on the arrival date. "
    "This helps separate check-in volume from actual room-night revenue."
)

try:
    stay_df_rev = expand_stay_nights(df)
    occ_rev_df = aggregate_stay_nights(stay_df_rev)
    rev_hist = occ_rev_df.iloc[-90:] if len(occ_rev_df) >= 90 else occ_rev_df

    fig_rev_stay = go.Figure()
    fig_rev_stay.add_trace(
        go.Bar(
            x=rev_hist.index,
            y=rev_hist["daily_stay_revenue"].values,
            name="Revenue (stay nights)",
            marker_color="darkorange",
            opacity=0.85,
        )
    )
    fig_rev_stay.update_layout(
        xaxis_title="Date",
        yaxis_title="Revenue (€)",
        height=360,
        hovermode="x unified",
    )
    st.plotly_chart(fig_rev_stay, use_container_width=True)
except Exception as exc:
    st.warning(f"Could not compute stay-night revenue: {exc}")
