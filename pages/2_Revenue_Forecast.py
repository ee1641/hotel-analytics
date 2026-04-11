"""
Page 2 — Revenue & Demand Forecast
XGBoost vs Prophet vs naive baseline, metrics, cross-validation table.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import load_and_prepare_data, get_or_train_models
from src.forecasting import forecast_xgb, forecast_prophet, naive_baseline_forecast

st.set_page_config(page_title="Revenue Forecast", page_icon="📈", layout="wide")
st.title("📈 Revenue & Demand Forecast")
st.caption(
    "Revenue and demand figures are aggregated by **arrival date**. "
    "Multi-night stays are not spread across stay nights."
)

try:
    with st.spinner("Loading models..."):
        _, daily_df = load_and_prepare_data()
        models = get_or_train_models()
    eval_metrics = models["eval_metrics"]
except Exception as e:
    st.error(f"Failed to load data or models: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    series_choice = st.radio("Series", ["Revenue", "Demand"], horizontal=True)
with col_ctrl2:
    model_choice = st.radio("Forecast Model", ["XGBoost", "Prophet", "Naive Baseline"], horizontal=True)

# ---------------------------------------------------------------------------
# Resolve series + model keys
# ---------------------------------------------------------------------------
if series_choice == "Revenue":
    series = daily_df["daily_revenue"]
    y_label = "Revenue (€)"
    xgb_key, prophet_key = "xgb_revenue", "prophet_revenue"
    xgb_metric_key, prophet_metric_key, naive_key = "revenue_xgb", "revenue_prophet", "revenue_naive"
else:
    series = daily_df["daily_bookings"]
    y_label = "Bookings"
    xgb_key, prophet_key = "xgb_demand", "prophet_demand"
    xgb_metric_key, prophet_metric_key, naive_key = "demand_xgb", "demand_prophet", "demand_naive"

xgb_model, feat_cols = models[xgb_key]
prophet_model = models[prophet_key]

# ---------------------------------------------------------------------------
# Metrics row
# ---------------------------------------------------------------------------
st.subheader("Model Performance — Held-Out Test Set (Last 30 Days, Unseen During Training)")
m_xgb     = eval_metrics[xgb_metric_key]
m_prophet = eval_metrics.get(prophet_metric_key, {})
m_naive   = eval_metrics[naive_key]

metric_cols = st.columns(3)
with metric_cols[0]:
    st.markdown("**XGBoost**")
    st.metric("MAE",  f"{m_xgb['mae']:,.1f}")
    st.metric("MAPE", f"{m_xgb['mape']:.1f}%")
with metric_cols[1]:
    st.markdown("**Prophet**")
    st.metric("MAE",  f"{m_prophet.get('mae', 0):,.1f}")
    st.metric("MAPE", f"{m_prophet.get('mape', 0):.1f}%")
with metric_cols[2]:
    st.markdown("**Naive Baseline**")
    st.metric("MAE",  f"{m_naive['mae']:,.1f}")
    st.metric("MAPE", f"{m_naive['mape']:.1f}%")

st.info(
    "**Evaluation methodology:** Evaluation models are trained on all data except the final 30 days. "
    "Metrics above are computed on those 30 days (never seen during training). "
    "The production forecast below uses models retrained on the full dataset."
)

# Improvement callout
if model_choice == "XGBoost":
    selected_mape = m_xgb["mape"]
elif model_choice == "Prophet":
    selected_mape = m_prophet.get("mape", m_naive["mape"])
else:
    selected_mape = m_naive["mape"]

improvement = m_naive["mape"] - selected_mape
if model_choice != "Naive Baseline":
    if improvement > 0:
        st.info(f"{model_choice} outperforms the naive baseline by **{improvement:.1f} MAPE points**.")
    else:
        st.warning(f"{model_choice} underperforms the naive baseline by **{abs(improvement):.1f} MAPE points**.")

st.divider()

# ---------------------------------------------------------------------------
# Forecast chart — last 90 days + 30-day forecast + optional confidence band
# ---------------------------------------------------------------------------
st.subheader(f"30-Day {series_choice} Forecast")
history_window = series.iloc[-90:]

show_ci = False
if model_choice == "XGBoost":
    forecast_30 = forecast_xgb(xgb_model, feat_cols, series, periods=30)
elif model_choice == "Prophet":
    prophet_df = forecast_prophet(prophet_model, periods=30)
    forecast_30 = prophet_df.set_index("ds")["yhat"].rename("forecast")
    forecast_30.index = pd.to_datetime(forecast_30.index)
    show_ci = True
else:
    forecast_30 = naive_baseline_forecast(series, periods=30)

baseline_30 = naive_baseline_forecast(series, periods=30)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=history_window.index, y=history_window.values,
    name="Historical", line=dict(color="steelblue", width=2),
))

if show_ci and model_choice == "Prophet":
    fig.add_trace(go.Scatter(
        x=prophet_df["ds"].tolist() + prophet_df["ds"].tolist()[::-1],
        y=prophet_df["yhat_upper"].tolist() + prophet_df["yhat_lower"].tolist()[::-1],
        fill="toself", fillcolor="rgba(255,99,71,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="90% Confidence Band", showlegend=True,
    ))

fig.add_trace(go.Scatter(
    x=forecast_30.index, y=forecast_30.values,
    name=f"{model_choice} Forecast",
    line=dict(color="tomato", width=2, dash="dash"),
))
fig.add_trace(go.Scatter(
    x=baseline_30.index, y=baseline_30.values,
    name="Naive Baseline", line=dict(color="gray", width=1, dash="dot"),
))
fig.update_layout(
    hovermode="x unified",
    xaxis_title="Date",
    yaxis_title=y_label,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=420,
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Forecast table
# ---------------------------------------------------------------------------
with st.expander("View 30-Day Forecast Data"):
    forecast_df = pd.DataFrame({
        "Date": forecast_30.index.date,
        f"{model_choice} Forecast": forecast_30.values.round(2),
        "Naive Baseline": baseline_30.values.round(2),
    })
    st.dataframe(forecast_df, use_container_width=True)

# ---------------------------------------------------------------------------
# Cross-validation table (revenue only)
# ---------------------------------------------------------------------------
with st.expander("Cross-Validation Detail (Revenue, XGBoost, expanding window)"):
    cv_df = eval_metrics.get("cv_scores")
    if cv_df is not None and len(cv_df) > 0:
        st.dataframe(cv_df, use_container_width=True)
        avg_mae = cv_df["mae"].mean()
        st.caption(f"Average CV MAE: {avg_mae:,.2f}")
    else:
        st.info("Cross-validation scores not available.")
