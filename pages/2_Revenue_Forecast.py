"""
Page 2 - Revenue and Demand Forecast
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
from src.time_series import expand_stay_nights, aggregate_stay_nights

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

daily_df = models.get("daily_revenue_views", daily_df)

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
            "prophet_key": "prophet_revenue",
            "metrics": ("revenue_booked_xgb", "revenue_booked_prophet", "revenue_booked_naive"),
        },
        "Realized Revenue": {
            "column": "daily_realized_revenue",
            "label": "Realized Revenue (€)",
            "xgb_key": "xgb_realized_revenue",
            "prophet_key": "prophet_realized_revenue",
            "metrics": ("revenue_realized_xgb", "revenue_realized_prophet", "revenue_realized_naive"),
        },
        "Expected Realized Revenue": {
            "column": "daily_expected_realized_revenue",
            "label": "Expected Realized Revenue (€)",
            "xgb_key": "xgb_expected_revenue",
            "prophet_key": "prophet_expected_revenue",
            "metrics": ("revenue_expected_xgb", "revenue_expected_prophet", "revenue_expected_naive"),
        },
    }[revenue_type]

    series = daily_df[revenue_config["column"]]
    y_label = revenue_config["label"]
    xgb_key = revenue_config["xgb_key"]
    prophet_key = revenue_config["prophet_key"]
    xgb_metric_key, prophet_metric_key, naive_key = revenue_config["metrics"]
    forecast_title = f"30-Day {revenue_type} Forecast"
else:
    series = daily_df["daily_bookings"]
    y_label = "Bookings"
    xgb_key, prophet_key = "xgb_demand", "prophet_demand"
    xgb_metric_key, prophet_metric_key, naive_key = "demand_xgb", "demand_prophet", "demand_naive"
    forecast_title = "30-Day Demand Forecast"

xgb_model, feat_cols = models[xgb_key]
prophet_model = models[prophet_key]

st.subheader("Model Performance — Held-Out Test Set (Last 30 Days, Unseen During Training)")
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
st.info(
    "**Evaluation methodology:** Models are trained on all data except the final 30 days. "
    "Metrics above are computed on those 30 days (never seen during training). "
    "The production forecast uses models retrained on the full dataset."
)

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

st.subheader(forecast_title)
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

with st.expander("View 30-Day Forecast Data"):
    forecast_df = pd.DataFrame({
        "Date": forecast_30.index.date,
        f"{model_choice} Forecast": forecast_30.values.round(2),
        "Naive Baseline": baseline_30.values.round(2),
    })
    st.dataframe(forecast_df, use_container_width=True)

with st.expander("Cross-Validation Detail (Booked Revenue, XGBoost, expanding window)"):
    cv_df = eval_metrics.get("cv_scores")
    if cv_df is not None and len(cv_df) > 0:
        st.dataframe(cv_df, use_container_width=True)
        avg_mae = cv_df["mae"].mean()
        st.caption(f"Average CV MAE: {avg_mae:,.2f}")
    else:
        st.info("Cross-validation scores not available.")

# ---------------------------------------------------------------------------
# Revenue by stay night (historical)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Revenue by Stay Night (Historical)")
st.caption(
    "Revenue is spread across all occupied nights — not just the arrival date. "
    "ADR × nights occupied gives a truer picture of when revenue was earned."
)

try:
    stay_df_rev = expand_stay_nights(df)
    occ_rev_df = aggregate_stay_nights(stay_df_rev)
    rev_hist = occ_rev_df.iloc[-90:] if len(occ_rev_df) >= 90 else occ_rev_df

    fig_rev_stay = go.Figure()
    fig_rev_stay.add_trace(go.Bar(
        x=rev_hist.index,
        y=rev_hist["daily_stay_revenue"].values,
        name="Revenue (stay nights)",
        marker_color="darkorange",
        opacity=0.8,
    ))
    fig_rev_stay.update_layout(
        xaxis_title="Date",
        yaxis_title="Revenue (€)",
        height=380,
        hovermode="x unified",
    )
    st.plotly_chart(fig_rev_stay, use_container_width=True)
    st.caption(
        f"Total stay-night revenue (last 90 days): "
        f"€{rev_hist['daily_stay_revenue'].sum():,.0f}  |  "
        f"Avg/night: €{rev_hist['daily_stay_revenue'].mean():,.0f}"
    )
except Exception as exc:
    st.warning(f"Could not compute stay-night revenue: {exc}")
