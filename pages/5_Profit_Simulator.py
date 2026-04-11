"""
Page 5 - Profit Simulator
Adjust tax rate and cost ratio to see a 30-day profit breakdown.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import plotly.express as px
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import load_and_prepare_data, get_or_train_models
from src.forecasting import forecast_xgb
from src.analytics import estimate_profit

st.set_page_config(page_title="Profit Simulator", page_icon="💰", layout="wide")
st.title("💰 Profit and Tax Simulator")
st.markdown(
    "Adjust financial parameters below to simulate the impact on your 30-day profit projection. "
    "Profit calculations use **expected realized revenue** so cancellation risk is reflected in the baseline."
)

try:
    _, daily_df = load_and_prepare_data()
    models = get_or_train_models()
except Exception as e:
    st.error(f"Failed to load data or models: {e}")
    st.stop()

daily_df = models.get("daily_revenue_views", daily_df)

booked_model, booked_cols = models["xgb_revenue"]
expected_model, expected_cols = models["xgb_expected_revenue"]
realized_model, realized_cols = models["xgb_realized_revenue"]

booked_series = daily_df["daily_booked_revenue"]
realized_series = daily_df["daily_realized_revenue"]
expected_series = daily_df["daily_expected_realized_revenue"]

forecast_booked = forecast_xgb(booked_model, booked_cols, booked_series, periods=30)
forecast_realized = forecast_xgb(realized_model, realized_cols, realized_series, periods=30)
forecast_expected = forecast_xgb(expected_model, expected_cols, expected_series, periods=30)

projected_booked = float(forecast_booked.sum())
projected_realized = float(forecast_realized.sum())
projected_expected = float(forecast_expected.sum())

st.sidebar.header("Financial Parameters")
tax_rate = st.sidebar.slider("Tax Rate (%)", min_value=0, max_value=50, value=10, step=1) / 100
cost_ratio = st.sidebar.slider("Cost Ratio (%)", min_value=0, max_value=90, value=60, step=5) / 100

fin = estimate_profit(projected_expected, tax_rate=tax_rate, cost_ratio=cost_ratio)
op_cost = float(fin["operational_cost"])
tax = float(fin["tax"])
profit = float(fin["profit"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Projected Booked Revenue", f"€{projected_booked:,.0f}")
col2.metric("Projected Realized Revenue", f"€{projected_realized:,.0f}")
col3.metric("Projected Expected Revenue", f"€{projected_expected:,.0f}")
col4.metric("Net Profit", f"€{profit:,.0f}")

st.caption(
    "Booked revenue is the gross booking value. Realized revenue excludes actual cancellations. "
    "Expected realized revenue is the profit baseline because it adjusts upcoming revenue for cancellation risk."
)

st.divider()

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Expected Revenue Breakdown")
    breakdown_df = pd.DataFrame({
        "Component": ["Operational Cost", "Tax", "Net Profit"],
        "Amount": [op_cost, tax, profit],
    })
    color_map = {
        "Operational Cost": "#EF553B",
        "Tax": "#FFA15A",
        "Net Profit": "#00CC96",
    }
    fig_pie = px.pie(
        breakdown_df, values="Amount", names="Component",
        color="Component", color_discrete_map=color_map,
        title=f"30-Day Expected Revenue Breakdown (Tax {tax_rate:.0%}, Cost {cost_ratio:.0%})",
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

with chart_col2:
    st.subheader("30-Day Revenue Outlook")
    revenue_outlook_df = pd.DataFrame({
        "Date": forecast_expected.index.date,
        "Booked Revenue": forecast_booked.values,
        "Realized Revenue": forecast_realized.values,
        "Expected Realized Revenue": forecast_expected.values,
    })
    fig_lines = px.line(
        revenue_outlook_df,
        x="Date",
        y=["Booked Revenue", "Realized Revenue", "Expected Realized Revenue"],
        labels={"value": "€", "variable": "Revenue Type"},
        color_discrete_map={
            "Booked Revenue": "steelblue",
            "Realized Revenue": "seagreen",
            "Expected Realized Revenue": "darkorange",
        },
    )
    fig_lines.update_layout(height=380)
    st.plotly_chart(fig_lines, use_container_width=True)

st.subheader("30-Day Summary")
summary_df = pd.DataFrame({
    "Component": [
        "Projected Booked Revenue",
        "Projected Realized Revenue",
        "Projected Expected Revenue",
        "Operational Cost",
        "Tax",
        "Net Profit",
    ],
    "Amount (€)": [projected_booked, projected_realized, projected_expected, op_cost, tax, profit],
    "% of Expected Revenue": [
        (projected_booked / projected_expected * 100) if projected_expected else 0,
        (projected_realized / projected_expected * 100) if projected_expected else 0,
        100.0,
        cost_ratio * 100,
        tax_rate * 100,
        (profit / projected_expected * 100) if projected_expected else 0,
    ],
})

st.dataframe(
    summary_df.style.format({"Amount (€)": "€{:,.2f}", "% of Expected Revenue": "{:.1f}%"}),
    use_container_width=True,
)

if profit > 0:
    margin = profit / projected_expected if projected_expected else 0
    st.success(f"Projected net margin on expected realized revenue: **{margin:.1%}**.")
else:
    st.error("Projected net profit is negative under current parameters. Consider reducing costs or renegotiating tax arrangements.")
