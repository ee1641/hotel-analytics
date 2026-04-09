"""
Page 5 — Profit Simulator
Adjust tax rate and cost ratio → see 30-day profit breakdown.
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
st.title("💰 Profit & Tax Simulator")
st.markdown(
    "Adjust financial parameters below to simulate the impact on your 30-day profit projection. "
    "Base revenue is derived from the XGBoost forecast."
)

try:
    with st.spinner("Loading models..."):
        _, daily_df = load_and_prepare_data()
        models = get_or_train_models()
except Exception as e:
    st.error(f"Failed to load data or models: {e}")
    st.stop()

xgb_model, feat_cols = models["xgb_revenue"]
rev_series  = daily_df["daily_revenue"]
forecast_30 = forecast_xgb(xgb_model, feat_cols, rev_series, periods=30)
total_rev   = float(forecast_30.sum())

# ---------------------------------------------------------------------------
# Sliders
# ---------------------------------------------------------------------------
st.sidebar.header("Financial Parameters")
tax_rate   = st.sidebar.slider("Tax Rate (%)",   min_value=0, max_value=50, value=10, step=1) / 100
cost_ratio = st.sidebar.slider("Cost Ratio (%)", min_value=0, max_value=90, value=60, step=5) / 100

# ---------------------------------------------------------------------------
# Compute financials
# ---------------------------------------------------------------------------
fin = estimate_profit(total_rev, tax_rate=tax_rate, cost_ratio=cost_ratio)
op_cost = float(fin["operational_cost"])
tax     = float(fin["tax"])
profit  = float(fin["profit"])

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Projected Revenue",  f"€{total_rev:,.0f}")
col2.metric("Operational Cost",   f"€{op_cost:,.0f}")
col3.metric("Tax",                f"€{tax:,.0f}")
profit_delta = profit - (total_rev * (1 - 0.10 - 0.60))  # vs defaults
col4.metric("Net Profit", f"€{profit:,.0f}", delta=f"€{profit_delta:+,.0f} vs defaults")

st.divider()

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Revenue Breakdown")
    breakdown_df = pd.DataFrame({
        "Component": ["Operational Cost", "Tax", "Net Profit"],
        "Amount":    [op_cost, tax, profit],
    })
    color_map = {
        "Operational Cost": "#EF553B",
        "Tax":              "#FFA15A",
        "Net Profit":       "#00CC96",
    }
    fig_pie = px.pie(
        breakdown_df, values="Amount", names="Component",
        color="Component", color_discrete_map=color_map,
        title=f"30-Day Revenue Breakdown (Tax {tax_rate:.0%}, Cost {cost_ratio:.0%})",
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

with chart_col2:
    st.subheader("Daily Profit Breakdown")
    daily_fin = estimate_profit(forecast_30, tax_rate=tax_rate, cost_ratio=cost_ratio)
    daily_df_fin = pd.DataFrame({
        "Date":             forecast_30.index.date,
        "Operational Cost": daily_fin["operational_cost"].values,
        "Tax":              daily_fin["tax"].values,
        "Net Profit":       daily_fin["profit"].values,
    })
    fig_bar = px.bar(
        daily_df_fin, x="Date",
        y=["Operational Cost", "Tax", "Net Profit"],
        labels={"value": "€", "variable": "Component"},
        color_discrete_map=color_map,
        title="Daily Financial Breakdown (Next 30 Days)",
    )
    fig_bar.update_layout(barmode="stack", height=380)
    st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
st.subheader("30-Day Summary")
summary_df = pd.DataFrame({
    "Component":  ["Projected Revenue", "Operational Cost", "Tax", "Net Profit"],
    "Amount (€)": [total_rev, op_cost, tax, profit],
    "% of Revenue": [100.0, cost_ratio * 100, tax_rate * 100, (profit / total_rev * 100) if total_rev else 0],
})

st.dataframe(
    summary_df.style.format({"Amount (€)": "€{:,.2f}", "% of Revenue": "{:.1f}%"}),
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Insight
# ---------------------------------------------------------------------------
if profit > 0:
    margin = profit / total_rev if total_rev else 0
    st.success(f"Projected net margin: **{margin:.1%}**. Operations look profitable for the next 30 days.")
else:
    st.error("Projected net profit is negative under current parameters. Consider reducing costs or renegotiating tax arrangements.")
