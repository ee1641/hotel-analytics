"""
Page 1 - Overview Dashboard
KPIs, revenue and bookings over time, hotel/date filters.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import plotly.express as px
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import load_and_prepare_data, get_or_train_models
from src.time_series import aggregate_daily

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")
st.title("📊 Overview Dashboard")

try:
    df, daily_df = load_and_prepare_data()
    models = get_or_train_models()
except Exception as e:
    st.error(f"Failed to load data or models: {e}")
    st.stop()

df = models.get("df_with_expected_revenue", df)
daily_df = models.get("daily_revenue_views", daily_df)

st.sidebar.header("Filters")

hotel_options = ["All"] + sorted(df["hotel"].dropna().unique().tolist())
selected_hotel = st.sidebar.selectbox("Hotel Type", hotel_options)

date_min = daily_df.index.min().date()
date_max = daily_df.index.max().date()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max,
)

df_filtered = df.copy()
if selected_hotel != "All":
    df_filtered = df_filtered[df_filtered["hotel"] == selected_hotel]

if len(date_range) == 2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    df_filtered = df_filtered[
        (df_filtered["arrival_date"] >= start_date) &
        (df_filtered["arrival_date"] <= end_date)
    ]
    daily_filtered = aggregate_daily(df_filtered)
    daily_filtered = daily_filtered[start_date:end_date]
else:
    daily_filtered = daily_df.copy()

booked_revenue = daily_filtered["daily_booked_revenue"].sum()
realized_revenue = daily_filtered["daily_realized_revenue"].sum()
expected_realized_revenue = daily_filtered["daily_expected_realized_revenue"].sum()
total_bookings = int(daily_filtered["daily_bookings"].sum())
cancel_rate = df_filtered["is_canceled"].mean() if len(df_filtered) > 0 else 0.0
revenue_gap = booked_revenue - realized_revenue

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Booked Revenue", f"€{booked_revenue:,.0f}")
col2.metric("Realized Revenue", f"€{realized_revenue:,.0f}")
col3.metric("Expected Realized", f"€{expected_realized_revenue:,.0f}")
col4.metric("Revenue Lost", f"€{revenue_gap:,.0f}")
col5.metric("Total Bookings", f"{total_bookings:,}")
col6.metric("Cancellation Rate", f"{cancel_rate:.1%}")

st.divider()

st.subheader("Daily Revenue Over Time")
revenue_plot_df = daily_filtered.reset_index().rename(
    columns={
        "daily_booked_revenue": "Booked Revenue",
        "daily_realized_revenue": "Realized Revenue",
        "daily_expected_realized_revenue": "Expected Realized Revenue",
    }
)
fig_rev = px.line(
    revenue_plot_df,
    x="arrival_date",
    y=["Booked Revenue", "Realized Revenue", "Expected Realized Revenue"],
    labels={"arrival_date": "Date", "value": "Revenue (€)", "variable": "Revenue Type"},
    color_discrete_map={
        "Booked Revenue": "steelblue",
        "Realized Revenue": "seagreen",
        "Expected Realized Revenue": "darkorange",
    },
)
fig_rev.update_layout(hovermode="x unified", height=350)
st.plotly_chart(fig_rev, use_container_width=True)

st.subheader("Daily Bookings Over Time")
fig_bk = px.line(
    daily_filtered.reset_index(),
    x="arrival_date",
    y="daily_bookings",
    labels={"arrival_date": "Date", "daily_bookings": "Bookings"},
    color_discrete_sequence=["darkorange"],
)
fig_bk.update_layout(hovermode="x unified", height=350)
st.plotly_chart(fig_bk, use_container_width=True)

st.subheader("Monthly Cancellation Breakdown")
if len(df_filtered) > 0:
    df_filtered = df_filtered.copy()
    df_filtered["year_month"] = df_filtered["arrival_date"].dt.to_period("M").astype(str)
    monthly = (
        df_filtered.groupby("year_month")["is_canceled"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "cancellations", "count": "bookings"})
        .reset_index()
    )
    monthly["cancel_rate"] = monthly["cancellations"] / monthly["bookings"]
    fig_m = px.bar(
        monthly,
        x="year_month",
        y="cancel_rate",
        labels={"year_month": "Month", "cancel_rate": "Cancellation Rate"},
        color="cancel_rate",
        color_continuous_scale="RdYlGn_r",
    )
    fig_m.update_layout(height=300, xaxis_tickangle=-45)
    fig_m.update_coloraxes(showscale=False)
    st.plotly_chart(fig_m, use_container_width=True)

if cancel_rate > 0.40:
    st.warning(f"High cancellation rate detected: **{cancel_rate:.1%}**. Consider reviewing deposit policies.")
elif cancel_rate > 0.30:
    st.info(f"Cancellation rate is moderate: **{cancel_rate:.1%}**.")
else:
    st.success(f"Cancellation rate is healthy: **{cancel_rate:.1%}**.")

expected_capture = (expected_realized_revenue / booked_revenue) if booked_revenue else 0.0
st.caption(
    f"Booked revenue includes all bookings. Realized revenue excludes cancellations. "
    f"Expected realized revenue applies cancellation-risk estimates and currently captures **{expected_capture:.1%}** of booked revenue."
)

cross = models["eval_metrics"].get("cross_dataset", {})
if cross:
    with st.expander("Cross-Dataset Validation (INN Hotels Group dataset)"):
        st.markdown(
            "The cancellation models were trained on the Kaggle hotel_bookings dataset "
            "and evaluated on a separate INN Hotels Group dataset to assess generalization."
        )
        rows = [
            {"Model": name, "Accuracy": v.get("accuracy"), "ROC-AUC": v.get("roc_auc"),
             "Samples": v.get("n_samples")}
            for name, v in cross.items()
            if v.get("roc_auc") is not None
        ]
        if rows:
            import pandas as _pd
            cross_df = _pd.DataFrame(rows)
            st.dataframe(
                cross_df.style.format({"Accuracy": "{:.3f}", "ROC-AUC": "{:.3f}"}),
                use_container_width=True,
            )
        else:
            st.info("Cross-dataset evaluation results not available.")
