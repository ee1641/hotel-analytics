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
    blend_forecasts,
    forecast_prophet,
    forecast_xgb,
    forecast_xgb_direct,
    forecast_xgb_with_intervals,
    naive_baseline_forecast,
)
from src.revenue_management import (
    build_pickup_report,
    default_snapshot_date,
    fit_price_elasticity_models,
    recommend_dynamic_prices,
)
from src.weather import build_inference_weather
from src.time_series import aggregate_daily
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

_hotel_type_options = ["All hotels (combined)"] + models.get("hotel_types_available", [])
hotel_type_choice = st.sidebar.selectbox("Hotel type", _hotel_type_options, index=0)

# Resolve series DataFrame and df slice based on hotel type selection
if hotel_type_choice == "All hotels (combined)":
    _slug_suffix = ""
    _series_source_df = models.get("daily_revenue_views", daily_df)
    _df_for_pickup = df
else:
    _slug = hotel_type_choice.lower().replace(" ", "_")
    _slug_suffix = f"_{_slug}"
    _hotel_mask = df["hotel"] == hotel_type_choice
    _df_for_pickup = df[_hotel_mask]
    _series_source_df = aggregate_daily(_df_for_pickup)
    if _series_source_df.empty:
        st.info(f"No booking data found for '{hotel_type_choice}'. Showing combined model.")
        _slug_suffix = ""
        _series_source_df = models.get("daily_revenue_views", daily_df)
        _df_for_pickup = df

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
    _base_configs = {
        "Booked Revenue": {
            "column": "daily_booked_revenue",
            "label": "Booked Revenue (€)",
            "xgb_base": "xgb_revenue",
            "direct_base": "xgb_revenue_direct",
            "interval_base": "xgb_revenue_intervals",
            "prophet_key": "prophet_revenue",
            "metrics": ("revenue_booked_xgb", "revenue_booked_prophet", "revenue_booked_naive"),
        },
        "Realized Revenue": {
            "column": "daily_realized_revenue",
            "label": "Realized Revenue (€)",
            "xgb_base": "xgb_realized_revenue",
            "direct_base": "xgb_realized_revenue_direct",
            "interval_base": "xgb_realized_revenue_intervals",
            "prophet_key": "prophet_realized_revenue",
            "metrics": ("revenue_realized_xgb", "revenue_realized_prophet", "revenue_realized_naive"),
        },
        "Expected Realized Revenue": {
            "column": "daily_expected_realized_revenue",
            "label": "Expected Realized Revenue (€)",
            "xgb_base": "xgb_expected_revenue",
            "direct_base": "xgb_expected_revenue_direct",
            "interval_base": "xgb_expected_revenue_intervals",
            "prophet_key": "prophet_expected_revenue",
            "metrics": ("revenue_expected_xgb", "revenue_expected_prophet", "revenue_expected_naive"),
        },
    }[revenue_type]

    _col = _base_configs["column"]
    series = _series_source_df[_col] if _col in _series_source_df.columns else daily_df[_col]
    y_label = _base_configs["label"]
    xgb_key = _base_configs["xgb_base"] + _slug_suffix
    direct_key = _base_configs["direct_base"] + _slug_suffix
    interval_key = _base_configs["interval_base"] + _slug_suffix
    prophet_key = _base_configs["prophet_key"]
    xgb_metric_key, prophet_metric_key, naive_key = _base_configs["metrics"]
    forecast_title = f"30-Day {revenue_type} Forecast"
else:
    series = _series_source_df["daily_bookings"] if "daily_bookings" in _series_source_df.columns else daily_df["daily_bookings"]
    y_label = "Bookings"
    xgb_key = "xgb_demand" + _slug_suffix
    direct_key = "xgb_demand_direct" + _slug_suffix
    interval_key = "xgb_demand_intervals" + _slug_suffix
    prophet_key = "prophet_demand"
    xgb_metric_key, prophet_metric_key, naive_key = "demand_xgb", "demand_prophet", "demand_naive"
    forecast_title = "30-Day Demand Forecast"

# Fall back to combined model if per-type model is unavailable
if models.get(xgb_key) is None and _slug_suffix:
    st.info(f"Per-type model for '{hotel_type_choice}' is not yet available; using combined model.")
    xgb_key = xgb_key.replace(_slug_suffix, "")
    interval_key = interval_key.replace(_slug_suffix, "")
    direct_key = direct_key.replace(_slug_suffix, "")

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

_tuning_rows = [
    ("Demand",                   "demand_xgb_baseline",           "demand_xgb"),
    ("Booked Revenue",           "revenue_booked_xgb_baseline",   "revenue_booked_xgb"),
    ("Realized Revenue",         "revenue_realized_xgb_baseline", "revenue_realized_xgb"),
    ("Expected Realized Rev.",   "revenue_expected_xgb_baseline", "revenue_expected_xgb"),
]
_has_tuning_data = all(eval_metrics.get(b) and eval_metrics.get(t) for _, b, t in _tuning_rows)
if _has_tuning_data:
    with st.expander("Tuning Improvement — before vs. after hyperparameter search"):
        st.caption(
            "Each XGBoost forecaster was re-evaluated on the same 30-day holdout using "
            "fixed default params (Default) and time-series CV-tuned params (Tuned). "
            "Lower MAPE is better."
        )
        _table_data = []
        for label, baseline_key, tuned_key in _tuning_rows:
            b_mape = eval_metrics[baseline_key]["mape"]
            t_mape = eval_metrics[tuned_key]["mape"]
            gain = t_mape - b_mape
            _table_data.append({
                "Series": label,
                "Default MAPE": f"{b_mape:.1f}%",
                "Tuned MAPE": f"{t_mape:.1f}%",
                "Gain (pts)": f"{gain:+.1f}",
            })
        _tuning_df = pd.DataFrame(_table_data)

        def _color_gain(val):
            try:
                v = float(val.replace("+", ""))
                color = "green" if v < 0 else ("red" if v > 0 else "gray")
            except Exception:
                color = "gray"
            return f"color: {color}; font-weight: bold"

        st.dataframe(
            _tuning_df.style.applymap(_color_gain, subset=["Gain (pts)"]),
            use_container_width=True,
            hide_index=True,
        )

st.divider()

st.subheader(forecast_title)
history_window = series.iloc[-90:]
show_ci = False
band_df = None

# Build otb_30d and otb_60d future features for inference.
# otb_30d: active bookings as of snapshot  → matches training snapshot at d-30
# otb_60d: active bookings as of snapshot-30 → matches training snapshot at d-60
_future_dates_30 = pd.date_range(snapshot_date + pd.Timedelta(days=1), periods=30, freq="D")
_snapshot_ts = pd.Timestamp(snapshot_date)
_snapshot_60d_ts = _snapshot_ts - pd.Timedelta(days=30)

def _count_active_otb(data, as_of_ts, future_dates):
    if "reservation_status_date" in data.columns:
        _a = data[
            (data["is_canceled"] == 0) |
            ((data["is_canceled"] == 1) & (data["reservation_status_date"] > as_of_ts))
        ]
    else:
        _a = data[data["is_canceled"] == 0]
    return _a.groupby("arrival_date").size().reindex(future_dates, fill_value=0)

_otb_30d_counts = _count_active_otb(_df_for_pickup, _snapshot_ts,    _future_dates_30)
_otb_60d_counts = _count_active_otb(_df_for_pickup, _snapshot_60d_ts, _future_dates_30)
_future_otb = pd.DataFrame(
    {"otb_30d": _otb_30d_counts.values, "otb_60d": _otb_60d_counts.values},
    index=_future_dates_30,
)

# Weather features for the 30-day forecast window (Live API for days 1-16, climatology beyond)
try:
    _weather_future = build_inference_weather(_future_dates_30)
    if not _weather_future.empty:
        _future_otb = pd.concat([_future_otb, _weather_future.reindex(_future_dates_30).fillna(0)], axis=1)
except Exception as _we:
    pass  # forecast proceeds without weather features if API unavailable

if model_choice == "XGBoost":
    direct_models = models.get(direct_key)
    interval_models = models.get(interval_key)
    if direct_models:
        # Direct multi-step: each horizon uses actual historical values, no error accumulation
        forecast_30 = forecast_xgb_direct(
            direct_models, series, holiday_country=APP_CONFIG.get("holiday_country")
        )
        if interval_models:
            band_df = forecast_xgb_with_intervals(interval_models, series, periods=30,
                                                  future_features=_future_otb)
            band_df["ds"] = pd.to_datetime(band_df["ds"])
            show_ci = True
    elif interval_models:
        band_df = forecast_xgb_with_intervals(interval_models, series, periods=30,
                                              future_features=_future_otb)
        band_df["ds"] = pd.to_datetime(band_df["ds"])
        forecast_30 = band_df.set_index("ds")["yhat"].rename("forecast")
        show_ci = True
    else:
        forecast_30 = forecast_xgb(xgb_model, feat_cols, series, periods=30,
                                   future_features=_future_otb)
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
    _df_for_pickup,
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
# Slice the bookings history to the snapshot date so the pricing forecast
# covers the same window as the pickup report (snapshot+1 … snapshot+30).
bookings_to_snapshot = daily_df["daily_bookings"].loc[: snapshot_date]
if len(bookings_to_snapshot) < 2:
    bookings_to_snapshot = daily_df["daily_bookings"]
if demand_bands:
    demand_forecast_df = forecast_xgb_with_intervals(demand_bands, bookings_to_snapshot, periods=30,
                                                     future_features=_future_otb)
else:
    point_forecast = forecast_xgb(models["xgb_demand"][0], models["xgb_demand"][1],
                                  bookings_to_snapshot, periods=30, future_features=_future_otb)
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
