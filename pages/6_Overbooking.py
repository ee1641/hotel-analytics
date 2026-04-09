"""
Page 6 — Overbooking Recommendation Engine
Date selector + cost inputs → optimal overbook count + risk table.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import load_and_prepare_data, get_or_train_models
from src.overbooking import calculate_overbooking_recommendation
from src.booking_risk_scorer import score_bookings

st.set_page_config(page_title="Overbooking", page_icon="🏨", layout="wide")
st.title("🏨 Overbooking Recommendation Engine")
st.markdown(
    "Use your cancellation model to find the optimal number of rooms to overbook "
    "for any arrival date — balancing walk costs against lost revenue from no-shows."
)

try:
    with st.spinner("Loading models..."):
        df, _ = load_and_prepare_data()
        models = get_or_train_models()
except Exception as e:
    st.error(f"Failed to load data or models: {e}")
    st.stop()

cancel_model = models["cancel_models"]["xgboost"]

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Parameters")

available_dates = sorted(df["arrival_date"].dt.date.unique())
mid = len(available_dates) // 2
target_date = st.sidebar.date_input(
    "Arrival Date to Analyse",
    value=available_dates[mid],
    min_value=available_dates[0],
    max_value=available_dates[-1],
)
total_rooms     = st.sidebar.number_input("Total Rooms",                   min_value=10,  max_value=5000, value=100)
overbooking_cost = st.sidebar.number_input("Walk Cost per Guest (€)",      min_value=0,   max_value=5000, value=150)
empty_room_cost  = st.sidebar.number_input("Lost Revenue per Empty Room (€)", min_value=0, max_value=5000, value=100)

# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------
rec = calculate_overbooking_recommendation(
    df, cancel_model, target_date, total_rooms,
    overbooking_cost=float(overbooking_cost),
    empty_room_cost=float(empty_room_cost),
)

_RISK_ICON = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
col1, col2, col3, col4 = st.columns(4)
col1.metric("Expected Cancellations", f"{rec['expected_cancellations']:.1f}")
col2.metric("Recommended Overbooks",  str(rec["recommended_overbooks"]))
col3.metric("Risk Level", f"{_RISK_ICON[rec['risk_level']]} {rec['risk_level']}")
col4.metric("Revenue Impact",         f"€{rec['expected_revenue_impact']:,.0f}")

ci_low, ci_high = rec["confidence_interval"]
st.caption(
    f"80% confidence interval for cancellations: [{ci_low:.0f} – {ci_high:.0f}]"
)

st.divider()

# ---------------------------------------------------------------------------
# What-if slider
# ---------------------------------------------------------------------------
st.subheader("What-If Analysis")
manual_overbook = st.slider(
    "Manually adjust overbooking count",
    min_value=0, max_value=20,
    value=rec["recommended_overbooks"],
)

day_bookings = df[
    (df["arrival_date"].dt.date == target_date) &
    (df["is_canceled"] == 0)
]
avg_adr = float(day_bookings["adr"].mean()) if len(day_bookings) > 0 and "adr" in day_bookings.columns else float(empty_room_cost)
lam = rec["expected_cancellations"]
manual_revenue = (manual_overbook * avg_adr) - (lam * float(overbooking_cost))
st.info(
    f"At **{manual_overbook} overbook(s)**: estimated revenue impact = **€{manual_revenue:,.0f}** "
    f"(avg ADR €{avg_adr:.0f}, expected {lam:.1f} cancellations)"
)

st.divider()

# ---------------------------------------------------------------------------
# High-risk bookings table
# ---------------------------------------------------------------------------
st.subheader(f"Highest-Risk Bookings for {target_date}")
if len(day_bookings) == 0:
    st.info("No non-cancelled bookings found for this date in the dataset.")
else:
    try:
        scored = score_bookings(day_bookings, cancel_model)
        display_cols = [c for c in ["cancel_probability", "risk_tier", "lead_time", "adr", "market_segment", "customer_type"] if c in scored.columns]
        st.dataframe(
            scored[display_cols].head(25).style.format(
                {c: "{:.1%}" if c == "cancel_probability" else ("€{:.0f}" if c == "adr" else "{}") for c in display_cols}
            ),
            use_container_width=True,
        )
        st.caption(f"{len(day_bookings)} active bookings for this date. Showing top 25 by risk.")
    except Exception as e:
        st.warning(f"Could not score bookings: {e}")
