"""
Page 8 — Landing / About
Sales landing page with value props and demo CTA.
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.analytics_tracker import track_event

st.set_page_config(page_title="Hotel Analytics", page_icon="🏨", layout="wide")
track_event("page_view", "landing")

# Hero
st.markdown("""
<div style="text-align:center;padding:60px 20px;background:linear-gradient(135deg,#1a1a2e,#16213e);
            border-radius:12px;margin-bottom:32px;">
  <h1 style="color:white;font-size:2.5em;margin-bottom:16px;">
    Stop losing revenue to cancellations.
  </h1>
  <p style="color:#aaa;font-size:1.2em;max-width:600px;margin:0 auto 24px;">
    Know which bookings will cancel before they do — and turn that knowledge into revenue.
  </p>
</div>
""", unsafe_allow_html=True)

# Value props
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### 🎯 Cancellation Prediction")
    st.write("Know your risk before it's a loss. ML models score every booking with cancellation probability and top risk factors.")
with col2:
    st.markdown("### 📈 Smart Overbooking")
    st.write("Fill every room. Our Newsvendor-optimized engine tells you exactly how many rooms to overbook for each date.")
with col3:
    st.markdown("### 📊 Demand Forecasting")
    st.write("Staff and price with confidence. Prophet + XGBoost models forecast demand 30 days ahead with confidence bands.")

st.divider()

# CTA buttons
cta_col1, cta_col2 = st.columns(2)
with cta_col1:
    if st.button("Try the Demo →", type="primary", use_container_width=True):
        track_event("button_click", "landing", {"button": "try_demo"})
        st.switch_page("pages/1_Overview.py")
with cta_col2:
    if st.button("Get Started (Contact Us)", use_container_width=True):
        track_event("button_click", "landing", {"button": "get_started"})
        st.info("Email us at: analytics@yourhotel.com")

st.divider()
st.markdown("*Demo data from a sample hotel in Portugal. Upload your own CSV to see your hotel's insights.*")
