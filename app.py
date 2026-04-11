"""
Hotel Analytics Dashboard — Streamlit entry point.
Run with: streamlit run app.py

Pages are in the pages/ directory (auto-discovered by Streamlit).
This file defines the shared cached loaders used by all pages.
Shared business logic lives in src/services.py (no Streamlit dependency).
"""
import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
import src.services as _svc

DATA_PATH = "hotel_bookings.csv"
APP_MODE = os.getenv("APP_MODE", "demo")


# ---------------------------------------------------------------------------
# Shared cached loaders — imported by every page
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading and preprocessing data...")
def load_and_prepare_data(data_path: str = DATA_PATH):
    """
    Thin Streamlit-cached wrapper around src.services.load_and_prepare_data.
    Returns: (df, daily_df)
    """
    return _svc.load_and_prepare_data(data_path)


@st.cache_resource(show_spinner="Initializing models (first run trains everything — ~5 min)...")
def get_or_train_models(data_path: str = DATA_PATH):
    """
    Thin Streamlit-cached wrapper around src.services.get_or_train_models.
    Returns a dict with all ML artifacts and eval_metrics.
    """
    return _svc.get_or_train_models(data_path)


# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Hotel Analytics",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

if APP_MODE == "demo":
    st.info("📊 You're viewing demo data from a sample hotel. Upload your own data to see your hotel's insights.")

st.title("🏨 Hotel Analytics Dashboard")
st.markdown(
    """
    An interactive ML-powered analytics platform built on hotel booking data (119k+ records, 2015–2017).

    **Use the sidebar to navigate between pages:**

    | Page | Description |
    |------|-------------|
    | **Overview** | KPIs, revenue & bookings charts, hotel/date filters |
    | **Revenue Forecast** | XGBoost vs SARIMA vs naive baseline, cross-validation |
    | **Demand & Staffing** | 30-day demand forecast + staffing recommendations |
    | **Cancellation Predictor** | Input a booking → get cancellation probability |
    | **Profit Simulator** | Adjust tax/cost rates → see 30-day profit breakdown |
    """
)

st.info("Models load from disk on startup. First run trains everything (~5 min). Subsequent runs start in ~2s.")

# Warm up the cache so pages load instantly
with st.spinner("Loading data and models..."):
    load_and_prepare_data()
    get_or_train_models()

st.success("Ready! Select a page from the sidebar.")
