"""
Page 4 - Cancellation Predictor
Cancellation probability plus time-to-cancellation survival view.
"""
import warnings

warnings.filterwarnings("ignore")

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import get_or_train_models
from src.cancellation import (
    get_survival_curve,
    predict_booking_survival_profile,
    predict_cancellation_proba,
)

st.set_page_config(page_title="Cancellation Predictor", page_icon="🔮", layout="wide")
st.title("🔮 Cancellation Risk Predictor")
st.markdown(
    "Score a booking for cancellation risk and estimate how likely it is to remain alive as arrival approaches."
)

models = get_or_train_models()
cancel_models = models["cancel_models"]
survival_model = models.get("cancel_survival_model")
feature_columns = models["feature_columns"]
label_encoders = models["label_encoders"]
feature_defaults = models["feature_defaults"]
selected_features = models.get("selected_features") or feature_columns

_WIDGET_SPEC = {
    "lead_time": ("Lead Time (days)", "number", dict(min_value=0, max_value=737, value=30)),
    "adr": ("Avg Daily Rate (€)", "number", dict(min_value=0.0, max_value=5000.0, value=100.0, step=10.0)),
    "total_nights": ("Total Nights", "number", dict(min_value=1, max_value=50, value=3)),
    "total_guests": ("Total Guests", "number", dict(min_value=1, max_value=20, value=2)),
    "hotel": ("Hotel Type", "select", dict(options=["City Hotel", "Resort Hotel"])),
    "market_segment": (
        "Market Segment",
        "select",
        dict(options=["Aviation", "Complementary", "Corporate", "Direct", "Groups", "Offline TA/TO", "Online TA", "Undefined"]),
    ),
    "deposit_type": ("Deposit Type", "select", dict(options=["No Deposit", "Non Refund", "Refundable"])),
    "meal": ("Meal Plan", "select", dict(options=["BB", "FB", "HB", "SC", "Undefined"])),
    "customer_type": ("Customer Type", "select", dict(options=["Contract", "Group", "Transient", "Transient-Party"])),
    "distribution_channel": ("Distribution Channel", "select", dict(options=["Corporate", "Direct", "GDS", "TA/TO", "Undefined"])),
    "reserved_room_type": ("Reserved Room Type", "select", dict(options=["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"])),
    "is_repeated_guest": ("Repeated Guest", "number", dict(min_value=0, max_value=1, value=0)),
    "previous_cancellations": ("Prior Cancellations", "number", dict(min_value=0, max_value=30, value=0)),
    "previous_bookings_not_canceled": ("Prior Completed Bookings", "number", dict(min_value=0, max_value=72, value=0)),
    "total_of_special_requests": ("Special Requests", "number", dict(min_value=0, max_value=5, value=0)),
    "booking_month": ("Booking Month", "number", dict(min_value=1, max_value=12, value=6)),
    "season": ("Season (0=spring, 1=summer, 2=fall, 3=winter)", "number", dict(min_value=0, max_value=3, value=1)),
    "is_weekend_arrival": ("Weekend Arrival", "number", dict(min_value=0, max_value=1, value=0)),
    "adults": ("Adults", "number", dict(min_value=1, max_value=20, value=2)),
    "children": ("Children", "number", dict(min_value=0, max_value=10, value=0)),
    "babies": ("Babies", "number", dict(min_value=0, max_value=5, value=0)),
    "stays_in_weekend_nights": ("Weekend Nights", "number", dict(min_value=0, max_value=20, value=1)),
    "stays_in_week_nights": ("Week Nights", "number", dict(min_value=0, max_value=50, value=2)),
}

displayed_features = [f for f in selected_features if f in _WIDGET_SPEC][:12]
for extra in ["deposit_type", "market_segment", "total_nights", "lead_time", "adr"]:
    if extra not in displayed_features and extra in _WIDGET_SPEC:
        displayed_features.append(extra)


def _encode_cat(col: str, value: str) -> float:
    le = label_encoders.get(col)
    if le is None:
        return 0.0
    try:
        return float(le.transform([value])[0])
    except ValueError:
        return 0.0


with st.form("cancellation_form"):
    st.subheader("Booking Details")
    st.caption("The binary model predicts cancel / not-cancel. The survival model estimates when cancellations are likely to happen.")

    cols = st.columns(3)
    form_values = {}
    for i, feat in enumerate(displayed_features):
        label, wtype, kwargs = _WIDGET_SPEC[feat]
        with cols[i % 3]:
            if wtype == "number":
                form_values[feat] = st.number_input(label, **kwargs, key=feat)
            else:
                form_values[feat] = st.selectbox(label, **kwargs, key=feat)

    st.subheader("Select Classifier")
    model_name = st.selectbox("Model", ["xgboost", "random_forest", "logistic"])
    submitted = st.form_submit_button("Predict Cancellation Risk", type="primary")

if submitted:
    overrides = {}
    for feat, val in form_values.items():
        spec = _WIDGET_SPEC.get(feat)
        if spec and spec[1] == "select":
            overrides[feat] = _encode_cat(feat, val)
        else:
            overrides[feat] = float(val)

    row = dict(feature_defaults)
    row.update(overrides)
    input_vec = np.array([[row.get(col, 0.0) for col in feature_columns]], dtype=float)

    selected_model = cancel_models[model_name]
    prob = float(predict_cancellation_proba(selected_model, input_vec)[0])

    if prob < 0.35:
        risk, color, emoji = "Low", "green", "✅"
    elif prob < 0.65:
        risk, color, emoji = "Medium", "orange", "⚠️"
    else:
        risk, color, emoji = "High", "red", "🚨"

    st.divider()
    st.subheader("Prediction Result")

    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.metric("Cancellation Probability", f"{prob:.1%}")
        st.markdown(f"**Risk Category:** :{color}[{emoji} {risk}]")
    with res_col2:
        st.progress(prob)
        if risk == "Low":
            st.success("Low cancellation risk. Standard processing recommended.")
        elif risk == "Medium":
            st.warning("Moderate cancellation risk. Consider a follow-up or deposit.")
        else:
            st.error("High cancellation risk. Consider overbooking buffer or stricter deposit policy.")

    if survival_model is not None:
        survival_input = {
            "lead_time": float(form_values.get("lead_time", 30)),
            "adr": float(form_values.get("adr", 100.0)),
            "total_nights": float(form_values.get("total_nights", 3)),
            "market_segment": form_values.get("market_segment", "Online TA"),
            "deposit_type": form_values.get("deposit_type", "No Deposit"),
            "stays_in_weekend_nights": float(form_values.get("stays_in_weekend_nights", 1)),
            "stays_in_week_nights": float(form_values.get("stays_in_week_nights", 2)),
        }
        survival_profile = predict_booking_survival_profile(survival_model, survival_input)
        baseline_curve = get_survival_curve(survival_model)

        st.divider()
        st.subheader("Cancellation Survival Profile")
        surv_cols = st.columns(len(survival_profile))
        for idx, row in survival_profile.iterrows():
            surv_cols[idx].metric(
                f"Alive {int(row['days_before_arrival'])}d Before",
                f"{row['survival_probability']:.1%}",
            )

        chart = go.Figure()
        chart.add_trace(
            go.Scatter(
                x=baseline_curve["timeline"],
                y=baseline_curve["survival_probability"],
                name="Portfolio KM Curve",
                line=dict(color="steelblue", width=2),
            )
        )
        chart.add_trace(
            go.Scatter(
                x=(float(form_values.get("lead_time", 30)) - survival_profile["days_before_arrival"]).clip(lower=0),
                y=survival_profile["survival_probability"],
                mode="lines+markers",
                name="This Booking",
                line=dict(color="tomato", width=3),
            )
        )
        chart.update_layout(
            hovermode="x unified",
            xaxis_title="Days Since Booking",
            yaxis_title="Probability Booking Is Still Alive",
            height=360,
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(chart, use_container_width=True)
        st.dataframe(survival_profile, use_container_width=True)

    if model_name == "xgboost":
        with st.expander("Top risk factors for this prediction (SHAP)"):
            try:
                import shap

                xgb_model = cancel_models["xgboost"]
                explainer = shap.TreeExplainer(xgb_model)
                sv = explainer.shap_values(input_vec)
                if isinstance(sv, list):
                    sv = sv[1]
                shap_row = pd.DataFrame(
                    {
                        "Feature": feature_columns,
                        "SHAP Value": sv[0],
                    }
                ).reindex(pd.Series(sv[0]).abs().sort_values(ascending=False).index)
                shap_row = shap_row.head(3)
                shap_row["Direction"] = shap_row["SHAP Value"].apply(
                    lambda v: "↑ Increases risk" if v > 0 else "↓ Decreases risk"
                )
                st.dataframe(
                    shap_row[["Feature", "Direction", "SHAP Value"]].style.format({"SHAP Value": "{:.4f}"}),
                    use_container_width=True,
                )
            except Exception as e:
                st.info(f"SHAP explanation unavailable: {e}")
