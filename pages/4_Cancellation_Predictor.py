"""
Page 4 — Cancellation Predictor
Interactive form → cancellation probability + risk category.
Shows only SHAP-selected features; remaining features use training-set defaults.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import get_or_train_models
from src.cancellation import predict_cancellation_proba

st.set_page_config(page_title="Cancellation Predictor", page_icon="🔮", layout="wide")
st.title("🔮 Cancellation Risk Predictor")
st.markdown(
    "Fill in the booking details below to get a machine-learning prediction of "
    "whether this booking is likely to be cancelled."
)

models = get_or_train_models()
cancel_models    = models["cancel_models"]
feature_columns  = models["feature_columns"]
label_encoders   = models["label_encoders"]
feature_defaults = models["feature_defaults"]
selected_features = models.get("selected_features") or feature_columns

# ---------------------------------------------------------------------------
# Widget spec — label, widget type, kwargs for each possible column
# ---------------------------------------------------------------------------
_WIDGET_SPEC = {
    "lead_time":              ("Lead Time (days)",         "number", dict(min_value=0, max_value=737, value=30)),
    "adr":                    ("Avg Daily Rate (€)",        "number", dict(min_value=0.0, max_value=5000.0, value=100.0, step=10.0)),
    "total_nights":           ("Total Nights",             "number", dict(min_value=0, max_value=50, value=3)),
    "total_guests":           ("Total Guests",             "number", dict(min_value=1, max_value=20, value=2)),
    "hotel":                  ("Hotel Type",               "select", dict(options=["City Hotel", "Resort Hotel"])),
    "market_segment":         ("Market Segment",           "select", dict(options=["Aviation","Complementary","Corporate","Direct","Groups","Offline TA/TO","Online TA","Undefined"])),
    "meal":                   ("Meal Plan",                "select", dict(options=["BB","FB","HB","SC","Undefined"])),
    "customer_type":          ("Customer Type",            "select", dict(options=["Contract","Group","Transient","Transient-Party"])),
    "distribution_channel":   ("Distribution Channel",    "select", dict(options=["Corporate","Direct","GDS","TA/TO","Undefined"])),
    "reserved_room_type":     ("Reserved Room Type",       "select", dict(options=["A","B","C","D","E","F","G","H","L","P"])),
    "is_repeated_guest":      ("Repeated Guest",           "number", dict(min_value=0, max_value=1, value=0)),
    "previous_cancellations": ("Prior Cancellations",      "number", dict(min_value=0, max_value=30, value=0)),
    "previous_bookings_not_canceled": ("Prior Completed Bookings", "number", dict(min_value=0, max_value=72, value=0)),
    "total_of_special_requests": ("Special Requests",     "number", dict(min_value=0, max_value=5, value=0)),
    "booking_month":          ("Booking Month",            "number", dict(min_value=1, max_value=12, value=6)),
    "season":                 ("Season (0=spr 1=sum 2=fall 3=win)", "number", dict(min_value=0, max_value=3, value=1)),
    "is_weekend_arrival":     ("Weekend Arrival",          "number", dict(min_value=0, max_value=1, value=0)),
    "adults":                 ("Adults",                   "number", dict(min_value=1, max_value=20, value=2)),
    "children":               ("Children",                 "number", dict(min_value=0, max_value=10, value=0)),
    "babies":                 ("Babies",                   "number", dict(min_value=0, max_value=5, value=0)),
    "stays_in_weekend_nights":("Weekend Nights",           "number", dict(min_value=0, max_value=20, value=1)),
    "stays_in_week_nights":   ("Week Nights",              "number", dict(min_value=0, max_value=50, value=2)),
}

# Show up to 12 selected features in the form
displayed_features = [f for f in selected_features if f in _WIDGET_SPEC][:12]
if not displayed_features:
    displayed_features = [f for f in feature_columns if f in _WIDGET_SPEC][:12]


def _encode_cat(col: str, value: str) -> float:
    le = label_encoders.get(col)
    if le is None:
        return 0.0
    try:
        return float(le.transform([value])[0])
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Form
# ---------------------------------------------------------------------------
with st.form("cancellation_form"):
    st.subheader("Booking Details")
    st.caption(f"Showing {len(displayed_features)} most predictive features (selected by SHAP analysis).")

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

# ---------------------------------------------------------------------------
# Prediction output
# ---------------------------------------------------------------------------
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

    # SHAP top-3 risk factors for this specific prediction (XGBoost only)
    if model_name == "xgboost":
        with st.expander("Top risk factors for this prediction (SHAP)"):
            try:
                import shap
                xgb_model = cancel_models["xgboost"]
                explainer = shap.TreeExplainer(xgb_model)
                sv = explainer.shap_values(input_vec)
                if isinstance(sv, list):
                    sv = sv[1]
                import pandas as pd
                shap_row = pd.DataFrame({
                    "Feature": feature_columns,
                    "SHAP Value": sv[0],
                }).reindex(pd.Series(sv[0]).abs().sort_values(ascending=False).index)
                shap_row = shap_row.head(3)
                shap_row["Direction"] = shap_row["SHAP Value"].apply(
                    lambda v: "↑ Increases risk" if v > 0 else "↓ Decreases risk"
                )
                st.dataframe(
                    shap_row[["Feature", "Direction", "SHAP Value"]]
                    .style.format({"SHAP Value": "{:.4f}"}),
                    use_container_width=True,
                )
            except Exception as e:
                st.info(f"SHAP explanation unavailable: {e}")

    with st.expander("What drives cancellation risk?"):
        st.markdown("""
        **Key factors typically associated with higher cancellation probability:**
        - **Long lead time** — bookings made far in advance cancel more often
        - **Online TA segment** — highest cancellation rates in the dataset (~40%)
        - **No previous completed bookings** — first-time guests cancel more
        - **High ADR** — premium bookings may be cancelled when better deals arise
        """)
