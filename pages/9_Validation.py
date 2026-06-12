"""
Page 9 - Model Validation
Blind evaluation of the trained cancellation model against an external labelled dataset.
Upload any bookings CSV that contains is_canceled; predictions are compared row-by-row.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src import model_cache
from src.cancellation import (
    apply_probability_calibrator,
    predict_cancellation_proba,
    predict_stacked_proba,
)
from src.cancellation_evaluation import _align_to_feature_columns
from src.evaluation import calibration_metrics, classification_report_dict
from src.ingestion import normalize_uploaded_dataframe
from src.preprocessing import (
    add_temporal_features,
    clean_data,
    drop_leaky_features,
    engineer_features,
)
from src.time_series import build_arrival_date

st.set_page_config(page_title="Model Validation", page_icon="🧪", layout="wide")
st.title("🧪 Model Validation")

# ---------------------------------------------------------------------------
# Section 0 — Forecasting model MAPE (no upload required)
# ---------------------------------------------------------------------------
_eval_metrics = model_cache.load_artifact("eval_metrics")
if _eval_metrics:
    st.subheader("Forecasting Model MAPE")
    st.caption(
        "30-day holdout evaluation from training. Combined = all hotel types. "
        "Per-type models use the same holdout period on their respective segments."
    )
    _rows = []
    for _label, _combined_key, _resort_key, _city_key in [
        ("Demand",            "demand_xgb",           "demand_xgb_resort_hotel",            "demand_xgb_city_hotel"),
        ("Booked Revenue",    "revenue_booked_xgb",   "revenue_booked_xgb_resort_hotel",    "revenue_booked_xgb_city_hotel"),
        ("Realized Revenue",  "revenue_realized_xgb", "revenue_realized_xgb_resort_hotel",  "revenue_realized_xgb_city_hotel"),
        ("Expected Realized", "revenue_expected_xgb", "revenue_expected_xgb_resort_hotel",  "revenue_expected_xgb_city_hotel"),
    ]:
        def _fmt(key: str) -> str:
            v = _eval_metrics.get(key, {})
            if not v:
                return "—"
            mape = v.get("mape", float("nan"))
            return f"{mape:.1f}%" if mape == mape else "—"  # noqa: PLR0124

        _rows.append({
            "Series": _label,
            "Combined": _fmt(_combined_key),
            "Resort Hotel": _fmt(_resort_key),
            "City Hotel": _fmt(_city_key),
        })
    st.dataframe(pd.DataFrame(_rows), width='stretch', hide_index=True)

    # Conformal interval coverage — how often the predicted band actually
    # contains the truth on the same 30-day holdout. The nominal target is
    # 1 - alpha (default 90%).
    _coverage_rows = []
    for _series_label, _key in [("Booked Revenue", "revenue_conformal_coverage"),
                                 ("Demand",         "demand_conformal_coverage")]:
        _block = _eval_metrics.get(_key) or {}
        if not _block:
            continue
        _coverage_rows.append({
            "Series": _series_label,
            "Nominal":   f"{_block.get('nominal_coverage', float('nan')):.0%}",
            "Empirical": f"{_block.get('coverage', float('nan')):.0%}",
            "Mean Width": f"{_block.get('mean_width', float('nan')):,.1f}",
            "N":         int(_block.get("n", 0)),
        })
    if _coverage_rows:
        st.subheader("Forecast Interval Coverage")
        st.caption(
            "Split-conformal intervals on the 30-day holdout. Empirical coverage should sit "
            "near the nominal target (90% by default); a gap signals miscalibration."
        )
        st.dataframe(pd.DataFrame(_coverage_rows), width='stretch', hide_index=True)
    st.divider()

# ---------------------------------------------------------------------------
# Section 0b — Metrics over time (one snapshot per scripts/retrain.py run)
# ---------------------------------------------------------------------------
_history_file = "data/metrics_history.jsonl"
try:
    _history = (
        pd.read_json(_history_file, lines=True)
        if os.path.exists(_history_file) else pd.DataFrame()
    )
except Exception:
    _history = pd.DataFrame()

if not _history.empty and "timestamp" in _history.columns:
    st.subheader("Model Metrics Over Time")
    st.caption(
        "One snapshot per `scripts/retrain.py` run. Rising MAPE or coverage below the "
        "0.85/0.80 bands triggers a model_drift alert on the Alerts page."
    )
    _history["timestamp"] = pd.to_datetime(_history["timestamp"], errors="coerce", utc=True)
    _history = _history.dropna(subset=["timestamp"]).sort_values("timestamp")

    _hist_col1, _hist_col2 = st.columns(2)
    with _hist_col1:
        _mape_fig = go.Figure()
        for _col, _label in [("demand_mape", "Demand MAPE"), ("revenue_mape", "Revenue MAPE")]:
            if _col in _history.columns:
                _mape_fig.add_trace(go.Scatter(
                    x=_history["timestamp"], y=_history[_col],
                    mode="lines+markers", name=_label,
                ))
        _mape_fig.update_layout(
            yaxis_title="MAPE (%)", height=320,
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(_mape_fig, width='stretch')
    with _hist_col2:
        _cov_fig = go.Figure()
        for _col, _label in [("revenue_coverage", "Revenue coverage"),
                             ("demand_coverage", "Demand coverage")]:
            if _col in _history.columns:
                _cov_fig.add_trace(go.Scatter(
                    x=_history["timestamp"], y=_history[_col],
                    mode="lines+markers", name=_label,
                ))
        _cov_fig.add_hline(y=0.90, line_dash="dash", line_color="gray",
                           annotation_text="nominal 90%")
        _cov_fig.add_hline(y=0.85, line_dash="dot", line_color="orange")
        _cov_fig.add_hline(y=0.80, line_dash="dot", line_color="red")
        _cov_fig.update_layout(
            yaxis_title="Interval coverage", yaxis=dict(range=[0.5, 1.02]), height=320,
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(_cov_fig, width='stretch')

    _last_drift = _history.iloc[-1].get("drift") if "drift" in _history.columns else None
    if isinstance(_last_drift, list) and _last_drift:
        for _f in _last_drift:
            _msg = _f.get("message", str(_f)) if isinstance(_f, dict) else str(_f)
            _sev = _f.get("severity", "warning") if isinstance(_f, dict) else "warning"
            (st.error if _sev == "critical" else st.warning)(f"Drift: {_msg}")
    else:
        st.success("No drift detected in the latest retrain snapshot.")
    st.divider()

st.caption(
    "Upload a bookings CSV with known outcomes to run a blind test of the trained "
    "cancellation model. Predictions are generated without using the `is_canceled` column, "
    "then compared to it."
)

# ---------------------------------------------------------------------------
# Section 1 — File upload
# ---------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload bookings CSV",
    type=["csv"],
    help="Must contain an `is_canceled` column (0 = not canceled, 1 = canceled). "
         "All other columns are optional — missing features are filled with training-set defaults.",
)

if uploaded is None:
    st.info(
        "**What to upload:** any hotel bookings CSV that has an `is_canceled` column. "
        "Column order doesn't matter. Missing columns are handled automatically.\n\n"
        "**What you'll see:**\n"
        "- Aggregate metrics (Accuracy, Precision, Recall, F1, ROC-AUC) for each model\n"
        "- Confusion matrix and probability distribution chart\n"
        "- Row-level table: every booking with predicted probability, prediction, and actual outcome"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Section 2 — Load trained artifacts
# ---------------------------------------------------------------------------
cancel_models = model_cache.load_artifact("cancel_models")
feature_columns = model_cache.load_artifact("feature_columns")
label_encoders = model_cache.load_artifact("label_encoders")
feature_defaults = model_cache.load_artifact("feature_defaults")

if any(x is None for x in [cancel_models, feature_columns, label_encoders, feature_defaults]):
    st.error(
        "Trained models not found in cache. Visit any other page first to trigger training, "
        "then return here."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Section 3 — Ingest and preprocess
# ---------------------------------------------------------------------------
try:
    raw_df = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

df = normalize_uploaded_dataframe(raw_df.copy())

if "is_canceled" not in df.columns:
    found = ", ".join(f"`{c}`" for c in df.columns.tolist())
    st.error(
        f"The uploaded file does not contain an `is_canceled` column (checked after column "
        f"normalization). Found columns: {found}"
    )
    st.stop()

# Normalize the target column
df["is_canceled"] = pd.to_numeric(df["is_canceled"], errors="coerce")
df = df.dropna(subset=["is_canceled"])
df["is_canceled"] = df["is_canceled"].astype(int)

if len(df) == 0:
    st.warning("No valid rows remain after cleaning the `is_canceled` column.")
    st.stop()

try:
    df = clean_data(df)
    df = engineer_features(df)
    if "arrival_date" in df.columns:
        df = build_arrival_date(df)
        df = add_temporal_features(df)
except Exception as exc:
    st.warning(f"Preprocessing warning (continuing with available features): {exc}")

# ---------------------------------------------------------------------------
# Section 4 — Column coverage summary
# ---------------------------------------------------------------------------
df_for_features = drop_leaky_features(df)
present = [c for c in feature_columns if c in df_for_features.columns]
defaulted = [c for c in feature_columns if c not in df_for_features.columns]

with st.expander(
    f"Feature coverage — {len(present)}/{len(feature_columns)} training features found in upload",
    expanded=False,
):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Found in upload ({len(present)})**")
        st.write(present)
    with col_b:
        st.markdown(f"**Filled with training defaults ({len(defaulted)})**")
        if defaulted:
            st.write(defaulted)
        else:
            st.write("None — all features present.")

if defaulted:
    st.info(
        f"{len(defaulted)} feature(s) not in the uploaded file will use the training-set "
        "median/mode as a substitute. Predictions are still valid but may be less precise "
        "for those dimensions."
    )

# ---------------------------------------------------------------------------
# Section 5 — Build feature matrix and run predictions
# ---------------------------------------------------------------------------
X = _align_to_feature_columns(df_for_features, feature_columns, label_encoders, feature_defaults)
y_true = df["is_canceled"].values

# Only iterate fitted base models — skip meta-learners, calibrators, encoding maps.
from src.cancellation import _NON_MODEL_KEYS

model_results = {}
for name, model in cancel_models.items():
    if name in _NON_MODEL_KEYS:
        continue
    try:
        prob = predict_cancellation_proba(model, X)
        pred = (prob >= 0.5).astype(int)
        try:
            metrics = classification_report_dict(y_true, pred, prob)
        except Exception:
            metrics = {
                "accuracy": float(np.mean(pred == y_true)),
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "roc_auc": float("nan"),
            }
        model_results[name] = {"prob": prob, "pred": pred, "metrics": metrics}
    except Exception as exc:
        model_results[name] = {"error": str(exc)}

# Add the stacked + calibrated path explicitly so the reliability panel can compare it.
if "stacked_meta" in cancel_models:
    try:
        stacked_prob = predict_stacked_proba(
            cancel_models["stacked_meta"],
            cancel_models,
            X,
            calibrator=cancel_models.get("probability_calibrator"),
        )
        thresh = float(cancel_models.get("optimal_threshold", 0.5))
        stacked_pred = (stacked_prob >= thresh).astype(int)
        try:
            stacked_metrics = classification_report_dict(y_true, stacked_pred, stacked_prob)
        except Exception:
            stacked_metrics = {
                "accuracy": float(np.mean(stacked_pred == y_true)),
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "roc_auc": float("nan"),
            }
        model_results["stacked_calibrated"] = {
            "prob": stacked_prob, "pred": stacked_pred, "metrics": stacked_metrics,
        }
    except Exception as exc:
        model_results["stacked_calibrated"] = {"error": str(exc)}

n_total = len(y_true)
n_canceled = int(y_true.sum())
actual_rate = n_canceled / n_total if n_total else 0.0

st.divider()
st.subheader("Dataset Overview")
ov_cols = st.columns(3)
ov_cols[0].metric("Total Bookings", f"{n_total:,}")
ov_cols[1].metric("Actual Cancellations", f"{n_canceled:,}")
ov_cols[2].metric("Actual Cancellation Rate", f"{actual_rate:.1%}")

# ---------------------------------------------------------------------------
# Section 6 — Aggregate metrics per model
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Model Performance")

MODEL_DISPLAY = {
    "stacked_calibrated": ("Stacked (Calibrated)", True),
    "xgboost": ("XGBoost", False),
    "lightgbm": ("LightGBM", False),
    "catboost": ("CatBoost", False),
    "xgboost_dart": ("XGBoost DART", False),
    "random_forest": ("Random Forest", False),
    "logistic": ("Logistic Regression", False),
}

metric_cols = st.columns(len(model_results))
for col, (name, res) in zip(metric_cols, model_results.items()):
    display_name, is_primary = MODEL_DISPLAY.get(name, (name.title(), False))
    with col:
        label = f"**{display_name}**" + (" *(primary)*" if is_primary else "")
        st.markdown(label)
        if "error" in res:
            st.error(res["error"])
            continue
        m = res["metrics"]
        st.metric("Accuracy", f"{m['accuracy']:.1%}")
        st.metric("Precision", f"{m['precision']:.1%}")
        st.metric("Recall", f"{m['recall']:.1%}")
        st.metric("F1 Score", f"{m['f1']:.3f}")
        roc = m.get("roc_auc", float("nan"))
        st.metric("ROC-AUC", f"{roc:.3f}" if not (isinstance(roc, float) and np.isnan(roc)) else "N/A")

# ---------------------------------------------------------------------------
# Section 7 — Charts
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Diagnostic Charts")

xgb_res = model_results.get("xgboost", {})
if "error" not in xgb_res and xgb_res:
    chart_col1, chart_col2 = st.columns(2)

    # --- Confusion matrix ---
    with chart_col1:
        st.markdown("**Confusion Matrix — XGBoost**")
        pred_xgb = xgb_res["pred"]
        tp = int(((pred_xgb == 1) & (y_true == 1)).sum())
        fp = int(((pred_xgb == 1) & (y_true == 0)).sum())
        fn = int(((pred_xgb == 0) & (y_true == 1)).sum())
        tn = int(((pred_xgb == 0) & (y_true == 0)).sum())

        matrix = [[tn, fp], [fn, tp]]
        pct = [[tn / n_total, fp / n_total], [fn / n_total, tp / n_total]]
        labels = [["TN", "FP"], ["FN", "TP"]]
        text = [
            [f"{labels[i][j]}<br>{matrix[i][j]:,}<br>({pct[i][j]:.1%})" for j in range(2)]
            for i in range(2)
        ]

        cm_fig = go.Figure(
            go.Heatmap(
                z=matrix,
                x=["Predicted: Not Canceled", "Predicted: Canceled"],
                y=["Actual: Not Canceled", "Actual: Canceled"],
                text=text,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False,
            )
        )
        cm_fig.update_layout(height=340, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(cm_fig, width='stretch')

    # --- Probability distribution ---
    with chart_col2:
        st.markdown("**Predicted Probability Distribution — XGBoost**")
        prob_xgb = xgb_res["prob"]
        prob_canceled = prob_xgb[y_true == 1]
        prob_not_canceled = prob_xgb[y_true == 0]

        dist_fig = go.Figure()
        dist_fig.add_trace(
            go.Histogram(
                x=prob_not_canceled,
                name="Actual: Not Canceled",
                opacity=0.65,
                marker_color="steelblue",
                nbinsx=30,
            )
        )
        dist_fig.add_trace(
            go.Histogram(
                x=prob_canceled,
                name="Actual: Canceled",
                opacity=0.65,
                marker_color="tomato",
                nbinsx=30,
            )
        )
        dist_fig.add_vline(x=0.5, line_dash="dash", line_color="gray",
                           annotation_text="threshold", annotation_position="top right")
        dist_fig.update_layout(
            barmode="overlay",
            xaxis_title="Predicted Cancellation Probability",
            yaxis_title="Count",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=340,
            margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(dist_fig, width='stretch')

# ---------------------------------------------------------------------------
# Section 7b — Calibration / Reliability
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Probability Calibration")
st.caption(
    "Brier and ECE measure how well predicted probabilities match observed frequencies. "
    "The reliability curve plots each model's mean predicted prob (by bin) against the "
    "empirical positive rate; a perfectly calibrated model traces the dashed diagonal."
)

_calibration_rows = []
_curve_traces = []
_curve_palette = {
    "stacked_calibrated": "#0b5394",
    "xgboost":            "#cc4125",
    "lightgbm":           "#674ea7",
    "catboost":           "#e69138",
    "random_forest":      "#38761d",
    "logistic":           "#6c757d",
    "xgboost_dart":       "#a64d79",
}
for _name, _res in model_results.items():
    if "error" in _res or _name not in MODEL_DISPLAY:
        continue
    _display, _ = MODEL_DISPLAY[_name]
    _calres = calibration_metrics(y_true, _res["prob"], n_bins=10)
    _calibration_rows.append({
        "Model":  _display,
        "Brier":  round(_calres["brier"], 4),
        "ECE":    round(_calres["ece"], 4),
        "ROC-AUC": round(_res["metrics"].get("roc_auc", float("nan")), 3),
    })
    _curve = _calres["curve"].dropna(subset=["mean_pred", "empirical_pos_rate"])
    if not _curve.empty:
        _curve_traces.append(
            go.Scatter(
                x=_curve["mean_pred"],
                y=_curve["empirical_pos_rate"],
                mode="lines+markers",
                name=_display,
                line=dict(color=_curve_palette.get(_name, "#444"), width=2),
                marker=dict(size=7),
            )
        )

if _calibration_rows:
    cal_col1, cal_col2 = st.columns([1, 2])
    with cal_col1:
        st.markdown("**Calibration Metrics**")
        st.dataframe(
            pd.DataFrame(_calibration_rows).sort_values("ECE"),
            width='stretch', hide_index=True,
        )
        st.caption("Lower Brier and ECE are better. ROC-AUC is rank-based and unaffected by calibration.")
    with cal_col2:
        st.markdown("**Reliability Curve**")
        cal_fig = go.Figure()
        cal_fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="gray", dash="dash", width=1),
            name="Perfect calibration",
            hoverinfo="skip",
        ))
        for _trace in _curve_traces:
            cal_fig.add_trace(_trace)
        cal_fig.update_layout(
            xaxis_title="Mean predicted probability (per bin)",
            yaxis_title="Empirical positive rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="left", x=0),
            margin=dict(t=30, b=30, l=20, r=20),
        )
        st.plotly_chart(cal_fig, width='stretch')
else:
    st.info("Reliability metrics unavailable — no model produced valid probabilities.")

# ---------------------------------------------------------------------------
# Section 8 — Row-level table
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Row-Level Predictions")

if xgb_res and "error" not in xgb_res:
    prob_xgb = xgb_res["prob"]
    pred_xgb = xgb_res["pred"]

    display_cols = {}
    for col in ["hotel", "lead_time", "adr", "market_segment", "deposit_type",
                "adults", "total_nights"]:
        if col in df.columns:
            display_cols[col] = df[col].values

    table_df = pd.DataFrame(display_cols)
    table_df["Predicted Prob"] = (prob_xgb * 100).round(1)
    table_df["Predicted"] = np.where(pred_xgb == 1, "Canceled", "Not Canceled")
    table_df["Actual"] = np.where(y_true == 1, "Canceled", "Not Canceled")
    table_df["Result"] = np.where(pred_xgb == y_true, "✓ Correct", "✗ Wrong")
    table_df = table_df.sort_values("Predicted Prob", ascending=False).reset_index(drop=True)
    table_df.index += 1

    filter_choice = st.radio(
        "Show",
        ["All", "Correct only", "Wrong only", "Actual cancellations", "Actual non-cancellations"],
        horizontal=True,
    )

    filtered = table_df.copy()
    if filter_choice == "Correct only":
        filtered = filtered[filtered["Result"] == "✓ Correct"]
    elif filter_choice == "Wrong only":
        filtered = filtered[filtered["Result"] == "✗ Wrong"]
    elif filter_choice == "Actual cancellations":
        filtered = filtered[filtered["Actual"] == "Canceled"]
    elif filter_choice == "Actual non-cancellations":
        filtered = filtered[filtered["Actual"] == "Not Canceled"]

    st.caption(f"Showing {len(filtered):,} of {len(table_df):,} bookings — sorted by predicted probability (highest first)")

    st.dataframe(
        filtered.style.applymap(
            lambda v: "color: green; font-weight: bold" if v == "✓ Correct"
                      else ("color: red; font-weight: bold" if v == "✗ Wrong" else ""),
            subset=["Result"],
        ).applymap(
            lambda v: "color: tomato;" if v == "Canceled" else "",
            subset=["Predicted", "Actual"],
        ).format({"Predicted Prob": "{:.1f}%"}),
        width='stretch',
        height=500,
    )

    # Download button
    csv_bytes = filtered.to_csv(index=True).encode()
    st.download_button(
        "Download filtered results as CSV",
        data=csv_bytes,
        file_name="validation_results.csv",
        mime="text/csv",
    )
