"""
Booking risk scorer.

Enriches a bookings DataFrame with cancellation probability,
risk tier, and optional SHAP-based top risk factors per booking.
"""
import numpy as np
import pandas as pd
from src.preprocessing import drop_leaky_features, encode_categoricals


def score_bookings(
    bookings_df: pd.DataFrame,
    cancellation_model,
    shap_explainer=None,
) -> pd.DataFrame:
    """
    Add cancel_probability, risk_tier, and optionally top_risk_factors to each booking.

    Parameters
    ----------
    bookings_df : raw bookings DataFrame (same schema as hotel_bookings.csv)
    cancellation_model : fitted classifier with predict_proba
    shap_explainer : optional shap.TreeExplainer for per-prediction risk factors

    Returns
    -------
    DataFrame sorted by cancel_probability descending, with added columns:
        cancel_probability (float 0–1)
        risk_tier ("Low" | "Medium" | "High")
        top_risk_factors (list of dicts, only if shap_explainer provided)
    """
    df = bookings_df.copy()
    X, _, _ = encode_categoricals(drop_leaky_features(df))

    if hasattr(cancellation_model, "predict_proba"):
        proba = cancellation_model.predict_proba(X.values)[:, 1]
    else:
        proba = cancellation_model.predict(X.values).astype(float)

    df = df.copy()
    df["cancel_probability"] = proba
    df["risk_tier"] = pd.cut(
        df["cancel_probability"],
        bins=[-0.001, 0.30, 0.60, 1.001],
        labels=["Low", "Medium", "High"],
    )

    if shap_explainer is not None:
        shap_values = shap_explainer.shap_values(X.values)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        feature_names = list(X.columns)
        risk_factors = []
        for i, row_shap in enumerate(shap_values):
            top_idx = np.argsort(np.abs(row_shap))[-3:][::-1]
            factors = [
                {
                    "feature": feature_names[j],
                    "value": float(X.iloc[i, j]),
                    "impact": f"{row_shap[j]:+.2f}",
                }
                for j in top_idx
            ]
            risk_factors.append(factors)
        df["top_risk_factors"] = risk_factors

    return df.sort_values("cancel_probability", ascending=False).reset_index(drop=True)
