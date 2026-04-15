"""
Cancellation modelling:
  - Binary classification for cancellation risk
  - Survival analysis for time-to-cancellation
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.revenue_management import prepare_booking_timeline


def train_cancellation_models(X_train, y_train) -> dict:
    """
    Fit three classifiers on training data.
    """
    models = {
        "logistic": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            n_jobs=1,
            random_state=42,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss",
            verbosity=0,
            random_state=42,
        ),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def predict_cancellation_proba(model, X) -> np.ndarray:
    """Return probability of cancellation (class 1) for each row."""
    return model.predict_proba(X)[:, 1]


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """Return (count_0 / count_1) for use as XGBoost scale_pos_weight."""
    counts = np.bincount(y.astype(int))
    return float(counts[0] / counts[1])


def tune_xgb_cancellation(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    scale_pos_weight: float,
    n_iter: int = 10,
    cv: int = 3,
) -> dict:
    """
    Run RandomizedSearchCV over XGBoost hyperparameters.
    Returns best_params_ dict; pass as xgb_params to train_cancellation_models_balanced.
    Uses n_jobs=-1 for parallel fold evaluation.
    """
    param_dist = {
        "n_estimators": [100, 200, 400, 600],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
    }
    base = XGBClassifier(
        eval_metric="logloss",
        verbosity=0,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )
    search = RandomizedSearchCV(
        base,
        param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_tr, y_tr)
    return search.best_params_


def train_cancellation_models_balanced(
    X_train,
    y_train,
    scale_pos_weight: float = None,
    xgb_params: dict | None = None,
) -> dict:
    """
    Like train_cancellation_models but with class-imbalance corrections.
    Pass xgb_params (from tune_xgb_cancellation) to override default XGB hyperparameters.
    """
    spw = scale_pos_weight if scale_pos_weight is not None else compute_scale_pos_weight(y_train)
    models = {
        "logistic": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            n_jobs=1,
            random_state=42,
            class_weight="balanced",
        ),
        "xgboost": XGBClassifier(
            **{
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                **(xgb_params or {}),
                # Always enforced regardless of xgb_params:
                "eval_metric": "logloss",
                "verbosity": 0,
                "random_state": 42,
                "scale_pos_weight": spw,
            }
        ),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def get_shap_feature_importance(model, X_test: np.ndarray, feature_names: list) -> pd.DataFrame:
    """
    Compute mean absolute SHAP values per feature using TreeExplainer.
    """
    import shap

    actual_model = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    explainer = shap.TreeExplainer(actual_model)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def select_top_features(shap_df: pd.DataFrame, threshold: float = 0.01) -> list:
    """Return feature names with mean |SHAP| above threshold."""
    return shap_df.loc[shap_df["mean_abs_shap"] > threshold, "feature"].tolist()


def _km_curve(durations: np.ndarray, events: np.ndarray) -> pd.DataFrame:
    """
    Kaplan-Meier survival curve implemented without external dependencies.
    """
    durations = np.asarray(durations, dtype=int)
    events = np.asarray(events, dtype=int)
    if len(durations) == 0:
        return pd.DataFrame({"timeline": [0], "survival_probability": [1.0]})

    max_duration = int(durations.max())
    at_risk = float(len(durations))
    survival = 1.0
    rows = [{"timeline": 0, "survival_probability": 1.0}]

    for t in range(1, max_duration + 1):
        observed = float(((durations == t) & (events == 1)).sum())
        censored = float(((durations == t) & (events == 0)).sum())
        if at_risk > 0 and observed > 0:
            survival *= 1.0 - (observed / at_risk)
        rows.append({"timeline": t, "survival_probability": survival})
        at_risk -= observed + censored
        if at_risk <= 0:
            break

    return pd.DataFrame(rows)


def prepare_survival_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a booking-level survival dataset using booking and cancellation timing.
    """
    timeline = prepare_booking_timeline(df)
    timeline["event_observed"] = timeline.get("is_canceled", 0).fillna(0).astype(int)

    event_end = np.where(
        timeline["event_observed"].eq(1),
        timeline["cancel_date"],
        timeline["arrival_date"],
    )
    event_end = pd.to_datetime(event_end, errors="coerce")
    durations = (event_end - timeline["booking_date"]).dt.days
    timeline["duration"] = durations.fillna(timeline["lead_time"]).clip(lower=0).astype(int)
    timeline["total_nights"] = pd.to_numeric(
        timeline.get("total_nights", 1),
        errors="coerce",
    ).fillna(1).clip(lower=1)
    timeline["adr"] = pd.to_numeric(timeline.get("adr", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    timeline["deposit_type"] = timeline.get("deposit_type", "Unknown").fillna("Unknown").astype(str)
    timeline["market_segment"] = timeline.get("market_segment", "Unknown").fillna("Unknown").astype(str)
    return timeline[
        ["duration", "event_observed", "lead_time", "adr", "total_nights", "deposit_type", "market_segment"]
    ].copy()


def _build_survival_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    features = df[["lead_time", "adr", "total_nights"]].copy()
    features["adr"] = np.log1p(features["adr"])
    categoricals = pd.get_dummies(
        df[["deposit_type", "market_segment"]],
        prefix=["deposit_type", "market_segment"],
        dtype=float,
    )
    return pd.concat([features, categoricals], axis=1)


def train_cancellation_survival_model(df: pd.DataFrame) -> dict:
    """
    Train a personalized cancellation survival model.

    Uses lifelines when available. Falls back to a calibrated Kaplan-Meier + logistic
    risk multiplier when lifelines is not installed.
    """
    surv_df = prepare_survival_frame(df)
    surv_df = surv_df[surv_df["duration"] >= 0].reset_index(drop=True)
    km_curve = _km_curve(surv_df["duration"].values, surv_df["event_observed"].values)
    overall_cancel_rate = float(surv_df["event_observed"].mean()) if len(surv_df) else 0.0

    bundle = {
        "km_curve": km_curve,
        "overall_cancel_rate": overall_cancel_rate,
        "feature_columns": [],
        "kind": "fallback",
    }

    X = _build_survival_design_matrix(surv_df)
    bundle["feature_columns"] = X.columns.tolist()

    try:
        from lifelines import CoxPHFitter

        cox_df = pd.concat(
            [surv_df[["duration", "event_observed"]], X],
            axis=1,
        )
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(cox_df, duration_col="duration", event_col="event_observed")
        bundle["kind"] = "cox"
        bundle["cox_model"] = cph
        bundle["feature_means"] = X.mean().to_dict()
        return bundle
    except Exception:
        fallback = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
            ]
        )
        fallback.fit(X.values, surv_df["event_observed"].values)
        bundle["fallback_model"] = fallback
        bundle["feature_means"] = X.mean().to_dict()
        return bundle


def _survival_lookup(km_curve: pd.DataFrame, duration: int) -> float:
    eligible = km_curve[km_curve["timeline"] <= int(duration)]
    if eligible.empty:
        return 1.0
    return float(eligible.iloc[-1]["survival_probability"])


def _align_survival_row(model_bundle: dict, booking: pd.DataFrame | dict) -> pd.DataFrame:
    row_df = pd.DataFrame([booking] if isinstance(booking, dict) else booking).copy()
    if "total_nights" not in row_df.columns:
        weekend = pd.to_numeric(row_df.get("stays_in_weekend_nights", 0), errors="coerce").fillna(0)
        week = pd.to_numeric(row_df.get("stays_in_week_nights", 0), errors="coerce").fillna(0)
        row_df["total_nights"] = weekend + week
    row_df["deposit_type"] = row_df.get("deposit_type", "Unknown").fillna("Unknown").astype(str)
    row_df["market_segment"] = row_df.get("market_segment", "Unknown").fillna("Unknown").astype(str)
    row_df["lead_time"] = pd.to_numeric(row_df.get("lead_time", 0), errors="coerce").fillna(0).clip(lower=0)
    row_df["adr"] = pd.to_numeric(row_df.get("adr", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    row_df["total_nights"] = pd.to_numeric(row_df.get("total_nights", 1), errors="coerce").fillna(1).clip(lower=1)

    design = _build_survival_design_matrix(row_df)
    for col in model_bundle.get("feature_columns", []):
        if col not in design.columns:
            design[col] = model_bundle.get("feature_means", {}).get(col, 0.0)
    design = design[model_bundle.get("feature_columns", design.columns.tolist())]
    return design


def predict_booking_survival_profile(
    model_bundle: dict,
    booking: pd.DataFrame | dict,
    checkpoints: tuple[int, ...] = (30, 14, 7, 0),
) -> pd.DataFrame:
    """
    Predict the probability that a booking is still alive at selected days-before-arrival checkpoints.
    """
    design = _align_survival_row(model_bundle, booking)
    booking_df = pd.DataFrame([booking] if isinstance(booking, dict) else booking).copy()
    lead_time = float(pd.to_numeric(booking_df.get("lead_time", 0), errors="coerce").fillna(0).iloc[0])
    durations = [max(int(round(lead_time - checkpoint)), 0) for checkpoint in checkpoints]

    if model_bundle.get("kind") == "cox" and "cox_model" in model_bundle:
        survival_fn = model_bundle["cox_model"].predict_survival_function(design, times=durations)
        probs = [float(survival_fn.iloc[i, 0]) for i in range(len(durations))]
    else:
        km_curve = model_bundle["km_curve"]
        base_probs = [_survival_lookup(km_curve, duration) for duration in durations]
        raw_prob = float(model_bundle["fallback_model"].predict_proba(design.values)[0][1])
        overall = max(float(model_bundle.get("overall_cancel_rate", 0.0)), 1e-6)
        risk_multiplier = float(np.clip(raw_prob / overall, 0.35, 3.0))
        probs = [float(np.clip(base ** risk_multiplier, 0.0, 1.0)) for base in base_probs]

    return pd.DataFrame(
        {
            "days_before_arrival": list(checkpoints),
            "survival_probability": probs,
        }
    )


def get_survival_curve(model_bundle: dict) -> pd.DataFrame:
    """Return the baseline Kaplan-Meier survival curve."""
    return model_bundle.get("km_curve", pd.DataFrame({"timeline": [0], "survival_probability": [1.0]})).copy()
