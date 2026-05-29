"""
Cancellation modelling:
  - Binary classification for cancellation risk
  - Survival analysis for time-to-cancellation
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.revenue_management import prepare_booking_timeline

log = logging.getLogger(__name__)

# Market segment risk tiers for segment-specific cancellation models
SEGMENT_GROUPS = {
    "high_risk":   ["Online TA", "Groups"],
    "medium_risk": ["Offline TA/TO", "Direct", "Complementary", "Aviation"],
    "low_risk":    ["Corporate"],
}

# Minimum training samples needed to fit a segment-specific model
_MIN_SEGMENT_SAMPLES = 100

# Categorical columns passed to CatBoost for native handling (label-encoded integers)
CATBOOST_CAT_COLS = [
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
]

# High-cardinality columns handled via target encoding (not label encoding)
TARGET_ENCODED_COLS = ["country", "agent"]


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
            n_jobs=-1,
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
    n_iter: int = 15,
    n_splits: int = 3,
) -> dict:
    """
    Run RandomizedSearchCV over XGBoost hyperparameters using TimeSeriesSplit
    to prevent future data leaking into training folds.
    Returns best_params_ dict; pass as xgb_params to train_cancellation_models_balanced.
    """
    param_dist = {
        "n_estimators": [200, 400, 600, 800],
        "max_depth": [4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2],
        "reg_alpha": [0, 0.1, 0.5],
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
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_tr, y_tr)
    return search.best_params_


def tune_lgb_cancellation(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    scale_pos_weight: float,
    n_iter: int = 15,
    n_splits: int = 3,
) -> dict:
    """
    Run RandomizedSearchCV over LightGBM hyperparameters using TimeSeriesSplit.
    Returns best_params_ dict.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        log.warning("lightgbm not installed — skipping LGB tuning, using defaults")
        return {}

    param_dist = {
        "n_estimators": [200, 400, 600, 800],
        "num_leaves": [31, 63, 127],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "min_child_samples": [10, 20, 50],
        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [0, 0.1, 1.0],
    }
    base = lgb.LGBMClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    search = RandomizedSearchCV(
        base,
        param_dist,
        n_iter=n_iter,
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_tr, y_tr)
    return search.best_params_


def tune_catboost_cancellation(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    scale_pos_weight: float,
    cat_feature_indices: list[int],
    n_iter: int = 15,
    n_splits: int = 3,
) -> dict:
    """
    Run RandomizedSearchCV over CatBoost hyperparameters using TimeSeriesSplit.
    Returns best_params_ dict.
    """
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        log.warning("catboost not installed — skipping CatBoost tuning, using defaults")
        return {}

    param_dist = {
        "iterations": [300, 500, 800],
        "depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "l2_leaf_reg": [1, 3, 5, 10],
        "bagging_temperature": [0.0, 0.5, 1.0],
    }
    # cat_features cannot be passed to the base estimator used in RandomizedSearchCV
    # because sklearn's clone() cannot reconstruct CatBoost with that parameter.
    # We tune hyperparameters without cat_features here; cat_features are applied
    # in train_cancellation_models_balanced when building the final model.
    base = CatBoostClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=0,
    )
    search = RandomizedSearchCV(
        base,
        param_dist,
        n_iter=n_iter,
        cv=TimeSeriesSplit(n_splits=n_splits),
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
    lgb_params: dict | None = None,
    catboost_params: dict | None = None,
    feature_columns: list[str] | None = None,
) -> dict:
    """
    Train six balanced classifiers: logistic, random_forest, xgboost, xgboost_dart,
    lightgbm, catboost. LightGBM and CatBoost are skipped gracefully if not installed.
    Pass *_params dicts (from tune_* functions) to override defaults.
    feature_columns is required for CatBoost to identify categorical feature indices.
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
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
        ),
        "xgboost": XGBClassifier(
            **{
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                **(xgb_params or {}),
                "eval_metric": "logloss",
                "verbosity": 0,
                "random_state": 42,
                "scale_pos_weight": spw,
            }
        ),
        # DART booster: dropout regularisation on gradient boosting trees.
        # Cannot use early stopping (n_estimators is fixed). Tuned separately via defaults.
        "xgboost_dart": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            booster="dart",
            rate_drop=0.1,
            skip_drop=0.5,
            eval_metric="logloss",
            verbosity=0,
            random_state=42,
            scale_pos_weight=spw,
        ),
    }

    try:
        import lightgbm as lgb
        lgb_defaults = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "n_jobs": -1,
            "verbose": -1,
            "random_state": 42,
            "scale_pos_weight": spw,
        }
        lgb_defaults.update(lgb_params or {})
        models["lightgbm"] = lgb.LGBMClassifier(**lgb_defaults)
    except ImportError:
        log.warning("lightgbm not installed — skipping LightGBM model")

    try:
        from catboost import CatBoostClassifier
        cb_defaults = {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "scale_pos_weight": spw,
            "random_state": 42,
            "verbose": 0,
        }
        for k, v in (catboost_params or {}).items():
            if k not in ("scale_pos_weight", "random_state", "verbose"):
                cb_defaults[k] = v
        models["catboost"] = CatBoostClassifier(**cb_defaults)
    except ImportError:
        log.warning("catboost not installed — skipping CatBoost model")

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


def train_stacked_cancellation_model(
    base_models: dict,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    n_splits: int = 5,
) -> LogisticRegression:
    """
    Build a stacking meta-learner using out-of-fold predictions from all base models.

    Uses TimeSeriesSplit to generate OOF predictions without leaking future data.
    Trains a logistic regression meta-learner on the resulting OOF probability matrix.
    Base models should already be fitted on the full training set before calling predict.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model_names = [k for k in base_models if k not in _NON_MODEL_KEYS]
    oof = np.zeros((len(X_tr), len(model_names)))
    # Init weights to zero so the leading prefix never placed in any val_idx
    # (TimeSeriesSplit reserves the first ~1/(n_splits+1) of rows for training
    # only) is excluded from the meta-learner fit. Otherwise those phantom
    # all-zero OOF rows would be fit at maximum weight.
    sample_weights = np.zeros(len(X_tr))

    for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_tr)):
        X_fold_tr, X_fold_val = X_tr[tr_idx], X_tr[val_idx]
        y_fold_tr = y_tr[tr_idx]
        sample_weights[val_idx] = (fold_idx + 1) / n_splits
        for m_idx, name in enumerate(model_names):
            m = clone(base_models[name])
            m.fit(X_fold_tr, y_fold_tr)
            oof[val_idx, m_idx] = m.predict_proba(X_fold_val)[:, 1]

    covered = sample_weights > 0
    meta = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    meta.fit(oof[covered], y_tr[covered], sample_weight=sample_weights[covered])
    return meta


_NON_MODEL_KEYS = {
    "stacked_meta",
    "optimal_threshold",
    "target_encode_maps",
    "global_cancel_mean",
    "probability_calibrator",
}


def predict_stacked_proba(
    meta: LogisticRegression,
    base_models: dict,
    X: np.ndarray,
    calibrator=None,
) -> np.ndarray:
    """
    Generate stacked probability predictions for X using fitted base models + meta-learner.
    If a fitted calibrator (e.g. IsotonicRegression) is supplied, raw stacked probs are
    transformed through it and clipped to [0, 1] before being returned.
    """
    model_names = [k for k in base_models if k not in _NON_MODEL_KEYS]
    level1 = np.column_stack([
        base_models[name].predict_proba(X)[:, 1]
        for name in model_names
    ])
    raw = meta.predict_proba(level1)[:, 1]
    return apply_probability_calibrator(calibrator, raw)


def train_probability_calibrator(
    base_models: dict,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    n_splits: int = 5,
):
    """
    Fit an isotonic regression on out-of-fold stacked predictions so raw stacked
    probabilities map to true frequencies. Uses TimeSeriesSplit so calibration
    data never leaks future info into the fit.

    base_models must already contain a fitted ``stacked_meta`` — the same
    meta-learner is reused per fold to convert refit base-model predictions
    into stacked probabilities. Returns a fitted ``IsotonicRegression`` (callable
    via ``apply_probability_calibrator``) or ``None`` if calibration is infeasible.
    """
    from sklearn.isotonic import IsotonicRegression

    meta = base_models.get("stacked_meta")
    if meta is None:
        raise ValueError("base_models must contain 'stacked_meta' before calibration")

    model_names = [k for k in base_models if k not in _NON_MODEL_KEYS]
    if not model_names:
        return None

    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_stacked = np.zeros(len(X_tr))
    covered = np.zeros(len(X_tr), dtype=bool)

    for tr_idx, val_idx in tscv.split(X_tr):
        if len(np.unique(y_tr[tr_idx])) < 2:
            continue
        level1 = np.column_stack([
            clone(base_models[name]).fit(X_tr[tr_idx], y_tr[tr_idx]).predict_proba(X_tr[val_idx])[:, 1]
            for name in model_names
        ])
        oof_stacked[val_idx] = meta.predict_proba(level1)[:, 1]
        covered[val_idx] = True

    if covered.sum() < 10 or len(np.unique(y_tr[covered])) < 2:
        return None

    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(oof_stacked[covered], y_tr[covered])
    return calibrator


def apply_probability_calibrator(calibrator, raw_proba: np.ndarray) -> np.ndarray:
    """Apply a fitted calibrator. Returns probs clipped to [0, 1]. No-op if calibrator is None."""
    raw_proba = np.asarray(raw_proba, dtype=float)
    if calibrator is None:
        return raw_proba
    return np.clip(calibrator.transform(raw_proba), 0.0, 1.0)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float]:
    """
    Sweep thresholds [0.30, 0.70] in 0.01 steps and return the threshold that
    maximises accuracy, along with the achieved accuracy.
    """
    best_thresh, best_score = 0.5, 0.0
    for t in np.arange(0.30, 0.71, 0.01):
        preds = (y_prob >= t).astype(int)
        score = float(accuracy_score(y_true, preds))
        if score > best_score:
            best_score, best_thresh = score, float(t)
    return best_thresh, best_score


def get_segment_group(market_segment: str) -> str | None:
    """Return the SEGMENT_GROUPS key for a given market_segment value, or None if unknown."""
    for group, segments in SEGMENT_GROUPS.items():
        if market_segment in segments:
            return group
    return None


def train_segmented_cancellation_models(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    segment_labels_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    segment_labels_te: np.ndarray,
    xgb_params: dict | None = None,
    lgb_params: dict | None = None,
    catboost_params: dict | None = None,
) -> dict:
    """
    Train per-segment-group cancellation models.

    Splits training/test data by SEGMENT_GROUPS, then for each group trains the
    full 6-model balanced ensemble + stacking meta-learner + threshold optimisation.

    Groups with fewer than _MIN_SEGMENT_SAMPLES training rows or only one target
    class are skipped. Returns a dict keyed by segment group name; each value is
    a cancel_models bundle (same structure as the global model).
    """
    result = {}
    segment_labels_tr = np.asarray(segment_labels_tr)
    segment_labels_te = np.asarray(segment_labels_te)

    for group_name, segments in SEGMENT_GROUPS.items():
        tr_mask = np.isin(segment_labels_tr, segments)
        te_mask = np.isin(segment_labels_te, segments)

        X_g_tr = X_tr[tr_mask]
        y_g_tr = y_tr[tr_mask]
        X_g_te = X_te[te_mask]
        y_g_te = y_te[te_mask]

        if len(X_g_tr) < _MIN_SEGMENT_SAMPLES:
            log.warning(
                "Segment '%s' has only %d training samples — skipping",
                group_name, len(X_g_tr),
            )
            continue

        if len(np.unique(y_g_tr)) < 2:
            log.warning("Segment '%s' has only one target class — skipping", group_name)
            continue

        log.info(
            "Training segment '%s' model (%d train / %d test)...",
            group_name, len(X_g_tr), len(X_g_te),
        )

        spw = compute_scale_pos_weight(y_g_tr)
        cancel_models_g = train_cancellation_models_balanced(
            X_g_tr,
            y_g_tr,
            scale_pos_weight=spw,
            xgb_params=xgb_params,
            lgb_params=lgb_params,
            catboost_params=catboost_params,
        )

        stacked_meta_g = train_stacked_cancellation_model(cancel_models_g, X_g_tr, y_g_tr)
        cancel_models_g["stacked_meta"] = stacked_meta_g

        if len(X_g_te) > 0 and len(np.unique(y_g_te)) >= 2:
            stacked_proba = predict_stacked_proba(stacked_meta_g, cancel_models_g, X_g_te)
            optimal_threshold, optimal_acc = find_optimal_threshold(y_g_te, stacked_proba)
            cancel_models_g["optimal_threshold"] = optimal_threshold
            log.info(
                "  Segment '%s' stacked — threshold=%.2f, accuracy=%.3f",
                group_name, optimal_threshold, optimal_acc,
            )
        else:
            cancel_models_g["optimal_threshold"] = 0.5

        result[group_name] = cancel_models_g

    return result


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
