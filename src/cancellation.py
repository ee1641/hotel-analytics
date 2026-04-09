"""
Binary classification for booking cancellation and no-show prediction.
Models: Logistic Regression, Random Forest, XGBoost.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def train_cancellation_models(X_train, y_train) -> dict:
    """
    Fit three classifiers on training data.

    Returns:
        dict mapping model name -> fitted model/pipeline:
          'logistic'      : StandardScaler + LogisticRegression pipeline
          'random_forest' : RandomForestClassifier
          'xgboost'       : XGBClassifier
    """
    models = {
        "logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, n_jobs=-1, random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", verbosity=0, random_state=42,
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


def train_cancellation_models_balanced(
    X_train,
    y_train,
    scale_pos_weight: float = None,
) -> dict:
    """
    Like train_cancellation_models but with class-imbalance corrections:
      - logistic      : class_weight='balanced'
      - random_forest : class_weight='balanced'
      - xgboost       : scale_pos_weight (pass compute_scale_pos_weight(y_train))
    """
    spw = scale_pos_weight if scale_pos_weight is not None else compute_scale_pos_weight(y_train)
    models = {
        "logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, n_jobs=-1, random_state=42,
            class_weight="balanced",
        ),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", verbosity=0, random_state=42,
            scale_pos_weight=spw,
        ),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def get_shap_feature_importance(model, X_test: np.ndarray, feature_names: list) -> pd.DataFrame:
    """
    Compute mean absolute SHAP values per feature using TreeExplainer.
    Only compatible with tree-based models (XGBoost, RandomForest).

    Returns a DataFrame sorted by mean_abs_shap descending with columns:
        feature, mean_abs_shap
    """
    import shap
    # XGBClassifier wrapped in Pipeline needs to be unwrapped
    actual_model = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    explainer = shap.TreeExplainer(actual_model)
    shap_values = explainer.shap_values(X_test)
    # shap_values may be a list (binary classification) — use class-1 values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def select_top_features(shap_df: pd.DataFrame, threshold: float = 0.01) -> list:
    """Return feature names with mean |SHAP| above threshold."""
    return shap_df.loc[shap_df["mean_abs_shap"] > threshold, "feature"].tolist()
