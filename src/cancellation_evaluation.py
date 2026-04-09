"""
Cross-dataset evaluation of cancellation models.
Trains on hotel_bookings.csv (dataset 1), evaluates on cancellation_Prediction.csv (dataset 2).
Demonstrates model generalization across different hotel datasets.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.evaluation import classification_report_dict
from src.cancellation import predict_cancellation_proba


def _align_to_feature_columns(
    df: pd.DataFrame,
    feature_columns: list,
    label_encoders: dict,
    feature_defaults: dict,
) -> np.ndarray:
    """
    Align a DataFrame to the expected feature matrix shape.
    - Missing columns are filled with feature_defaults values (or 0.0)
    - Categorical columns are encoded using saved label_encoders (unknown → 0)
    - Returns a numpy array of shape (n_rows, len(feature_columns))
    """
    result = {}
    for col in feature_columns:
        if col in df.columns:
            series = df[col].copy()
            if col in label_encoders:
                le = label_encoders[col]
                def _safe_encode(v):
                    try:
                        return float(le.transform([str(v)])[0])
                    except (ValueError, TypeError):
                        return 0.0
                result[col] = series.apply(_safe_encode).values.astype(float)
            else:
                result[col] = pd.to_numeric(series, errors="coerce").fillna(
                    feature_defaults.get(col, 0.0)
                ).values.astype(float)
        else:
            # Column not present in second dataset — use training-set default
            result[col] = np.full(len(df), feature_defaults.get(col, 0.0), dtype=float)

    X = np.column_stack([result[col] for col in feature_columns])
    return X


def evaluate_cross_dataset(
    models: dict,
    feature_columns: list,
    label_encoders: dict,
    feature_defaults: dict,
    second_df: pd.DataFrame,
) -> dict:
    """
    Evaluate trained cancellation models on a second hotel dataset.
    Handles schema differences by filling missing features with training-set defaults.

    Args:
        models          : dict of fitted classifiers {"logistic": ..., "random_forest": ..., "xgboost": ...}
        feature_columns : list of feature names in training order
        label_encoders  : dict of col → LabelEncoder (from training on dataset 1)
        feature_defaults: dict of col → median/mode (from training on dataset 1)
        second_df       : mapped DataFrame from the second dataset

    Returns:
        dict {model_name: {"accuracy": float, "roc_auc": float, "n_samples": int}}
    """
    if "is_canceled" not in second_df.columns:
        raise ValueError("second_df must contain 'is_canceled' column")

    # Drop rows where target is NaN
    df_eval = second_df.dropna(subset=["is_canceled"]).copy()
    y_true = df_eval["is_canceled"].astype(int).values

    X = _align_to_feature_columns(df_eval, feature_columns, label_encoders, feature_defaults)

    results = {}
    for name, model in models.items():
        try:
            y_pred = model.predict(X)
            y_prob = predict_cancellation_proba(model, X)
            m = classification_report_dict(y_true, y_pred, y_prob)
            results[name] = {
                "accuracy": round(m["accuracy"], 4),
                "roc_auc":  round(m["roc_auc"], 4),
                "n_samples": len(y_true),
            }
        except Exception as exc:
            results[name] = {"accuracy": None, "roc_auc": None, "error": str(exc)}

    return results
