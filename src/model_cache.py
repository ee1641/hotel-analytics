"""
Persistence helpers for trained models and fitted artefacts.
All models are stored under models_dir (default: 'models/').

Versioned format: artifacts are wrapped in a dict:
    {"artifact": obj, "metadata": {...}, "saved_at": "ISO timestamp"}
Legacy artifacts (raw object) are still loadable transparently.
"""
from datetime import datetime
from pathlib import Path
import joblib

MODELS_DIR = Path("models")

_PATHS: dict[str, str] = {
    "xgb_revenue":      "xgb_revenue_forecaster.pkl",
    "sarima_revenue":   "sarima_revenue_result.pkl",
    "xgb_demand":       "xgb_demand_forecaster.pkl",
    "sarima_demand":    "sarima_demand_result.pkl",
    "cancel_models":    "cancellation_models.pkl",
    "feature_columns":  "feature_columns.pkl",
    "label_encoders":   "label_encoders.pkl",
    "feature_defaults": "feature_defaults.pkl",
    "eval_metrics":     "eval_metrics.pkl",
    "selected_features":  "selected_features.pkl",
    "prophet_revenue":    "prophet_revenue_model.pkl",
    "prophet_demand":     "prophet_demand_model.pkl",
}


def ensure_models_dir(models_dir: Path = MODELS_DIR) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)


def _path_for(name: str, models_dir: Path) -> Path:
    filename = _PATHS.get(name, f"{name}.pkl")
    return models_dir / filename


def save_artifact(name: str, obj, metadata: dict = None, models_dir: Path = MODELS_DIR) -> Path:
    """
    Serialize obj to models_dir/<canonical_filename> using joblib.
    Wraps obj in a versioned envelope with metadata and timestamp.
    """
    ensure_models_dir(models_dir)
    dest = _path_for(name, models_dir)
    wrapper = {
        "artifact": obj,
        "metadata": metadata or {},
        "saved_at": datetime.now().isoformat(),
    }
    joblib.dump(wrapper, dest)
    return dest


def load_artifact(name: str, models_dir: Path = MODELS_DIR):
    """
    Deserialize and return the artifact, or None if the file does not exist.
    Transparently unwraps both new versioned format and legacy raw objects.
    """
    path = _path_for(name, models_dir)
    if not path.exists():
        return None
    raw = joblib.load(path)
    if isinstance(raw, dict) and "artifact" in raw and "saved_at" in raw:
        return raw["artifact"]
    return raw  # legacy format


def load_artifact_with_metadata(name: str, models_dir: Path = MODELS_DIR) -> tuple:
    """
    Returns (artifact, metadata_dict).
    metadata_dict is {} for legacy files or when artifact is missing.
    """
    path = _path_for(name, models_dir)
    if not path.exists():
        return None, {}
    raw = joblib.load(path)
    if isinstance(raw, dict) and "artifact" in raw and "saved_at" in raw:
        return raw["artifact"], raw.get("metadata", {})
    return raw, {}  # legacy format


def artifact_exists(name: str, models_dir: Path = MODELS_DIR) -> bool:
    """Return True if the persisted file exists on disk."""
    return _path_for(name, models_dir).exists()


def all_artifacts_exist(models_dir: Path = MODELS_DIR) -> bool:
    """Return True if every canonical artifact is present."""
    return all(_path_for(name, models_dir).exists() for name in _PATHS)
