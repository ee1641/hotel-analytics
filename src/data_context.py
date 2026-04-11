from __future__ import annotations

import hashlib
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.hotel_manager import HotelManager

DEMO_DATA_PATH = Path("hotel_bookings.csv")
DEMO_MODELS_DIR = Path("models")
STREAMLIT_UPLOADS_ROOT = Path("data/streamlit_uploads")

STREAMLIT_CONTEXT_KEYS = (
    "active_dataset_id",
    "active_data_path",
    "active_models_dir",
    "active_source_label",
)


@dataclass(frozen=True)
class DatasetContext:
    dataset_id: str
    data_path: str
    models_dir: str
    source_label: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def build_demo_dataset_context() -> DatasetContext:
    return DatasetContext(
        dataset_id="demo",
        data_path=str(DEMO_DATA_PATH),
        models_dir=str(DEMO_MODELS_DIR),
        source_label="Built-in Kaggle Demo Dataset",
    )


def _hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]


def fingerprint_file(path: str | Path) -> str:
    file_path = Path(path)
    return _hash_bytes(file_path.read_bytes())


def persist_streamlit_upload(mapped_df: pd.DataFrame) -> DatasetContext:
    csv_bytes = mapped_df.to_csv(index=False).encode("utf-8")
    fingerprint = _hash_bytes(csv_bytes)
    dataset_id = f"streamlit_{fingerprint}"
    dataset_root = STREAMLIT_UPLOADS_ROOT / dataset_id
    dataset_root.mkdir(parents=True, exist_ok=True)

    data_path = dataset_root / "bookings.csv"
    data_path.write_bytes(csv_bytes)

    models_dir = dataset_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    return DatasetContext(
        dataset_id=dataset_id,
        data_path=str(data_path),
        models_dir=str(models_dir),
        source_label=f"Custom Upload ({dataset_id})",
    )


def activate_streamlit_context(session_state, context: DatasetContext) -> None:
    session_state["active_dataset_id"] = context.dataset_id
    session_state["active_data_path"] = context.data_path
    session_state["active_models_dir"] = context.models_dir
    session_state["active_source_label"] = context.source_label


def activate_streamlit_demo(session_state) -> DatasetContext:
    for key in STREAMLIT_CONTEXT_KEYS:
        session_state.pop(key, None)
    return build_demo_dataset_context()


def resolve_streamlit_dataset_context(session_state) -> DatasetContext:
    dataset_id = session_state.get("active_dataset_id")
    data_path = session_state.get("active_data_path")
    models_dir = session_state.get("active_models_dir")
    source_label = session_state.get("active_source_label")

    if dataset_id and data_path and models_dir and source_label and Path(data_path).exists():
        return DatasetContext(
            dataset_id=dataset_id,
            data_path=data_path,
            models_dir=models_dir,
            source_label=source_label,
        )

    return build_demo_dataset_context()


def resolve_hotel_dataset_context(hotel_id: str, manager: HotelManager | None = None) -> DatasetContext:
    if hotel_id == "demo":
        return build_demo_dataset_context()

    manager = manager or HotelManager()
    data_path = manager.get_hotel_data_path(hotel_id)
    if not data_path.exists():
        raise FileNotFoundError(
            f"No uploaded dataset found for hotel '{hotel_id}'. "
            "Upload bookings CSV to /hotels/{hotel_id}/upload-data first."
        )

    return DatasetContext(
        dataset_id=f"{hotel_id}_{fingerprint_file(data_path)}",
        data_path=str(data_path),
        models_dir=str(manager.get_hotel_models_path(hotel_id)),
        source_label=f"Hotel Upload ({hotel_id})",
    )


def clear_models_dir(models_dir: str | Path) -> None:
    models_path = Path(models_dir)
    if not models_path.exists():
        return

    for item in models_path.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
