"""Tests that services.py accepts dataset_context without importing Streamlit."""
import importlib
import inspect
import sys
import types


def test_services_importable_without_streamlit():
    """Importing src.services must not require streamlit."""
    stub = types.ModuleType("streamlit")
    sys.modules.setdefault("streamlit", stub)
    importlib.invalidate_caches()
    import src.services
    assert hasattr(src.services, "load_and_prepare_data")
    assert hasattr(src.services, "get_or_train_models")


def test_load_and_prepare_data_accepts_dataset_context():
    from src.services import load_and_prepare_data
    sig = inspect.signature(load_and_prepare_data)
    assert "dataset_context" in sig.parameters


def test_get_or_train_models_accepts_dataset_context():
    from src.services import get_or_train_models
    sig = inspect.signature(get_or_train_models)
    assert "dataset_context" in sig.parameters
