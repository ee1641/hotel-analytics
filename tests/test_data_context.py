from pathlib import Path

import pandas as pd

import app as app_mod
from src.data_context import build_demo_dataset_context, persist_streamlit_upload, resolve_hotel_dataset_context
from src.hotel_manager import HotelConfig, HotelManager
from src.preprocessing import load_data
from src.time_series import aggregate_daily, build_arrival_date


def test_streamlit_upload_creates_isolated_dataset_storage(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data_context.STREAMLIT_UPLOADS_ROOT", tmp_path)

    df_one = pd.DataFrame(
        [{"arrival_date": "2024-01-01", "is_canceled": 0, "adr": 100.0}]
    )
    df_two = pd.DataFrame(
        [{"arrival_date": "2024-01-01", "is_canceled": 0, "adr": 250.0}]
    )

    ctx_one = persist_streamlit_upload(df_one)
    ctx_two = persist_streamlit_upload(df_two)

    assert ctx_one.dataset_id != ctx_two.dataset_id
    assert Path(ctx_one.data_path).exists()
    assert Path(ctx_two.data_path).exists()
    assert Path(ctx_one.models_dir) != Path(ctx_two.models_dir)


def test_hotel_context_and_wrappers_switch_between_demo_and_custom(monkeypatch, tmp_path):
    manager = HotelManager(data_root=tmp_path)
    manager.create_hotel(HotelConfig(hotel_id="hotel_a", hotel_name="A", total_rooms=20))

    hotel_data_path = manager.get_hotel_data_path("hotel_a")
    pd.DataFrame(
        [{"arrival_date": "2024-01-01", "is_canceled": 0, "adr": 150.0}]
    ).to_csv(hotel_data_path, index=False)

    captured = []

    def fake_load(data_path, dataset_id):
        captured.append(("load", data_path, dataset_id))
        return data_path, dataset_id

    def fake_models(data_path, models_dir, dataset_id):
        captured.append(("models", data_path, models_dir, dataset_id))
        return {"dataset_id": dataset_id, "models_dir": models_dir}

    monkeypatch.setattr(app_mod, "_load_and_prepare_data_from_path", fake_load)
    monkeypatch.setattr(app_mod, "_get_or_train_models_for_context", fake_models)

    demo_ctx = build_demo_dataset_context()
    custom_ctx = resolve_hotel_dataset_context("hotel_a", manager=manager)

    assert app_mod.load_and_prepare_data(dataset_context=demo_ctx)[1] == "demo"
    assert app_mod.load_and_prepare_data(dataset_context=custom_ctx)[1] == custom_ctx.dataset_id
    assert app_mod.get_or_train_models(dataset_context=custom_ctx)["models_dir"] == custom_ctx.models_dir

    assert captured[0][1:] == (demo_ctx.data_path, demo_ctx.dataset_id)
    assert captured[1][1:] == (custom_ctx.data_path, custom_ctx.dataset_id)
    assert captured[2][1:] == (custom_ctx.data_path, custom_ctx.models_dir, custom_ctx.dataset_id)


def test_cached_loader_uses_dataset_id_in_cache_key(monkeypatch):
    app_mod._load_and_prepare_data_from_path.clear()

    calls = {"count": 0}

    def fake_load_data(path):
        calls["count"] += 1
        return pd.DataFrame({"arrival_date": pd.to_datetime(["2024-01-01"]), "is_canceled": [0]})

    monkeypatch.setattr(app_mod, "load_data", fake_load_data)
    monkeypatch.setattr(app_mod, "clean_data", lambda df: df)
    monkeypatch.setattr(app_mod, "engineer_features", lambda df: df.assign(revenue=0.0))
    monkeypatch.setattr(app_mod, "build_arrival_date", lambda df: df)
    monkeypatch.setattr(app_mod, "add_temporal_features", lambda df: df)
    monkeypatch.setattr(app_mod, "aggregate_daily", lambda df: pd.DataFrame({"daily_revenue": [0.0], "daily_bookings": [1], "daily_cancellations": [0]}))

    app_mod._load_and_prepare_data_from_path("same.csv", "demo")
    app_mod._load_and_prepare_data_from_path("same.csv", "demo")
    app_mod._load_and_prepare_data_from_path("same.csv", "custom_123")

    assert calls["count"] == 2


def test_get_or_train_models_rejects_tiny_daily_history(monkeypatch):
    app_mod._get_or_train_models_for_context.clear()

    tiny_daily = pd.DataFrame(
        {
            "daily_revenue": [125.0],
            "daily_booked_revenue": [125.0],
            "daily_realized_revenue": [125.0],
            "daily_expected_realized_revenue": [100.0],
            "daily_bookings": [1],
            "daily_cancellations": [0],
        },
        index=pd.to_datetime(["2024-01-01"]),
    )

    monkeypatch.setattr(
        app_mod,
        "_load_and_prepare_data_from_path",
        lambda data_path, dataset_id: (
            pd.DataFrame(
                {
                    "arrival_date": pd.to_datetime(["2024-01-01"]),
                    "is_canceled": [0],
                    "booked_revenue": [125.0],
                    "realized_revenue": [125.0],
                    "expected_realized_revenue": [100.0],
                }
            ),
            tiny_daily,
        ),
    )
    monkeypatch.setattr(
        app_mod.model_cache,
        "artifact_exists",
        lambda name, *args, **kwargs: name == "cancel_models",
    )
    monkeypatch.setattr(
        app_mod.model_cache,
        "load_artifact",
        lambda name, *args, **kwargs: {"xgboost": object()} if name == "cancel_models" else None,
    )
    # _artifact_current also reads version metadata from disk; without this the
    # mocked cancel_models cache is treated as stale and training runs anyway.
    monkeypatch.setattr(
        app_mod.model_cache,
        "load_artifact_with_metadata",
        lambda name, *args, **kwargs: (
            ({"xgboost": object()}, {"version": app_mod.CANCELLATION_MODEL_VERSION})
            if name == "cancel_models"
            else (None, {})
        ),
    )
    monkeypatch.setattr(
        app_mod,
        "_build_expected_revenue_views",
        lambda df, cancellation_model, feature_columns, label_encoders, feature_defaults: (df, tiny_daily),
    )
    monkeypatch.setattr(
        app_mod,
        "train_xgb_forecaster",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("xgb training should not run")),
    )
    monkeypatch.setattr(
        app_mod,
        "train_prophet_forecaster",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("prophet training should not run")),
    )

    try:
        app_mod._get_or_train_models_for_context("tiny.csv", "models/tiny", "tiny_dataset")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "tiny_dataset" in str(exc)
        assert "at least 31 daily observations" in str(exc)


def test_load_data_repairs_duplicated_arrival_date_columns(tmp_path):
    csv_path = tmp_path / "broken_upload.csv"
    csv_path.write_text(
        "\n".join(
            [
                "hotel,is_canceled,arrival_date,arrival_date.1,arrival_date.2,arrival_date.3,arrival_date.4,stays_in_weekend_nights,stays_in_week_nights,adults,children,babies,adr",
                "City Hotel,0,2026,August,34,22,2026-08-22,1,2,2,0,0,120.0",
                "City Hotel,0,2026,August,35,23,2026-08-23,1,2,2,0,0,110.0",
                "City Hotel,0,2026,August,36,24,2026-08-24,1,2,2,0,0,130.0",
            ]
        ),
        encoding="utf-8",
    )

    df = load_data(csv_path)
    dated = build_arrival_date(df)
    daily = aggregate_daily(
        dated.assign(
            revenue=[360.0, 330.0, 390.0],
            booked_revenue=[360.0, 330.0, 390.0],
            realized_revenue=[360.0, 330.0, 390.0],
        )
    )

    assert str(dated["arrival_date"].dt.date.iloc[0]) == "2026-08-22"
    assert len(daily) == 3


def test_normalize_uploaded_dataframe_prefers_full_arrival_date_series():
    from src.ingestion import normalize_uploaded_dataframe

    df = pd.DataFrame(
        {
            "arrival_date": [2026, 2026, 2026],
            "arrival_date.1": ["August", "August", "August"],
            "arrival_date.2": [34, 35, 36],
            "arrival_date.3": [22, 23, 24],
            "arrival_date.4": ["2026-08-22", "2026-08-23", "2026-08-24"],
            "is_canceled": [0, 0, 1],
        }
    )

    normalized = normalize_uploaded_dataframe(df)

    assert "arrival_date.4" not in normalized.columns
    assert normalized["arrival_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-08-22", "2026-08-23", "2026-08-24"
    ]


def test_build_expected_revenue_views_aligns_to_saved_feature_columns():
    class ShapeCheckingModel:
        def __init__(self):
            self.seen = None

        def predict_proba(self, X):
            import numpy as np
            self.seen = X
            assert X.shape == (2, 2)
            return np.array([[0.7, 0.3], [0.4, 0.6]])

    df = pd.DataFrame(
        {
            "arrival_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "is_canceled": [0, 1],
            "lead_time": [10, 20],
            "adr": [100.0, 200.0],
            "booked_revenue": [300.0, 400.0],
            "realized_revenue": [300.0, 0.0],
            "unexpected_extra_feature": [999, 888],
        }
    )

    model = ShapeCheckingModel()
    scored_df, revenue_daily = app_mod._build_expected_revenue_views(
        df,
        model,
        feature_columns=["adr", "lead_time"],
        label_encoders={},
        feature_defaults={"adr": 0.0, "lead_time": 0.0},
    )

    assert model.seen.shape == (2, 2)
    assert scored_df["cancel_probability"].tolist() == [0.3, 0.6]
    assert "daily_expected_realized_revenue" in revenue_daily.columns
