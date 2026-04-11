"""Tests for the FastAPI backend (api/main.py)."""
from pathlib import Path

import pandas as pd
import pytest


def _sample_upload_csv(adr: float = 100.0) -> bytes:
    df = pd.DataFrame(
        [
            {"arrival_date": "2024-01-01", "is_canceled": 0, "adr": adr, "hotel": "City Hotel"},
            {"arrival_date": "2024-01-02", "is_canceled": 1, "adr": adr + 25, "hotel": "City Hotel"},
        ]
    )
    return df.to_csv(index=False).encode("utf-8")


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient backed by a temp directory for hotel data and API keys."""
    import src.hotel_manager as hm_mod
    monkeypatch.setattr(hm_mod, "DATA_ROOT", tmp_path)

    import api.auth as auth_mod
    monkeypatch.setattr(auth_mod, "_KEYS_FILE", tmp_path / "api_keys.json")

    import api.routes.hotels as hotels_mod
    monkeypatch.setattr(hotels_mod, "manager", hm_mod.HotelManager(data_root=tmp_path))

    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


@pytest.fixture
def hotel_and_key(client):
    """Create a hotel and return (hotel_id, api_key)."""
    resp = client.post("/hotels/", json={"hotel_name": "Test Hotel", "total_rooms": 50})
    assert resp.status_code == 200
    data = resp.json()["data"]
    return data["hotel_id"], data["api_key"]


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_create_hotel_returns_key(client):
    resp = client.post("/hotels/", json={"hotel_name": "H", "total_rooms": 100})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "api_key" in body["data"]
    assert "hotel_id" in body["data"]


def test_get_hotel_requires_auth(client, hotel_and_key):
    hotel_id, _ = hotel_and_key
    resp = client.get(f"/hotels/{hotel_id}")
    assert resp.status_code in (401, 403)


def test_get_hotel_with_valid_key(client, hotel_and_key):
    hotel_id, api_key = hotel_and_key
    resp = client.get(f"/hotels/{hotel_id}", headers={"X-API-Key": api_key})
    assert resp.status_code == 200
    assert resp.json()["data"]["hotel_id"] == hotel_id


def test_invalid_api_key_rejected(client, hotel_and_key):
    hotel_id, _ = hotel_and_key
    resp = client.get(f"/hotels/{hotel_id}", headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401  # invalid key = 401 Unauthorized


def test_wrong_hotel_id_returns_403(client, hotel_and_key):
    """A valid key used to access a different hotel returns 403 Forbidden."""
    hotel_id, api_key = hotel_and_key
    resp = client.get("/hotels/some-other-hotel-id", headers={"X-API-Key": api_key})
    assert resp.status_code == 403


def test_valid_key_own_hotel_returns_200(client, hotel_and_key):
    hotel_id, api_key = hotel_and_key
    resp = client.get(f"/hotels/{hotel_id}", headers={"X-API-Key": api_key})
    assert resp.status_code == 200


def test_cross_hotel_access_blocked(client):
    """Hotel A's key must not access Hotel B's data."""
    resp_a = client.post("/hotels/", json={"hotel_name": "Hotel A", "total_rooms": 50})
    hotel_b_resp = client.post("/hotels/", json={"hotel_name": "Hotel B", "total_rooms": 50})
    key_a = resp_a.json()["data"]["api_key"]
    hotel_b_id = hotel_b_resp.json()["data"]["hotel_id"]

    resp = client.get(f"/hotels/{hotel_b_id}", headers={"X-API-Key": key_a})
    assert resp.status_code == 403


def test_list_hotels_requires_auth(client):
    resp = client.get("/hotels/")
    assert resp.status_code in (401, 403)


def test_list_hotels_with_auth(client, hotel_and_key):
    _, api_key = hotel_and_key
    resp = client.get("/hotels/", headers={"X-API-Key": api_key})
    assert resp.status_code == 200
    assert isinstance(resp.json()["data"], list)


def test_hotel_forecast_route_uses_uploaded_dataset_context(client, hotel_and_key, monkeypatch, tmp_path):
    hotel_id, api_key = hotel_and_key

    second = client.post("/hotels/", json={"hotel_name": "Other Hotel", "total_rooms": 10}).json()["data"]
    other_hotel_id = second["hotel_id"]
    other_models_dir = tmp_path / other_hotel_id / "models"
    other_models_dir.mkdir(parents=True, exist_ok=True)
    other_marker = other_models_dir / "keep.pkl"
    other_marker.write_text("keep")

    models_dir = tmp_path / hotel_id / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    stale_model = models_dir / "stale.pkl"
    stale_model.write_text("stale")

    upload_resp = client.post(
        f"/hotels/{hotel_id}/upload-data",
        headers={"X-API-Key": api_key},
        files={"file": ("bookings.csv", _sample_upload_csv(adr=210.0), "text/csv")},
    )
    assert upload_resp.status_code == 200
    body = upload_resp.json()["data"]
    assert body["data_path"] == str(tmp_path / hotel_id / "bookings.csv")
    assert body["models_dir"] == str(models_dir)
    assert not stale_model.exists()
    assert other_marker.exists()

    import app as app_mod

    captured = {}

    def fake_load_and_prepare_data(*, dataset_context=None, **kwargs):
        captured["dataset_context"] = dataset_context
        daily_df = pd.DataFrame(
            {"daily_revenue": [100.0, 120.0], "daily_bookings": [1, 2]},
            index=pd.date_range("2024-01-01", periods=2),
        )
        return pd.DataFrame({"arrival_date": pd.to_datetime(["2024-01-01"])}), daily_df

    def fake_get_or_train_models(*, dataset_context=None, **kwargs):
        captured["models_context"] = dataset_context
        return {"xgb_revenue": ("model", ["f1"]), "xgb_demand": ("model", ["f1"])}

    monkeypatch.setattr(app_mod, "load_and_prepare_data", fake_load_and_prepare_data)
    monkeypatch.setattr(app_mod, "get_or_train_models", fake_get_or_train_models)

    import src.forecasting as forecasting_mod
    monkeypatch.setattr(
        forecasting_mod,
        "forecast_xgb",
        lambda model, feat_cols, series, periods: pd.Series(
            [float(i) for i in range(periods)],
            index=pd.date_range("2024-02-01", periods=periods),
        ),
    )

    forecast_resp = client.get(
        f"/hotels/{hotel_id}/forecasts",
        headers={"X-API-Key": api_key},
        params={"type": "revenue", "horizon": 2},
    )
    assert forecast_resp.status_code == 200
    assert captured["dataset_context"].data_path == str(tmp_path / hotel_id / "bookings.csv")
    assert captured["dataset_context"].models_dir == str(tmp_path / hotel_id / "models")
    assert captured["models_context"].dataset_id == captured["dataset_context"].dataset_id


def test_non_demo_hotel_without_uploaded_data_returns_404(client, hotel_and_key):
    hotel_id, api_key = hotel_and_key
    resp = client.get(
        f"/hotels/{hotel_id}/forecasts",
        headers={"X-API-Key": api_key},
        params={"type": "revenue", "horizon": 2},
    )
    assert resp.status_code == 404
    assert "Upload bookings CSV" in resp.json()["detail"]


def test_forecast_route_returns_400_for_tiny_uploaded_dataset(client, hotel_and_key, monkeypatch):
    hotel_id, api_key = hotel_and_key

    upload_resp = client.post(
        f"/hotels/{hotel_id}/upload-data",
        headers={"X-API-Key": api_key},
        files={"file": ("bookings.csv", _sample_upload_csv(adr=90.0), "text/csv")},
    )
    assert upload_resp.status_code == 200

    import app as app_mod
    monkeypatch.setattr(
        app_mod,
        "load_and_prepare_data",
        lambda *args, **kwargs: (
            pd.DataFrame({"arrival_date": pd.to_datetime(["2024-01-01"])}),
            pd.DataFrame(
                {"daily_revenue": [90.0], "daily_bookings": [1], "daily_cancellations": [0]},
                index=pd.to_datetime(["2024-01-01"]),
            ),
        ),
    )
    monkeypatch.setattr(
        app_mod,
        "get_or_train_models",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("Dataset 'tiny_dataset' has only 1 valid daily revenue rows after preprocessing. Forecasting requires at least 31 daily observations.")
        ),
    )

    resp = client.get(
        f"/hotels/{hotel_id}/forecasts",
        headers={"X-API-Key": api_key},
        params={"type": "revenue", "horizon": 2},
    )
    assert resp.status_code == 400
    assert "at least 31 daily observations" in resp.json()["detail"]
