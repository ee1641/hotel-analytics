"""Tests for the FastAPI backend (api/main.py)."""
import pytest
import json
from pathlib import Path


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient backed by a temp directory for hotel data and API keys."""
    # Redirect hotel data root and API keys file to tmp_path
    import src.hotel_manager as hm_mod
    monkeypatch.setattr(hm_mod, "DATA_ROOT", tmp_path)

    import api.auth as auth_mod
    monkeypatch.setattr(auth_mod, "_KEYS_FILE", tmp_path / "api_keys.json")

    # Re-import after patching so the router's manager picks up the new root
    import importlib
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


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Hotel CRUD
# ---------------------------------------------------------------------------

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
