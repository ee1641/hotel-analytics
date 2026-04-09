"""Tests for src/hotel_manager.py"""
import pytest
import json
from pathlib import Path
from src.hotel_manager import HotelManager, HotelConfig, generate_hotel_id


@pytest.fixture
def tmp_manager(tmp_path):
    """HotelManager backed by a temp directory."""
    return HotelManager(data_root=tmp_path)


# ---------------------------------------------------------------------------
# create_hotel + get_hotel
# ---------------------------------------------------------------------------

def test_create_and_get_hotel(tmp_manager):
    cfg = HotelConfig(hotel_id="h1", hotel_name="Test Hotel", total_rooms=50)
    tmp_manager.create_hotel(cfg)
    loaded = tmp_manager.get_hotel("h1")
    assert loaded.hotel_name == "Test Hotel"
    assert loaded.total_rooms == 50
    assert loaded.hotel_id == "h1"


def test_create_hotel_writes_config_json(tmp_manager, tmp_path):
    cfg = HotelConfig(hotel_id="h2", hotel_name="H2", total_rooms=80)
    tmp_manager.create_hotel(cfg)
    config_path = tmp_path / "h2" / "config.json"
    assert config_path.exists()
    data = json.loads(config_path.read_text())
    assert data["hotel_name"] == "H2"


def test_create_hotel_creates_models_dir(tmp_manager, tmp_path):
    cfg = HotelConfig(hotel_id="h3", hotel_name="H3", total_rooms=60)
    tmp_manager.create_hotel(cfg)
    assert (tmp_path / "h3" / "models").is_dir()


def test_get_hotel_missing_raises_file_not_found(tmp_manager):
    with pytest.raises(FileNotFoundError, match="nonexistent"):
        tmp_manager.get_hotel("nonexistent")


def test_hotel_config_defaults(tmp_manager):
    cfg = HotelConfig(hotel_id="h4", hotel_name="H4", total_rooms=100)
    tmp_manager.create_hotel(cfg)
    loaded = tmp_manager.get_hotel("h4")
    assert loaded.currency == "EUR"
    assert loaded.overbooking_cost == 150.0


# ---------------------------------------------------------------------------
# list_hotels
# ---------------------------------------------------------------------------

def test_list_hotels_returns_all(tmp_manager):
    for i in range(3):
        tmp_manager.create_hotel(
            HotelConfig(hotel_id=f"h{i}", hotel_name=f"Hotel {i}", total_rooms=50)
        )
    hotels = tmp_manager.list_hotels()
    assert len(hotels) == 3


def test_list_hotels_sorted_by_id(tmp_manager):
    for hid in ["hc", "ha", "hb"]:
        tmp_manager.create_hotel(HotelConfig(hotel_id=hid, hotel_name=hid, total_rooms=50))
    ids = [h.hotel_id for h in tmp_manager.list_hotels()]
    assert ids == sorted(ids)


def test_list_hotels_empty(tmp_manager):
    assert tmp_manager.list_hotels() == []


# ---------------------------------------------------------------------------
# Data isolation
# ---------------------------------------------------------------------------

def test_data_paths_are_isolated(tmp_manager):
    tmp_manager.create_hotel(HotelConfig(hotel_id="a", hotel_name="A", total_rooms=50))
    tmp_manager.create_hotel(HotelConfig(hotel_id="b", hotel_name="B", total_rooms=100))
    path_a = tmp_manager.get_hotel_data_path("a")
    path_b = tmp_manager.get_hotel_data_path("b")
    assert path_a != path_b
    assert "a" in str(path_a)
    assert "b" in str(path_b)


def test_model_paths_are_isolated(tmp_manager):
    path_x = tmp_manager.get_hotel_models_path("x")
    path_y = tmp_manager.get_hotel_models_path("y")
    assert path_x != path_y


# ---------------------------------------------------------------------------
# generate_hotel_id
# ---------------------------------------------------------------------------

def test_generate_hotel_id_format():
    hid = generate_hotel_id()
    assert hid.startswith("hotel_")
    assert len(hid) == len("hotel_") + 8


def test_generate_hotel_id_unique():
    ids = {generate_hotel_id() for _ in range(200)}
    assert len(ids) == 200


# ---------------------------------------------------------------------------
# Demo hotel config on disk
# ---------------------------------------------------------------------------

def test_demo_hotel_config_exists():
    demo_path = Path("data/hotels/demo/config.json")
    assert demo_path.exists(), "data/hotels/demo/config.json must exist"
    data = json.loads(demo_path.read_text())
    assert data["hotel_id"] == "demo"
    assert data["total_rooms"] > 0
