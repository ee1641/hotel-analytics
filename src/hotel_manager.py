"""
Multi-hotel management.

Provides HotelConfig and HotelManager for isolating each hotel's data
and model artifacts under data/hotels/{hotel_id}/.
"""
import json
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path

DATA_ROOT = Path("data/hotels")


@dataclass
class HotelConfig:
    hotel_id: str
    hotel_name: str
    total_rooms: int
    currency: str = "EUR"
    timezone: str = "Europe/Lisbon"
    overbooking_cost: float = 150.0
    empty_room_cost: float = 100.0


class HotelManager:
    """CRUD manager for hotel configurations stored as JSON files."""

    def __init__(self, data_root: Path | None = None):
        self._root = Path(data_root or DATA_ROOT)
        self._root.mkdir(parents=True, exist_ok=True)

    def create_hotel(self, config: HotelConfig) -> str:
        """
        Persist hotel config to disk. Creates directory structure.

        Returns
        -------
        str — the hotel_id
        """
        hotel_dir = self._root / config.hotel_id
        hotel_dir.mkdir(parents=True, exist_ok=True)
        (hotel_dir / "models").mkdir(exist_ok=True)
        (hotel_dir / "config.json").write_text(
            json.dumps(asdict(config), indent=2)
        )
        return config.hotel_id

    def get_hotel(self, hotel_id: str) -> HotelConfig:
        """
        Load a hotel config by ID.

        Raises
        ------
        FileNotFoundError if hotel_id is not registered.
        """
        path = self._root / hotel_id / "config.json"
        if not path.exists():
            raise FileNotFoundError(f"Hotel '{hotel_id}' not found at {path}")
        data = json.loads(path.read_text())
        return HotelConfig(**data)

    def list_hotels(self) -> list[HotelConfig]:
        """Return all registered hotels sorted by hotel_id."""
        configs = []
        for d in sorted(self._root.iterdir()):
            cfg_path = d / "config.json"
            if d.is_dir() and cfg_path.exists():
                try:
                    configs.append(HotelConfig(**json.loads(cfg_path.read_text())))
                except Exception:
                    pass
        return configs

    def get_hotel_data_path(self, hotel_id: str) -> Path:
        """Return the path where this hotel's bookings CSV should live."""
        return self._root / hotel_id / "bookings.csv"

    def get_hotel_models_path(self, hotel_id: str) -> Path:
        """Return the directory for this hotel's model artifacts."""
        return self._root / hotel_id / "models"


def generate_hotel_id() -> str:
    """Generate a short unique hotel ID like 'hotel_a3f2c1b0'."""
    return f"hotel_{uuid.uuid4().hex[:8]}"
