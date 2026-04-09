from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from api.auth import verify_api_key, create_api_key
from src.hotel_manager import HotelManager, HotelConfig, generate_hotel_id

router = APIRouter(prefix="/hotels", tags=["hotels"])
manager = HotelManager()


class HotelCreateRequest(BaseModel):
    hotel_name: str
    total_rooms: int
    currency: str = "EUR"
    timezone: str = "Europe/Lisbon"
    overbooking_cost: float = 150.0
    empty_room_cost: float = 100.0


@router.post("/")
def create_hotel(body: HotelCreateRequest):
    """Register a new hotel and return its hotel_id and API key."""
    hotel_id = generate_hotel_id()
    cfg = HotelConfig(hotel_id=hotel_id, **body.model_dump())
    manager.create_hotel(cfg)
    api_key = create_api_key(hotel_id)
    return {"status": "ok", "data": {"hotel_id": hotel_id, "api_key": api_key}}


@router.get("/{hotel_id}")
def get_hotel(hotel_id: str, _: str = Depends(verify_api_key)):
    """Get hotel config."""
    try:
        cfg = manager.get_hotel(hotel_id)
        from dataclasses import asdict
        return {"status": "ok", "data": asdict(cfg)}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Hotel {hotel_id} not found")


@router.get("/")
def list_hotels(_: str = Depends(verify_api_key)):
    """List all hotels."""
    from dataclasses import asdict
    return {"status": "ok", "data": [asdict(h) for h in manager.list_hotels()]}
