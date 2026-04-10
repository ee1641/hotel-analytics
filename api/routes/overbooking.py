from fastapi import APIRouter, HTTPException, Depends, Query
from datetime import date
from api.auth import verify_api_key
from src.data_context import resolve_hotel_dataset_context

router = APIRouter(prefix="/hotels/{hotel_id}/overbooking", tags=["overbooking"])


@router.get("")
def get_overbooking(
    hotel_id: str,
    target_date: date = Query(...),
    total_rooms: int = Query(100),
    overbooking_cost: float = Query(150.0),
    empty_room_cost: float = Query(100.0),
    _: str = Depends(verify_api_key),
):
    """Return overbooking recommendation for a specific date."""
    try:
        from app import load_and_prepare_data, get_or_train_models
        from src.overbooking import calculate_overbooking_recommendation
        context = resolve_hotel_dataset_context(hotel_id)
        df, _ = load_and_prepare_data(dataset_context=context)
        models = get_or_train_models(dataset_context=context)
        rec = calculate_overbooking_recommendation(
            df, models["cancel_models"]["xgboost"], target_date,
            total_rooms, overbooking_cost, empty_room_cost,
        )
        return {"status": "ok", "data": {"hotel_id": hotel_id, "date": str(target_date), **rec}}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
