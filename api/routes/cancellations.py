from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from api.auth import verify_api_key

router = APIRouter(prefix="/hotels/{hotel_id}", tags=["cancellations"])


class BookingInput(BaseModel):
    lead_time: int
    adults: int = 2
    children: int = 0
    stays_in_weekend_nights: int = 1
    stays_in_week_nights: int = 3
    meal: str = "BB"
    market_segment: str = "Online TA"
    is_repeated_guest: int = 0
    previous_cancellations: int = 0
    previous_bookings_not_canceled: int = 0
    adr: float = 100.0
    total_of_special_requests: int = 0
    hotel: str = "City Hotel"
    customer_type: str = "Transient"


@router.post("/predict-cancellation")
def predict_cancellation(
    hotel_id: str,
    booking: BookingInput,
    _: str = Depends(verify_api_key),
):
    """Predict cancellation probability for a single booking."""
    try:
        import pandas as pd
        from app import get_or_train_models
        from src.preprocessing import drop_leaky_features, encode_categoricals
        models = get_or_train_models()
        df = pd.DataFrame([booking.model_dump()])
        X, _, _ = encode_categoricals(drop_leaky_features(df))
        model = models["cancel_models"]["xgboost"]
        proba = float(model.predict_proba(X.values)[0][1])
        risk = "High" if proba > 0.6 else ("Medium" if proba > 0.3 else "Low")
        return {"status": "ok", "data": {"cancel_probability": round(proba, 4), "risk_tier": risk}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
