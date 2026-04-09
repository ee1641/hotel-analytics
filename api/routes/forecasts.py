from fastapi import APIRouter, HTTPException, Depends, Query
from api.auth import verify_api_key

router = APIRouter(prefix="/hotels/{hotel_id}/forecasts", tags=["forecasts"])


@router.get("")
def get_forecast(
    hotel_id: str,
    type: str = Query("revenue", enum=["revenue", "demand"]),
    horizon: int = Query(30, ge=1, le=90),
    hotel_id_from_key: str = Depends(verify_api_key),
):
    """Return revenue or demand forecast for the next `horizon` days."""
    try:
        from app import load_and_prepare_data, get_or_train_models
        from src.forecasting import forecast_xgb
        _, daily_df = load_and_prepare_data()
        models = get_or_train_models()
        key = "xgb_revenue" if type == "revenue" else "xgb_demand"
        series_key = "daily_revenue" if type == "revenue" else "daily_bookings"
        xgb_model, feat_cols = models[key]
        forecast = forecast_xgb(xgb_model, feat_cols, daily_df[series_key], periods=horizon)
        return {
            "status": "ok",
            "data": {
                "hotel_id": hotel_id,
                "type": type,
                "horizon": horizon,
                "forecast": [
                    {"date": str(d.date()), "value": round(v, 2)}
                    for d, v in zip(forecast.index, forecast.values)
                ],
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
