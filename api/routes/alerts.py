from fastapi import APIRouter, HTTPException, Depends
from api.auth import verify_api_key

router = APIRouter(prefix="/hotels/{hotel_id}/alerts", tags=["alerts"])


@router.get("")
def get_alerts(hotel_id: str, _: str = Depends(verify_api_key)):
    """Return today's alerts for the hotel."""
    try:
        from src.services import load_and_prepare_data, get_or_train_models
        from src.alerts import generate_daily_alerts
        from src.forecasting import forecast_xgb
        df, daily_df = load_and_prepare_data()
        models = get_or_train_models()
        xgb_model, feat_cols = models["xgb_demand"]
        forecast = forecast_xgb(xgb_model, feat_cols, daily_df["daily_bookings"], periods=30)
        alerts = generate_daily_alerts(df, forecast, models["cancel_models"]["xgboost"], {})
        from dataclasses import asdict
        return {"status": "ok", "data": [asdict(a) for a in alerts]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
