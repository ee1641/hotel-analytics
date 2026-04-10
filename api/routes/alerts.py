from fastapi import APIRouter, HTTPException, Depends
from api.auth import verify_api_key
from src.data_context import resolve_hotel_dataset_context

router = APIRouter(prefix="/hotels/{hotel_id}/alerts", tags=["alerts"])


@router.get("")
def get_alerts(hotel_id: str, _: str = Depends(verify_api_key)):
    """Return today's alerts for the hotel."""
    try:
        from app import load_and_prepare_data, get_or_train_models
        from src.alerts import generate_daily_alerts
        from src.forecasting import forecast_xgb
        context = resolve_hotel_dataset_context(hotel_id)
        df, daily_df = load_and_prepare_data(dataset_context=context)
        models = get_or_train_models(dataset_context=context)
        xgb_model, feat_cols = models["xgb_demand"]
        forecast = forecast_xgb(xgb_model, feat_cols, daily_df["daily_bookings"], periods=30)
        alerts = generate_daily_alerts(df, forecast, models["cancel_models"]["xgboost"], {})
        from dataclasses import asdict
        return {"status": "ok", "data": [asdict(a) for a in alerts]}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
