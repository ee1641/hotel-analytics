from fastapi import APIRouter, HTTPException, Depends, Query
import pandas as pd
from api.auth import verify_api_key, check_hotel_ownership
from src.data_context import resolve_hotel_dataset_context

router = APIRouter(prefix="/hotels/{hotel_id}/forecasts", tags=["forecasts"])


@router.get("")
def get_forecast(
    hotel_id: str,
    type: str = Query("revenue", enum=["revenue", "demand"]),
    horizon: int = Query(30, ge=1, le=90),
    include_intervals: bool = Query(False),
    authenticated_hotel_id: str = Depends(verify_api_key),
):
    """Return revenue or demand forecast for the next `horizon` days."""
    check_hotel_ownership(hotel_id, authenticated_hotel_id)
    try:
        from src.services import load_and_prepare_data, get_or_train_models
        from src.forecasting import forecast_xgb, forecast_xgb_with_intervals
        context = resolve_hotel_dataset_context(hotel_id)
        _, daily_df = load_and_prepare_data(dataset_context=context)
        models = get_or_train_models(dataset_context=context)
        key = "xgb_revenue" if type == "revenue" else "xgb_demand"
        interval_key = "xgb_revenue_intervals" if type == "revenue" else "xgb_demand_intervals"
        series_key = "daily_revenue" if type == "revenue" else "daily_bookings"

        if include_intervals and interval_key in models:
            forecast_df = forecast_xgb_with_intervals(models[interval_key], daily_df[series_key], periods=horizon)
            forecast_payload = [
                {
                    "date": str(pd.Timestamp(row.ds).date()),
                    "value": round(float(row.yhat), 2),
                    "low": round(float(row.yhat_lower), 2),
                    "high": round(float(row.yhat_upper), 2),
                }
                for row in forecast_df.itertuples(index=False)
            ]
        else:
            xgb_model, feat_cols = models[key]
            forecast = forecast_xgb(xgb_model, feat_cols, daily_df[series_key], periods=horizon)
            forecast_payload = [
                {"date": str(d.date()), "value": round(v, 2)}
                for d, v in zip(forecast.index, forecast.values)
            ]

        return {
            "status": "ok",
            "data": {
                "hotel_id": hotel_id,
                "type": type,
                "horizon": horizon,
                "forecast": forecast_payload,
            },
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
