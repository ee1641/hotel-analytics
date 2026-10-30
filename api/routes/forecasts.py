from fastapi import APIRouter, HTTPException, Depends, Query
import pandas as pd
from api.auth import verify_api_key, check_hotel_ownership
from src.conformal import forecast_conformal
from src.data_context import resolve_hotel_dataset_context

router = APIRouter(prefix="/hotels/{hotel_id}/forecasts", tags=["forecasts"])


@router.get("")
def get_forecast(
    hotel_id: str,
    type: str = Query("revenue", enum=["revenue", "demand"]),
    horizon: int = Query(30, ge=1, le=90),
    include_intervals: bool = Query(False),
    model: str = Query("xgb", enum=["xgb", "ensemble", "direct", "prophet"]),
    authenticated_hotel_id: str = Depends(verify_api_key),
):
    """
    Return revenue or demand forecast for the next `horizon` days.

    `model` selects the point forecaster: xgb (recursive, default), ensemble
    (XGB+Prophet blend using cached weights), direct (per-horizon multi-step),
    or prophet. `include_intervals` always uses the conformal/quantile bands and
    ignores `model`.
    """
    check_hotel_ownership(hotel_id, authenticated_hotel_id)
    try:
        from src.services import load_and_prepare_data, get_or_train_models
        from src.forecasting import (
            blend_forecasts, forecast_prophet, forecast_xgb,
            forecast_xgb_direct, forecast_xgb_with_intervals,
        )
        context = resolve_hotel_dataset_context(hotel_id)
        _, daily_df = load_and_prepare_data(dataset_context=context)
        models = get_or_train_models(dataset_context=context)
        key = "xgb_revenue" if type == "revenue" else "xgb_demand"
        interval_key = "xgb_revenue_intervals" if type == "revenue" else "xgb_demand_intervals"
        conformal_key = "xgb_revenue_conformal" if type == "revenue" else "xgb_demand_conformal"
        direct_key = "xgb_revenue_direct" if type == "revenue" else "xgb_demand_direct"
        prophet_key = "prophet_revenue" if type == "revenue" else "prophet_demand"
        series_key = "daily_revenue" if type == "revenue" else "daily_bookings"
        series = daily_df[series_key]

        if include_intervals and (conformal_key in models or interval_key in models):
            if conformal_key in models and models[conformal_key] is not None:
                forecast_df = forecast_conformal(
                    models[conformal_key], series, periods=horizon,
                )
            else:
                forecast_df = forecast_xgb_with_intervals(
                    models[interval_key], series, periods=horizon,
                )
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
            xgb_fc = forecast_xgb(xgb_model, feat_cols, series, periods=horizon)
            if model == "prophet" and models.get(prophet_key) is not None:
                forecast = pd.Series(
                    forecast_prophet(models[prophet_key], periods=horizon)["yhat"].values,
                    index=xgb_fc.index,
                )
            elif model == "direct" and models.get(direct_key) is not None:
                direct_fc = forecast_xgb_direct(models[direct_key], series)
                forecast = pd.Series(direct_fc.values[:horizon], index=xgb_fc.index[:len(direct_fc)])
            elif model == "ensemble" and models.get(prophet_key) is not None:
                # Cached weight = XGB share of the blend; missing → falls back to XGB.
                weight = float((models.get("ensemble_weights") or {}).get(type, 1.0))
                prophet_df = forecast_prophet(models[prophet_key], periods=horizon)
                forecast = blend_forecasts(xgb_fc, prophet_df, weight)
            else:
                forecast = xgb_fc  # default / fallback
            forecast_payload = [
                {"date": str(d.date()), "value": round(float(v), 2)}
                for d, v in zip(forecast.index, forecast.values)
            ]

        return {
            "status": "ok",
            "data": {
                "hotel_id": hotel_id,
                "type": type,
                "horizon": horizon,
                "model": "intervals" if include_intervals else model,
                "forecast": forecast_payload,
            },
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
