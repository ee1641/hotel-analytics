from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from api.auth import verify_api_key
import io

router = APIRouter(prefix="/hotels/{hotel_id}/upload-data", tags=["upload"])


@router.post("")
async def upload_data(
    hotel_id: str,
    file: UploadFile = File(...),
    _: str = Depends(verify_api_key),
):
    """Upload a hotel bookings CSV. Auto-detects and maps columns to canonical schema."""
    try:
        import pandas as pd
        from src.data_context import clear_models_dir, resolve_hotel_dataset_context
        from src.ingestion import auto_detect_columns, normalize_uploaded_dataframe, validate_schema, map_columns
        from src.hotel_manager import HotelManager
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        mapping = auto_detect_columns(df)
        df_mapped = normalize_uploaded_dataframe(map_columns(df, mapping))
        valid, issues = validate_schema(df_mapped)
        if not valid:
            raise HTTPException(status_code=400, detail={"issues": issues})
        manager = HotelManager()
        data_path = manager.get_hotel_data_path(hotel_id)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df_mapped.to_csv(data_path, index=False)
        models_dir = manager.get_hotel_models_path(hotel_id)
        models_dir.mkdir(parents=True, exist_ok=True)
        clear_models_dir(models_dir)
        context = resolve_hotel_dataset_context(hotel_id, manager=manager)
        return {
            "status": "ok",
            "data": {
                "hotel_id": hotel_id,
                "dataset_id": context.dataset_id,
                "rows": len(df_mapped),
                "data_path": str(data_path),
                "models_dir": str(models_dir),
                "columns_mapped": mapping,
                "validation_issues": issues,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
