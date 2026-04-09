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
        from src.ingestion import auto_detect_columns, validate_schema, map_columns
        from src.hotel_manager import HotelManager
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        mapping = auto_detect_columns(df)
        df_mapped = map_columns(df, mapping)
        valid, issues = validate_schema(df_mapped)
        if not valid:
            raise HTTPException(status_code=400, detail={"issues": issues})
        manager = HotelManager()
        data_path = manager.get_hotel_data_path(hotel_id)
        df_mapped.to_csv(data_path, index=False)
        return {
            "status": "ok",
            "data": {
                "hotel_id": hotel_id,
                "rows": len(df_mapped),
                "columns_mapped": mapping,
                "validation_issues": issues,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
