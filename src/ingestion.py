"""
Data ingestion layer — schema mapping, validation, and auto-detection.
Allows any hotel CSV to be mapped to the canonical schema used by the ML pipeline.
"""
import json
import difflib
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Load canonical schema
# ---------------------------------------------------------------------------
_SCHEMA_PATH = Path(__file__).parent / "schemas" / "canonical_schema.json"
with open(_SCHEMA_PATH, "r", encoding="utf-8") as _f:
    CANONICAL_SCHEMA: dict = json.load(_f)

# ---------------------------------------------------------------------------
# Second dataset mapping (INN Hotels Group — cancellation_Prediction.csv)
# ---------------------------------------------------------------------------
CANCELLATION_CSV_MAPPING: dict[str, str] = {
    "number of adults":    "adults",
    "number of children":  "children",
    "number of weekend nights": "stays_in_weekend_nights",
    "number of week nights":    "stays_in_week_nights",
    "type of meal":        "meal",
    "car parking space":   "required_car_parking_spaces",
    "room type":           "reserved_room_type",
    "lead time":           "lead_time",
    "market segment type": "market_segment",
    "repeated":            "is_repeated_guest",
    "P-C":                 "previous_cancellations",
    "P-not-C":             "previous_bookings_not_canceled",
    "average price":       "adr",
    "special requests":    "total_of_special_requests",
    "date of reservation": "arrival_date",
    "booking status":      "is_canceled",
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def map_columns(df: pd.DataFrame, column_mapping: dict) -> pd.DataFrame:
    """
    Rename df columns using mapping {user_col: canonical_col}.
    Columns not in the mapping are kept as-is.
    """
    return df.rename(columns=column_mapping)


def validate_schema(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Check that the DataFrame satisfies the canonical schema requirements.

    Returns:
        (is_valid, issues)  where issues is an empty list if valid.
    """
    issues = []
    required = CANONICAL_SCHEMA.get("required", [])
    col_specs = CANONICAL_SCHEMA.get("columns", {})

    for col in required:
        if col not in df.columns:
            issues.append(f"Missing required column: '{col}'")

    for col, spec in col_specs.items():
        if col not in df.columns:
            continue
        expected_type = spec.get("type")
        if expected_type == "int":
            if not pd.api.types.is_integer_dtype(df[col]) and not pd.api.types.is_float_dtype(df[col]):
                issues.append(f"Column '{col}' expected int/float, got {df[col].dtype}")
        elif expected_type == "float":
            if not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"Column '{col}' expected numeric, got {df[col].dtype}")

    return (len(issues) == 0), issues


# Common alias → canonical name lookup (checked before fuzzy matching)
_KNOWN_ALIASES: dict[str, str] = {
    "average price": "adr",
    "average daily rate": "adr",
    "rate": "adr",
    "price": "adr",
    "canceled": "is_canceled",
    "cancelled": "is_canceled",
    "booking status": "is_canceled",
    "cancellation": "is_canceled",
    "num adults": "adults",
    "no adults": "adults",
    "num children": "children",
    "no children": "children",
    "weekend nights": "stays_in_weekend_nights",
    "weeknight": "stays_in_week_nights",
    "week nights": "stays_in_week_nights",
    "special requests": "total_of_special_requests",
    "num special requests": "total_of_special_requests",
    "car parking": "required_car_parking_spaces",
    "parking": "required_car_parking_spaces",
    "room type": "reserved_room_type",
    "market segment type": "market_segment",
    "segment": "market_segment",
    "repeated": "is_repeated_guest",
    "repeat guest": "is_repeated_guest",
    "date of reservation": "arrival_date",
    "reservation date": "arrival_date",
    "check in": "arrival_date",
}


def auto_detect_columns(df: pd.DataFrame, similarity_threshold: float = 0.6) -> dict:
    """
    Use difflib.SequenceMatcher to suggest column mappings from the uploaded CSV
    to the canonical schema.

    Returns:
        dict {user_col: canonical_col} for matches above similarity_threshold.
        Only columns with a clear best match are included.
    """
    canonical_cols = list(CANONICAL_SCHEMA["columns"].keys())
    mapping = {}

    for user_col in df.columns:
        user_lower = user_col.lower().replace("_", " ").strip()

        # 1. Check known aliases first (exact match after normalisation)
        if user_lower in _KNOWN_ALIASES:
            mapping[user_col] = _KNOWN_ALIASES[user_lower]
            continue

        # 2. Fuzzy string similarity against canonical names
        best_match = None
        best_score = 0.0
        for canon_col in canonical_cols:
            canon_lower = canon_col.lower().replace("_", " ").strip()
            score = difflib.SequenceMatcher(None, user_lower, canon_lower).ratio()
            if score > best_score:
                best_score = score
                best_match = canon_col

        if best_score >= similarity_threshold and best_match:
            mapping[user_col] = best_match

    return mapping


def load_cancellation_prediction_csv(path: str = "cancellation_Prediction.csv") -> pd.DataFrame:
    """
    Load and map the INN Hotels Group cancellation dataset to the canonical schema.
    Converts booking status text labels to binary int (Canceled=1, Not_Canceled=0).
    Parses date of reservation (M/D/YYYY) to datetime.
    Drops the Booking_ID column.
    """
    df = pd.read_csv(path)

    # Drop ID column
    if "Booking_ID" in df.columns:
        df = df.drop(columns=["Booking_ID"])

    # Rename to canonical names
    df = map_columns(df, CANCELLATION_CSV_MAPPING)

    # Convert booking status → binary int
    if "is_canceled" in df.columns:
        df["is_canceled"] = df["is_canceled"].map(
            {"Canceled": 1, "Not_Canceled": 0}
        ).fillna(0).astype(int)

    # Parse arrival_date (M/D/YYYY format)
    if "arrival_date" in df.columns:
        df["arrival_date"] = pd.to_datetime(df["arrival_date"], format="%m/%d/%Y", errors="coerce")

    return df


def prepare_kaggle_demo(path: str = "hotel_bookings.csv") -> pd.DataFrame:
    """
    Load the primary Kaggle hotel_bookings dataset.
    Identity mapping — column names already match canonical schema.
    """
    return pd.read_csv(path)
