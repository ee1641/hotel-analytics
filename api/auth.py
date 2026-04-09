import json
import secrets
from pathlib import Path
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

_KEYS_FILE = Path("data/api_keys.json")
_header = APIKeyHeader(name="X-API-Key", auto_error=True)


def _load_keys() -> dict:
    if _KEYS_FILE.exists():
        return json.loads(_KEYS_FILE.read_text())
    return {}


def _save_keys(keys: dict) -> None:
    _KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _KEYS_FILE.write_text(json.dumps(keys, indent=2))


def create_api_key(hotel_id: str) -> str:
    keys = _load_keys()
    key = secrets.token_urlsafe(32)
    keys[key] = hotel_id
    _save_keys(keys)
    return key


def verify_api_key(api_key: str = Security(_header)) -> str:
    """Return hotel_id if key is valid, else raise 403."""
    keys = _load_keys()
    if api_key not in keys:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return keys[api_key]
