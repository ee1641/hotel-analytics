import json
from datetime import datetime
from pathlib import Path

_LOG_FILE = Path("data/analytics.jsonl")


def track_event(event_type: str, page: str, data: dict = None) -> None:
    """Append a single event to the analytics log (JSON Lines format)."""
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "page": page,
        "data": data or {},
    }
    with open(_LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")


def get_events(limit: int = 100) -> list[dict]:
    """Return the last `limit` events."""
    if not _LOG_FILE.exists():
        return []
    lines = _LOG_FILE.read_text().strip().splitlines()
    return [json.loads(line) for line in lines[-limit:]]
