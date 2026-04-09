#!/usr/bin/env bash
set -e
echo "=== Hotel Analytics Setup ==="

echo "[1/4] Installing dependencies..."
pip install -r requirements.txt

echo "[2/4] Preparing data directory..."
mkdir -p data/hotels/demo/models outputs models

if [ ! -f data/hotels/demo/config.json ]; then
  echo '{"hotel_id":"demo","hotel_name":"Demo Hotel (Portugal)","total_rooms":100,"currency":"EUR","timezone":"Europe/Lisbon","overbooking_cost":150.0,"empty_room_cost":100.0}' > data/hotels/demo/config.json
fi

echo "[3/4] Running tests..."
pytest tests/ -v --tb=short -q

echo "[4/4] All done!"
echo ""
echo "  Start API:       uvicorn api.main:app --reload"
echo "  Start Dashboard: streamlit run app.py"
echo "  API Docs:        http://localhost:8000/docs"
