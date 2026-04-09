# Hotel Analytics Platform

ML-powered hotel analytics: predict cancellations, optimize overbooking, and forecast demand — all from a single CSV upload.

Built with Python, XGBoost, Prophet, SHAP, Streamlit, and FastAPI.

## Quick Start

```bash
docker-compose up
```

- Dashboard: http://localhost:8501
- API: http://localhost:8000/docs

## Manual Setup

```bash
bash scripts/setup.sh
uvicorn api.main:app --reload &
streamlit run app.py
```

## Architecture

The platform has two layers: a Streamlit dashboard for interactive exploration and a FastAPI backend for programmatic access. Both share the same ML core (`src/`) which handles preprocessing, XGBoost/Prophet forecasting, SHAP-based cancellation prediction, and overbooking optimization. Each hotel's data and models are isolated under `data/hotels/{hotel_id}/`.

## Pages

| Page | Description |
|------|-------------|
| Overview | KPIs, revenue & bookings charts, cross-dataset AUC |
| Revenue Forecast | XGBoost, Prophet, Naive comparison |
| Demand & Staffing | 30-day demand + staffing estimate |
| Cancellation Predictor | SHAP-selected form + top-3 risk factors |
| Profit Simulator | Tax/cost sliders |
| Overbooking Engine | Newsvendor-optimal overbooking recommendation |
| Alerts | Severity-sorted actionable alerts with email preview |
| Landing | Product overview and demo CTA |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/hotels/` | Register hotel, get API key |
| GET | `/hotels/{id}` | Get hotel config |
| GET | `/hotels/{id}/forecasts` | Revenue or demand forecast |
| POST | `/hotels/{id}/predict-cancellation` | Single booking risk score |
| GET | `/hotels/{id}/overbooking` | Overbooking recommendation |
| GET | `/hotels/{id}/alerts` | Today's actionable alerts |
| POST | `/hotels/{id}/upload-data` | Upload bookings CSV |
