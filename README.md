# Hotel Analytics Dashboard

An end-to-end ML pipeline and interactive Streamlit dashboard built on 119k hotel booking records (2015–2017).

## Features

- **Revenue & demand forecasting** — XGBoost (lag features), SARIMA (weekly seasonality), naive baseline
- **Cancellation prediction** — Logistic Regression, Random Forest, XGBoost (class-balanced)
- **Staffing recommendations** — derived from demand forecast
- **Profit & tax simulation** — interactive sliders on 30-day revenue projection
- **Batch pipeline** — `main.py` for full offline run with PNG outputs

## Setup

```bash
pip install -r requirements.txt
```

Place `hotel_bookings.csv` in the project root.

## Run the Dashboard

```bash
streamlit run app.py
```

**First launch** trains all models and saves them to `models/` (~5 minutes, mostly SARIMA).  
**Subsequent launches** load from disk and start in ~2 seconds.

## Re-train Models

Delete one or all saved artifacts and relaunch:

```bash
# Re-train everything
rm models/*.pkl
streamlit run app.py

# Re-train only SARIMA
rm models/sarima_revenue_result.pkl models/sarima_demand_result.pkl
streamlit run app.py
```

## Run Batch Pipeline (no UI)

```bash
python main.py
```

Output PNGs are saved to `outputs/`.

## Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
app.py                  # Streamlit root — cached data + model loaders
pages/
  1_Overview.py         # KPIs, revenue/bookings charts, filters
  2_Revenue_Forecast.py # Forecast comparison + cross-validation
  3_Demand_Staffing.py  # Demand forecast + staffing plan
  4_Cancellation_Predictor.py  # Interactive cancellation risk form
  5_Profit_Simulator.py # Tax/cost sliders + profit breakdown
src/
  preprocessing.py      # load, clean, engineer, encode + temporal features
  time_series.py        # arrival_date construction, daily aggregation
  forecasting.py        # XGBoost, SARIMA, naive baseline
  cancellation.py       # 3 classifiers + class-balanced variants
  analytics.py          # staffing and profit estimation
  evaluation.py         # regression metrics, classification metrics, TS CV
  visualization.py      # Matplotlib PNG outputs (used by main.py)
  model_cache.py        # joblib save/load helpers
models/                 # Persisted model artifacts (auto-created)
outputs/                # PNG files from main.py
tests/                  # 45 unit tests
```

## ML Notes

- **SARIMA**: `order=(1,1,1)`, `seasonal_order=(1,1,1,7)` — captures weekly seasonality only (annual period is computationally intractable at daily frequency)
- **Class imbalance**: Cancellation rate ~37% — handled via `class_weight='balanced'` (LogReg/RF) and `scale_pos_weight` (XGBoost)
- **New features**: `booking_month`, `season` (spring/summer/fall/winter), `is_weekend_arrival`
- **Leakage prevention**: `reservation_status` and `reservation_status_date` excluded from all classification features
