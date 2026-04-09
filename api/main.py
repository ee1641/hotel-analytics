"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import forecasts, cancellations, overbooking, alerts, hotels, upload

app = FastAPI(
    title="Hotel Analytics API",
    description="ML-powered hotel analytics: forecasting, cancellation prediction, overbooking.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(hotels.router)
app.include_router(forecasts.router)
app.include_router(cancellations.router)
app.include_router(overbooking.router)
app.include_router(alerts.router)
app.include_router(upload.router)


@app.get("/health")
def health():
    return {"status": "ok"}
