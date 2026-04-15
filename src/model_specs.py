"""
Shared model versions and forecasting defaults.
"""

FORECAST_LAGS = [1, 7, 14, 21, 30, 60, 90, 364, 365]
FORECAST_QUANTILES = (0.1, 0.5, 0.9)
MIN_FORECAST_HISTORY = 31

FORECAST_MODEL_VERSION = "forecast_v2_rich_features"
PROPHET_MODEL_VERSION = "prophet_v2_config"
INTERVAL_MODEL_VERSION = "forecast_intervals_v1"
CANCELLATION_MODEL_VERSION = "cancellation_v3_improved"
SURVIVAL_MODEL_VERSION = "cancellation_survival_v1"
EVAL_METRICS_VERSION = "evaluation_v3_leak_free"
