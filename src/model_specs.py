"""
Shared model versions and forecasting defaults.
"""

FORECAST_LAGS = [1, 7, 14, 21, 30, 60, 90, 364, 365]
FORECAST_QUANTILES = (0.1, 0.5, 0.9)
MIN_FORECAST_HISTORY = 31

# Hyperparameter tuning config for forecasting
FORECAST_TUNE_N_ITER = 20   # random param candidates to evaluate per series
FORECAST_TUNE_N_SPLITS = 3  # expanding-window time-series CV folds

FORECAST_MODEL_VERSION        = "forecast_v10_weather"
PROPHET_MODEL_VERSION         = "prophet_v3_school_holidays"
INTERVAL_MODEL_VERSION        = "forecast_intervals_v9_weather"
CANCELLATION_MODEL_VERSION    = "cancellation_v5_featureset"
SURVIVAL_MODEL_VERSION        = "cancellation_survival_v1"
EVAL_METRICS_VERSION          = "evaluation_v9_weather"
DIRECT_FORECAST_MODEL_VERSION = "forecast_direct_v2_tuned"

# Per-segment model versions (Track 1 + Track 2 + Track 3: weather features)
SEGMENTED_MODEL_VERSION    = "forecast_v10_weather"
SEGMENTED_INTERVAL_VERSION = "forecast_intervals_v9_weather"

# Ensemble weights
ENSEMBLE_WEIGHTS_VERSION   = "ensemble_v1_xgb_prophet"
