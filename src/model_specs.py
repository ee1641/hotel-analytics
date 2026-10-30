"""
Shared model versions and forecasting defaults.
"""

FORECAST_LAGS = [1, 2, 3, 4, 5, 6, 7, 14, 21, 30, 60, 90, 364, 365]
FORECAST_QUANTILES = (0.1, 0.5, 0.9)
MIN_FORECAST_HISTORY = 31

# Hyperparameter tuning config for forecasting
FORECAST_TUNE_N_ITER = 10   # random param candidates to evaluate per series
FORECAST_TUNE_N_SPLITS = 3  # expanding-window time-series CV folds

FORECAST_MODEL_VERSION        = "forecast_v17_interactions"
PROPHET_MODEL_VERSION         = "prophet_v3_school_holidays"
INTERVAL_MODEL_VERSION        = "forecast_intervals_v13_interactions"
CONFORMAL_INTERVAL_VERSION    = "conformal_v2_adaptive_bins"
CANCELLATION_MODEL_VERSION    = "cancellation_v14_expanding_te_fix"
SURVIVAL_MODEL_VERSION        = "cancellation_survival_v1"
EVAL_METRICS_VERSION          = "evaluation_v21_ensemble_direct"
DIRECT_FORECAST_MODEL_VERSION = "forecast_direct_v7_interactions"
ADR_MODEL_VERSION             = "adr_v2_ffill"

# Per-segment model versions (Track 4+5+6: velocity, log-target, quantile loss)
SEGMENTED_MODEL_VERSION    = "forecast_v17_interactions"
SEGMENTED_INTERVAL_VERSION = "forecast_intervals_v13_interactions"

# Ensemble weights
ENSEMBLE_WEIGHTS_VERSION   = "ensemble_v1_xgb_prophet"

# Stacking ensemble
STACKING_META_VERSION      = "stacking_v2_oof_weighted"
