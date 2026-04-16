"""
End-to-end hotel analytics pipeline.

Usage:
    python main.py

Output PNGs are saved to ./outputs/
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.preprocessing import load_data, clean_data, engineer_features, encode_categoricals, drop_leaky_features
from src.time_series import build_arrival_date, aggregate_daily
from src.forecasting import (
    train_xgb_forecaster, forecast_xgb,
    train_prophet_forecaster, forecast_prophet,
)
from src.cancellation import (
    TARGET_ENCODED_COLS,
    compute_scale_pos_weight,
    find_optimal_threshold,
    predict_cancellation_proba,
    predict_stacked_proba,
    train_cancellation_models_balanced,
    train_stacked_cancellation_model,
    tune_catboost_cancellation,
    tune_lgb_cancellation,
    tune_xgb_cancellation,
)
from src.preprocessing import compute_group_rates, target_encode_column
from src.analytics import estimate_staffing, estimate_profit
from src.evaluation import regression_metrics, classification_report_dict
from src.model_specs import FORECAST_LAGS
from src.visualization import (
    plot_revenue_over_time,
    plot_forecast_vs_actual,
    plot_feature_importance,
    plot_cancellation_roc,
)

DATA_PATH = "hotel_bookings.csv"
OUTPUT_DIR = "outputs"
FORECAST_PERIODS = 30
TAX_RATE = 0.10
COST_RATIO = 0.60
GUESTS_PER_STAFF = 10


def run_pipeline():
    print("=" * 60)
    print("HOTEL ANALYTICS PIPELINE")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. PREPROCESSING
    # ------------------------------------------------------------------ #
    print("\n[1/6] Loading and preprocessing data...")
    df = engineer_features(clean_data(load_data(DATA_PATH)))
    df = build_arrival_date(df)
    print(f"  {len(df):,} bookings | "
          f"ADR range: EUR {df['adr'].min():.2f}-{df['adr'].max():.2f} | "
          f"Cancellation rate: {df['is_canceled'].mean():.1%}")

    # ------------------------------------------------------------------ #
    # 2. DAILY AGGREGATION
    # ------------------------------------------------------------------ #
    print("\n[2/6] Aggregating to daily time series...")
    daily = aggregate_daily(df)
    print(f"  Date range: {daily.index.min().date()} -> {daily.index.max().date()} "
          f"({len(daily)} days)")
    plot_revenue_over_time(daily, output_dir=OUTPUT_DIR)

    # ------------------------------------------------------------------ #
    # 3. FORECASTING -- REVENUE + DEMAND
    # ------------------------------------------------------------------ #
    print("\n[3/6] Training forecasting models...")

    # Chronological 80/20 split (time-aware -- no shuffling)
    split = int(len(daily) * 0.8)

    def _ts_eval(series: pd.Series, label: str):
        train, test = series.iloc[:split], series.iloc[split:]

        # XGBoost
        xgb_m, xgb_cols = train_xgb_forecaster(train, lags=FORECAST_LAGS)
        xgb_preds = forecast_xgb(xgb_m, xgb_cols, train, periods=len(test))
        xgb_preds.index = test.index
        m = regression_metrics(test.values, xgb_preds.values)
        print(f"  {label} XGB    -- MAE: {m['mae']:>8.2f}  RMSE: {m['rmse']:>8.2f}  MAPE: {m['mape']:.1f}%")
        plot_forecast_vs_actual(test, xgb_preds, title=f"{label}_XGBoost", output_dir=OUTPUT_DIR)

        # Prophet
        prophet_m = train_prophet_forecaster(train)
        prophet_df = forecast_prophet(prophet_m, periods=len(test))
        prophet_preds = pd.Series(prophet_df["yhat"].values, index=test.index)
        prophet_lower = pd.Series(prophet_df["yhat_lower"].values, index=test.index)
        prophet_upper = pd.Series(prophet_df["yhat_upper"].values, index=test.index)
        m2 = regression_metrics(test.values, prophet_preds.values)
        print(f"  {label} Prophet -- MAE: {m2['mae']:>8.2f}  RMSE: {m2['rmse']:>8.2f}  MAPE: {m2['mape']:.1f}%")
        plot_forecast_vs_actual(test, prophet_preds, title=f"{label}_Prophet",
                                output_dir=OUTPUT_DIR, lower=prophet_lower, upper=prophet_upper)

        # Future forecast using XGBoost (trained on full series)
        full_model, full_cols = train_xgb_forecaster(series, lags=FORECAST_LAGS)
        future = forecast_xgb(full_model, full_cols, series, periods=FORECAST_PERIODS)
        return future

    rev_future = _ts_eval(daily["daily_revenue"], "Revenue")
    dem_future = _ts_eval(daily["daily_bookings"], "Demand ")

    # ------------------------------------------------------------------ #
    # 4. CANCELLATION PREDICTION
    # ------------------------------------------------------------------ #
    print("\n[4/6] Training cancellation classifiers...")
    df_sorted = df.sort_values("arrival_date")
    n_test = max(1, int(len(df_sorted) * 0.2))
    df_train_raw = df_sorted.iloc[:-n_test]

    # Fix leakage: group cancel rates from training rows only
    df_with_rates = compute_group_rates(df_train_raw, df_sorted)

    X, y_cancel, _ = encode_categoricals(drop_leaky_features(df_with_rates))

    # Target-encode high-cardinality columns using training rows only
    global_cancel_mean = float(df_train_raw["is_canceled"].mean()) if "is_canceled" in df_train_raw.columns else 0.0
    for col in TARGET_ENCODED_COLS:
        if col in X.columns and col in df_train_raw.columns:
            rate_map = df_train_raw.groupby(col)["is_canceled"].mean().to_dict()
            original_col = df_with_rates[col] if col in df_with_rates.columns else None
            if original_col is not None:
                X[col] = original_col.values
                X[col] = X[col].map(rate_map).fillna(global_cancel_mean).astype(float)

    X_tr = X.iloc[:-n_test]
    y_tr = y_cancel.iloc[:-n_test]
    X_te = X.iloc[-n_test:]
    y_te = y_cancel.iloc[-n_test:]
    feature_columns = list(X.columns)

    spw = compute_scale_pos_weight(y_tr.values)
    best_xgb_params = tune_xgb_cancellation(X_tr.values, y_tr.values, scale_pos_weight=spw)
    best_lgb_params = tune_lgb_cancellation(X_tr.values, y_tr.values, scale_pos_weight=spw)
    best_cb_params = tune_catboost_cancellation(
        X_tr.values, y_tr.values, scale_pos_weight=spw, cat_feature_indices=[]
    )

    cancel_models = train_cancellation_models_balanced(
        X_tr.values, y_tr.values,
        scale_pos_weight=spw,
        xgb_params=best_xgb_params,
        lgb_params=best_lgb_params,
        catboost_params=best_cb_params,
        feature_columns=feature_columns,
    )

    stacked_meta = train_stacked_cancellation_model(cancel_models, X_tr.values, y_tr.values)
    cancel_models["stacked_meta"] = stacked_meta

    stacked_proba = predict_stacked_proba(stacked_meta, cancel_models, X_te.values)
    optimal_threshold, _ = find_optimal_threshold(y_te.values, stacked_proba)
    cancel_models["optimal_threshold"] = optimal_threshold

    for name, model in cancel_models.items():
        if name in ("stacked_meta", "optimal_threshold"):
            continue
        y_prob = predict_cancellation_proba(model, X_te.values)
        y_pred = (y_prob >= optimal_threshold).astype(int)
        m = classification_report_dict(y_te.values, y_pred, y_prob)
        print(f"  {name:<15s}  Acc={m['accuracy']:.3f}  "
              f"Prec={m['precision']:.3f}  Rec={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  ROC-AUC={m['roc_auc']:.3f}")

    stacked_pred = (stacked_proba >= optimal_threshold).astype(int)
    from sklearn.metrics import accuracy_score, roc_auc_score
    print(f"  {'stacked':<15s}  Acc={accuracy_score(y_te.values, stacked_pred):.3f}  "
          f"ROC-AUC={roc_auc_score(y_te.values, stacked_proba):.3f}  "
          f"(threshold={optimal_threshold:.2f})")

    plot_feature_importance(cancel_models["xgboost"], feature_columns, output_dir=OUTPUT_DIR)
    plot_cancellation_roc(cancel_models, X_te.values, y_te.values, output_dir=OUTPUT_DIR)

    # ------------------------------------------------------------------ #
    # 5. STAFFING RECOMMENDATIONS
    # ------------------------------------------------------------------ #
    print("\n[5/6] Staffing recommendations (next 30 days)...")
    for i, (date, bookings) in enumerate(dem_future.items()):
        staff = estimate_staffing(bookings, guests_per_staff=GUESTS_PER_STAFF)
        if i < 5:
            print(f"  {date.date()}  predicted bookings: {bookings:>6.1f}  staff needed: {staff}")
    if len(dem_future) > 5:
        print(f"  ... ({len(dem_future) - 5} more days)")

    # ------------------------------------------------------------------ #
    # 6. PROFIT + TAX ESTIMATION
    # ------------------------------------------------------------------ #
    print("\n[6/6] Profit & tax estimation (next 30 days)...")
    fin = estimate_profit(rev_future, tax_rate=TAX_RATE, cost_ratio=COST_RATIO)
    print(f"  Forecast period  : {rev_future.index[0].date()} -> {rev_future.index[-1].date()}")
    print(f"  Projected Revenue: EUR {rev_future.sum():>12,.2f}")
    print(f"  Operational Cost : EUR {fin['operational_cost'].sum():>12,.2f}")
    print(f"  Tax ({TAX_RATE*100:.0f}%)         : EUR {fin['tax'].sum():>12,.2f}")
    print(f"  Net Profit       : EUR {fin['profit'].sum():>12,.2f}")

    print("\n" + "=" * 60)
    print(f"Complete. Outputs saved to ./{OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
