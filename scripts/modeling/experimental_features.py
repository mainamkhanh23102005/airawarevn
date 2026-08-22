import numpy as np
import pandas as pd

from scripts.modeling.features import TARGET_COLUMN, build_v1_features


SHORT_LAGS = (2, 4, 8, 18)
LONG_LAGS = (48, 72, 168)
SHORT_LAG_COLUMNS = [f"pm25_lag_{lag}h" for lag in SHORT_LAGS]
LONG_LAG_COLUMNS = [f"pm25_lag_{lag}h" for lag in LONG_LAGS]
DYNAMICS_COLUMNS = [
    "pm25_diff_1h",
    "pm25_diff_3h",
    "pm25_trend_6h",
    "pm25_rolling_std_6h",
    "pm25_rolling_std_12h",
    "pm25_rolling_std_24h",
    "pm25_mean_gap_6h_24h",
]
CYCLICAL_CALENDAR_COLUMNS = ["hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "annual_sin", "annual_cos"]
FROZEN_A0_FEATURE_COLUMNS = [
    "pm25_lag_1h",
    "pm25_lag_3h",
    "pm25_lag_6h",
    "pm25_lag_12h",
    "pm25_lag_24h",
    "pm25_rolling_mean_6h",
    "pm25_rolling_mean_12h",
    "pm25_rolling_mean_24h",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
]
ABLATION_FEATURE_COLUMNS = {
    "A0": FROZEN_A0_FEATURE_COLUMNS.copy(),
    "A1": [*FROZEN_A0_FEATURE_COLUMNS, *DYNAMICS_COLUMNS],
    "A2": [*FROZEN_A0_FEATURE_COLUMNS, *SHORT_LAG_COLUMNS],
    "A3": [*FROZEN_A0_FEATURE_COLUMNS, *LONG_LAG_COLUMNS],
    "A4": [*FROZEN_A0_FEATURE_COLUMNS, *CYCLICAL_CALENDAR_COLUMNS],
    "A5": [*FROZEN_A0_FEATURE_COLUMNS, *DYNAMICS_COLUMNS, *SHORT_LAG_COLUMNS],
}


def build_experimental_features(dataframe, include_target=False):
    result = build_v1_features(dataframe, include_target=include_target)
    historical = result["pm25"].shift(1)
    for lag in (*SHORT_LAGS, *LONG_LAGS):
        result[f"pm25_lag_{lag}h"] = result["pm25"].shift(lag)
    result["pm25_diff_1h"] = historical - historical.shift(1)
    result["pm25_diff_3h"] = historical - historical.shift(3)
    result["pm25_trend_6h"] = historical - historical.shift(5)
    for window in (6, 12, 24):
        result[f"pm25_rolling_std_{window}h"] = historical.rolling(window, min_periods=window).std()
    result["pm25_mean_gap_6h_24h"] = historical.rolling(6, min_periods=6).mean() - historical.rolling(24, min_periods=24).mean()
    local = result["event_time"].dt.tz_convert("Asia/Ho_Chi_Minh")
    result["hour_sin"] = np.sin(2 * np.pi * local.dt.hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * local.dt.hour / 24)
    result["day_of_week_sin"] = np.sin(2 * np.pi * local.dt.dayofweek / 7)
    result["day_of_week_cos"] = np.cos(2 * np.pi * local.dt.dayofweek / 7)
    year_length = np.where(local.dt.is_leap_year, 366, 365)
    result["annual_sin"] = np.sin(2 * np.pi * (local.dt.dayofyear - 1) / year_length)
    result["annual_cos"] = np.cos(2 * np.pi * (local.dt.dayofyear - 1) / year_length)
    return result


def build_ablation_dataframe(raw, feature_columns):
    featured = build_experimental_features(raw, include_target=True)
    return featured.dropna(subset=[*feature_columns, TARGET_COLUMN]).reset_index(drop=True)
