import pandas as pd


V1_FEATURE_COLUMNS = [
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
TARGET_COLUMN = "target_pm25_t_plus_6"
FORECAST_HORIZON_HOURS = 6
WEATHER_COLUMNS = {
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "surface_pressure",
}


def build_v1_features(dataframe, include_target=False):
    required = {"event_time", "pm25"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    result = dataframe.copy()
    result["event_time"] = pd.to_datetime(result["event_time"], utc=True)
    result = result.sort_values("event_time").reset_index(drop=True)

    for lag in [1, 3, 6, 12, 24]:
        result[f"pm25_lag_{lag}h"] = result["pm25"].shift(lag)

    historical_pm25 = result["pm25"].shift(1)
    for window in [6, 12, 24]:
        result[f"pm25_rolling_mean_{window}h"] = historical_pm25.rolling(
            window=window,
            min_periods=window,
        ).mean()

    local_time = result["event_time"].dt.tz_convert("Asia/Ho_Chi_Minh")
    result["hour"] = local_time.dt.hour
    result["day_of_week"] = local_time.dt.dayofweek
    result["month"] = local_time.dt.month
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)

    if include_target:
        result[TARGET_COLUMN] = result["pm25"].shift(
            -FORECAST_HORIZON_HOURS
        )

    return result
