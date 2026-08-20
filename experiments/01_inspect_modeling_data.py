import json
from pathlib import Path

import pandas as pd


PM25_PATH = Path(
    ".artifacts/data_spike/coverage/"
    "openaq_normalized_sensor_13502151_20250731T170000+0000.json"
)

WEATHER_PATH = Path(
    ".artifacts/data_spike/weather/openmeteo_normalized.json"
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


# =========================================================
# 1. LOAD DATA
# =========================================================

pm25_data = load_json(PM25_PATH)
weather_data = load_json(WEATHER_PATH)


# =========================================================
# 2. PM2.5 -> DATAFRAME
# =========================================================

pm25_records = pm25_data["normalized_records"]
pm25_df = pd.DataFrame(pm25_records)

pm25_df["event_time"] = pd.to_datetime(
    pm25_df["event_time"],
    utc=True,
)

pm25_df["period_end_utc"] = pd.to_datetime(
    pm25_df["period_end_utc"],
    utc=True,
)


# =========================================================
# 3. WEATHER -> DATAFRAME
# =========================================================

weather_records = weather_data["records"]
weather_df = pd.DataFrame(weather_records)

weather_df["event_time"] = pd.to_datetime(
    weather_df["event_time"],
    utc=True,
)


# =========================================================
# 4. READ THE FROZEN MODELING WINDOW
# =========================================================

start = pd.to_datetime(
    pm25_data["frozen_candidate"]["start_utc"],
    utc=True,
)

end = pd.to_datetime(
    pm25_data["frozen_candidate"]["end_utc"],
    utc=True,
)


# =========================================================
# 5. FILTER BOTH DATASETS TO THE SAME ONE-YEAR WINDOW
# =========================================================

pm25_frozen = pm25_df[
    (pm25_df["event_time"] >= start)
    & (pm25_df["event_time"] < end)
].copy()

weather_frozen = weather_df[
    (weather_df["event_time"] >= start)
    & (weather_df["event_time"] < end)
].copy()

# ============================================================
# BUILD COMPLETE HOURLY MODELING GRID
# ============================================================

hourly_grid = pd.DataFrame(
    {
        "event_time": pd.date_range(
            start=start,
            end=end,
            freq="1h",
            inclusive="left",
        )
    }
)

print("\n=== COMPLETE HOURLY GRID ===")
print(f"Rows: {len(hourly_grid)}")
print(f"Min time: {hourly_grid['event_time'].min()}")
print(f"Max time: {hourly_grid['event_time'].max()}")


# ============================================================
# JOIN PM2.5
# ============================================================

modeling_df = hourly_grid.merge(
    pm25_frozen[["event_time", "value"]],
    on="event_time",
    how="left",
)

modeling_df = modeling_df.rename(
    columns={"value": "pm25"}
)


# ============================================================
# JOIN WEATHER
# ============================================================

modeling_df = modeling_df.merge(
    weather_frozen,
    on="event_time",
    how="left",
)

# ============================================================
# CREATE 6-HOUR-AHEAD FORECAST TARGET
# ============================================================

modeling_df["target_pm25_t_plus_6"] = modeling_df["pm25"].shift(-6)

print("\n=== 6-HOUR FORECAST TARGET ===")

print(
    "Target non-null:",
    modeling_df["target_pm25_t_plus_6"].notna().sum(),
)

print(
    "Target null:",
    modeling_df["target_pm25_t_plus_6"].isna().sum(),
)

print(
    modeling_df[
        [
            "event_time",
            "pm25",
            "target_pm25_t_plus_6",
        ]
    ].head(10)
)

# ============================================================
# CREATE HISTORICAL PM2.5 LAG FEATURES
# ============================================================
lag_hours = [1, 3, 6, 12, 24]

for lag in lag_hours:
    modeling_df[f"pm25_lag_{lag}h"] = modeling_df["pm25"].shift(lag)

print("\n=== PM2.5 LAG FEATURES ===")

lag_columns = [
    "pm25_lag_1h",
    "pm25_lag_3h",
    "pm25_lag_6h",
    "pm25_lag_12h",
    "pm25_lag_24h",
]

print(
    modeling_df[
        [
            "event_time",
            "pm25",
            *lag_columns,
            "target_pm25_t_plus_6"
        ]
    ].head(30)
)

print("\nLag null counts:")

for column in lag_columns:
    print(
        f"{column}: "
        f"{modeling_df[column].isna().sum()}"
    )

# ============================================================
# VERIFY MODELING TABLE
# ============================================================

print("\n=== MODELING TABLE ===")

print(f"Rows: {len(modeling_df)}")

print(
    f"PM2.5 non-null: "
    f"{modeling_df['pm25'].notna().sum()}"
)

print(
    f"PM2.5 null: "
    f"{modeling_df['pm25'].isna().sum()}"
)

weather_columns = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "surface_pressure",
]

print("\nWeather null counts:")

for column in weather_columns:
    print(
        f"{column}: "
        f"{modeling_df[column].isna().sum()}"
    )


print("\nFirst 5 rows:")
print(modeling_df.head())

print("\nLast 5 rows:")
print(modeling_df.tail())

# ============================================================
# CREATE HISTORICAL PM2.5 ROLLING FEATURES
# ============================================================

# Shift first so the rolling features use only hours before t.
historical_pm25 = modeling_df["pm25"].shift(1)

rolling_windows = [6, 12, 24]

for window in rolling_windows:
    modeling_df[f"pm25_rolling_mean_{window}h"] = (
        historical_pm25
        .rolling(
            window=window,
            min_periods=window,
        )
        .mean()
    )


rolling_columns = [
    "pm25_rolling_mean_6h",
    "pm25_rolling_mean_12h",
    "pm25_rolling_mean_24h",
]

print("\n=== PM2.5 ROLLING FEATURES ===")

print(
    modeling_df[
        [
            "event_time",
            "pm25",
            *rolling_columns,
            "target_pm25_t_plus_6",
        ]
    ].head(30)
)

print("\nRolling feature null counts:")

for column in rolling_columns:
    print(
        f"{column}: "
        f"{modeling_df[column].isna().sum()}"
    )

# ============================================================
# CREATE TIME / CALENDAR FEATURES
# ============================================================

local_time = modeling_df["event_time"].dt.tz_convert("Asia/Ho_Chi_Minh")

modeling_df["hour"] = local_time.dt.hour
modeling_df["day_of_week"] = local_time.dt.dayofweek
modeling_df["month"] = local_time.dt.month
modeling_df["is_weekend"] = (
    modeling_df["day_of_week"] >= 5
).astype(int)


print("\n=== TIME FEATURES ===")

print(
    modeling_df[
        [
            "event_time",
            "hour",
            "day_of_week",
            "month",
            "is_weekend",
        ]
    ].head(15)
)

# =========================================================
# 6. INSPECT ORIGINAL DATA
# =========================================================

print("\n=== ORIGINAL DATA ===")

print("\nPM2.5")
print("Rows:", len(pm25_df))
print("Columns:", list(pm25_df.columns))
print("Min time:", pm25_df["event_time"].min())
print("Max time:", pm25_df["event_time"].max())

print("\nWeather")
print("Rows:", len(weather_df))
print("Columns:", list(weather_df.columns))
print("Min time:", weather_df["event_time"].min())
print("Max time:", weather_df["event_time"].max())


# =========================================================
# 7. INSPECT FROZEN MODELING WINDOW
# =========================================================

print("\n=== FROZEN MODELING WINDOW ===")

print("Start:", start)
print("End:", end)

print("\nPM2.5 frozen")
print("Rows:", len(pm25_frozen))
print("Min time:", pm25_frozen["event_time"].min())
print("Max time:", pm25_frozen["event_time"].max())

print("\nWeather frozen")
print("Rows:", len(weather_frozen))
print("Min time:", weather_frozen["event_time"].min())
print("Max time:", weather_frozen["event_time"].max())


# =========================================================
# 8. BASIC DATA QUALITY CHECKS
# =========================================================

expected_hours = int(
    (end - start).total_seconds() / 3600
)

missing_pm25_hours = expected_hours - len(pm25_frozen)
missing_weather_hours = expected_hours - len(weather_frozen)

print("\n=== BASIC QUALITY CHECK ===")

print("Expected hours:", expected_hours)

print(
    "PM2.5 observed hours:",
    len(pm25_frozen),
)

print(
    "PM2.5 missing hours:",
    missing_pm25_hours,
)

print(
    "Weather observed hours:",
    len(weather_frozen),
)

print(
    "Weather missing hours:",
    missing_weather_hours,
)


# =========================================================
# 9. SHOW SAMPLE ROWS
# =========================================================

print("\n=== PM2.5 SAMPLE ===")
print(
    pm25_frozen[
        [
            "event_time",
            "value",
            "unit",
        ]
    ].head()
)

print("\n=== WEATHER SAMPLE ===")
print(
    weather_frozen[
        [
            "event_time",
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "precipitation",
            "surface_pressure",
        ]
    ].head()
)

# ============================================================
# POINT-IN-TIME AVAILABILITY INSPECTION
# ============================================================

print("\n=== PM2.5 TIMESTAMP AVAILABILITY INSPECTION ===")

print(
    pm25_frozen[
        [
            "event_time",
            "period_end_utc",
            "record_id",
            "sensor_id",
            "value",
        ]
    ].head(10)
)

print("\nTimestamp difference:")

timestamp_difference = (
    pm25_frozen["period_end_utc"]
    - pm25_frozen["event_time"]
)

print(timestamp_difference.describe())
print(timestamp_difference.value_counts().head(10))


print(
    "\nDone. Data is now loaded, converted to UTC datetimes, "
    "and restricted to the frozen one-year modeling window."
)