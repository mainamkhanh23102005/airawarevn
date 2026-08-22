from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.linear_model import LinearRegression

from scripts.modeling.features import (
    FORECAST_HORIZON_HOURS,
    TARGET_COLUMN,
    V1_FEATURE_COLUMNS,
)


MODEL_TYPE = "sklearn.linear_model.LinearRegression"
DEFAULT_ARTIFACT_PATH = Path(".artifacts/models/airaware_v1.joblib")


def train_v1_model(dataframe):
    required = ["event_time", *V1_FEATURE_COLUMNS, TARGET_COLUMN]
    missing_columns = [
        column for column in required if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(f"missing required columns: {missing_columns}")
    if dataframe[required].isna().any().any():
        raise ValueError("missing required values in training dataframe")
    if not dataframe["event_time"].is_monotonic_increasing:
        raise ValueError("training dataframe must be chronologically ordered")

    model = LinearRegression()
    model.fit(dataframe[V1_FEATURE_COLUMNS], dataframe[TARGET_COLUMN])
    metadata = {
        "artifact_version": 1,
        "model_type": MODEL_TYPE,
        "feature_columns": V1_FEATURE_COLUMNS.copy(),
        "feature_configuration": "A2",
        "target_column": TARGET_COLUMN,
        "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
        "calendar_timezone": "Asia/Ho_Chi_Minh",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_start": dataframe["event_time"].min().isoformat(),
        "training_end": dataframe["event_time"].max().isoformat(),
        "training_row_count": len(dataframe),
        "raw_pm25_is_feature": False,
        "weather_is_feature": False,
    }
    return model, metadata


def save_artifact(path, model, metadata):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, destination)
    return destination
