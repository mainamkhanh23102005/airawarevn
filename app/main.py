import json
import math
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from scripts.modeling.features import (
    FORECAST_HORIZON_HOURS,
    V1_FEATURE_COLUMNS,
    build_v1_features,
)
from scripts.modeling.predict import load_artifact, predict_pm25_t_plus_6
from scripts.modeling.train import MODEL_TYPE
from scripts.openmeteo import OpenMeteoError, TARGET_SENSOR_ID, load_pm25_artifact
from scripts.run_data_spike import resolve_hour_duplicates

MODEL_VERSION = "v1"
CALENDAR_TIMEZONE = "Asia/Ho_Chi_Minh"
MINIMUM_HISTORY_HOURS = 24
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPOSITORY_ROOT / ".artifacts" / "models" / "airaware_v1.joblib"
DEFAULT_PM25_ARTIFACT_PATH = (
    REPOSITORY_ROOT
    / ".artifacts"
    / "data_spike"
    / "coverage"
    / "openaq_normalized_sensor_13502151_20250731T170000+0000.json"
)
DEFAULT_CURRENT_PM25_ARTIFACT_PATH = REPOSITORY_ROOT / ".artifacts" / "live" / "current_pm25.json"
FRESHNESS_THRESHOLD_HOURS = 4
INDEX_PATH = Path(__file__).with_name("index.html")


class Pm25Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_time: datetime
    pm25: Annotated[float, Field(allow_inf_nan=False)]

    @field_validator("event_time")
    @classmethod
    def require_aware_event_time(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("event_time must include a timezone")
        return value


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_time: datetime
    history: list[Pm25Observation]

    @field_validator("prediction_time")
    @classmethod
    def require_aware_prediction_time(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("prediction_time must include a timezone")
        if value.minute or value.second or value.microsecond:
            raise ValueError("prediction_time must align to an hour boundary")
        return value


class HealthResponse(BaseModel):
    status: str
    model: str
    model_version: str
    forecast_horizon_hours: int


class PredictionResponse(BaseModel):
    prediction_time: datetime
    target_interval_start: datetime
    target_interval_end: datetime
    forecast_horizon_hours: int
    predicted_pm25: float
    unit: str
    model_version: str


class LatestForecastResponse(PredictionResponse):
    latest_completed_pm25: float
    history_start: datetime
    history_end: datetime
    data_mode: str


class CurrentForecastResponse(LatestForecastResponse):
    source_retrieved_at: datetime
    sensor_id: int
    freshness_status: str
    age_minutes: float


def _validate_metadata(metadata):
    expected = {
        "artifact_version": 1,
        "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
        "calendar_timezone": CALENDAR_TIMEZONE,
        "raw_pm25_is_feature": False,
        "weather_is_feature": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"artifact metadata {key!r} does not match V1 contract"
            )


def _build_prediction_features(payload):
    prediction_time = pd.Timestamp(payload.prediction_time).tz_convert("UTC")
    if len(payload.history) != MINIMUM_HISTORY_HOURS:
        raise ValueError(
            f"history must contain exactly {MINIMUM_HISTORY_HOURS} observations"
        )

    history = pd.DataFrame(
        {
            "event_time": [entry.event_time for entry in payload.history],
            "pm25": [entry.pm25 for entry in payload.history],
        }
    )
    history["event_time"] = pd.to_datetime(history["event_time"], utc=True)
    if history["event_time"].duplicated().any():
        raise ValueError("history contains duplicate timestamps")
    history = history.sort_values("event_time").reset_index(drop=True)

    expected_times = pd.date_range(
        end=prediction_time - pd.Timedelta(hours=1),
        periods=MINIMUM_HISTORY_HOURS,
        freq="1h",
    )
    if not history["event_time"].equals(pd.Series(expected_times)):
        raise ValueError(
            "history must cover each completed hourly interval from "
            "prediction_time - 24 hours through prediction_time - 1 hour"
        )

    source = pd.concat(
        [
            history,
            pd.DataFrame(
                {"event_time": [prediction_time], "pm25": [float("nan")]}
            ),
        ],
        ignore_index=True,
    )
    features = build_v1_features(source).iloc[[-1]]
    if not all(math.isfinite(value) for value in features[V1_FEATURE_COLUMNS].iloc[0]):
        raise ValueError("feature builder produced non-finite V1 features")
    return features


def _latest_prediction_request(artifact):
    grouped = {}
    for record in artifact.records:
        event_time = record["event_time"]
        try:
            period_end = datetime.fromisoformat(
                record["period_end_utc"].replace("Z", "+00:00")
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if event_time.minute or event_time.second or event_time.microsecond:
            continue
        if period_end != event_time + timedelta(hours=1):
            continue
        if artifact.start <= event_time < artifact.end:
            grouped.setdefault(event_time, []).append(record)

    values = {}
    for event_time, records in grouped.items():
        resolution = resolve_hour_duplicates(records)
        value = resolution.selected_value_ug_m3
        if resolution.has_valid_pm25 and value is not None and math.isfinite(value):
            values[event_time] = float(value)

    prediction_time = None
    run_length = 0
    previous = None
    for event_time in sorted(values):
        if previous is not None and event_time == previous + timedelta(hours=1):
            run_length += 1
        else:
            run_length = 1
        if run_length >= MINIMUM_HISTORY_HOURS:
            prediction_time = event_time + timedelta(hours=1)
        previous = event_time

    if prediction_time is None:
        raise ValueError(
            "normalized PM2.5 artifact has no contiguous 24-hour prediction window"
        )

    history = [
        Pm25Observation(event_time=event_time, pm25=values[event_time])
        for event_time in (
            prediction_time - timedelta(hours=offset)
            for offset in range(MINIMUM_HISTORY_HOURS, 0, -1)
        )
    ]
    return PredictionRequest(prediction_time=prediction_time, history=history)


def _current_prediction_request(artifact, now):
    now = now.astimezone(timezone.utc)
    grouped = {}
    for record in artifact.get("normalized_records", []):
        try:
            event_time = datetime.fromisoformat(record["event_time"].replace("Z", "+00:00")).astimezone(timezone.utc)
            period_end = datetime.fromisoformat(record["period_end_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if event_time.minute or event_time.second or event_time.microsecond:
            continue
        if period_end != event_time + timedelta(hours=1) or period_end > now:
            continue
        grouped.setdefault(event_time, []).append(record)

    values = {}
    for event_time, records in grouped.items():
        resolution = resolve_hour_duplicates(records)
        if resolution.has_valid_pm25 and resolution.selected_value_ug_m3 is not None:
            values[event_time] = float(resolution.selected_value_ug_m3)

    eligible_ends = [event_time + timedelta(hours=1) for event_time in values if event_time + timedelta(hours=1) <= now]
    if not eligible_ends:
        raise ValueError("fresh PM2.5 artifact has no completed hourly intervals")
    prediction_time = max(eligible_ends)
    required = [prediction_time - timedelta(hours=offset) for offset in range(MINIMUM_HISTORY_HOURS, 0, -1)]
    if any(event_time not in values for event_time in required):
        raise ValueError("fresh PM2.5 artifact lacks contiguous 24 completed hourly intervals ending at prediction time")
    history = [Pm25Observation(event_time=event_time, pm25=values[event_time]) for event_time in required]
    return PredictionRequest(prediction_time=prediction_time, history=history)


def _load_current_artifact(path):
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise OpenMeteoError(f"fresh PM2.5 artifact not found: {path}") from error
    except (OSError, ValueError, TypeError) as error:
        raise OpenMeteoError(f"fresh PM2.5 artifact is unreadable: {path}") from error
    if artifact.get("artifact_version") != 1 or artifact.get("sensor_id") != TARGET_SENSOR_ID:
        raise OpenMeteoError("fresh PM2.5 artifact metadata is incompatible")
    try:
        retrieved_at = datetime.fromisoformat(artifact["retrieved_at"].replace("Z", "+00:00"))
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise OpenMeteoError("fresh PM2.5 artifact lacks valid retrieval timestamp") from error
    return artifact, retrieved_at


def _predict(model, metadata, payload):
    features = _build_prediction_features(payload)
    result = predict_pm25_t_plus_6(model, metadata, features)
    predicted_pm25 = float(result[0])
    if not math.isfinite(predicted_pm25):
        raise ValueError("model produced a non-finite prediction")
    return predicted_pm25


def create_app(model_path=None, pm25_artifact_path=None, current_pm25_artifact_path=None, now=lambda: datetime.now(timezone.utc)):
    configured_path = Path(
        model_path or os.environ.get("AIRAWARE_MODEL_PATH", DEFAULT_MODEL_PATH)
    )
    configured_pm25_path = Path(
        pm25_artifact_path
        or os.environ.get("AIRAWARE_PM25_ARTIFACT_PATH", DEFAULT_PM25_ARTIFACT_PATH)
    )
    configured_current_pm25_path = Path(
        current_pm25_artifact_path
        or os.environ.get("AIRAWARE_CURRENT_PM25_ARTIFACT_PATH", DEFAULT_CURRENT_PM25_ARTIFACT_PATH)
    )

    @asynccontextmanager
    async def lifespan(application):
        model, metadata = load_artifact(configured_path)
        _validate_metadata(metadata)
        application.state.model = model
        application.state.metadata = metadata
        application.state.pm25_artifact_path = configured_pm25_path
        application.state.current_pm25_artifact_path = configured_current_pm25_path
        yield

    application = FastAPI(title="AirAware VN", version=MODEL_VERSION, lifespan=lifespan)

    @application.get("/", include_in_schema=False)
    def index():
        return FileResponse(INDEX_PATH, media_type="text/html")

    @application.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(
            status="ok",
            model=MODEL_TYPE.rsplit(".", 1)[-1],
            model_version=MODEL_VERSION,
            forecast_horizon_hours=FORECAST_HORIZON_HOURS,
        )

    @application.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request):
        try:
            predicted_pm25 = _predict(
                request.app.state.model,
                request.app.state.metadata,
                payload,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        target_start = payload.prediction_time + timedelta(
            hours=FORECAST_HORIZON_HOURS
        )
        return PredictionResponse(
            prediction_time=payload.prediction_time,
            target_interval_start=target_start,
            target_interval_end=target_start + timedelta(hours=1),
            forecast_horizon_hours=FORECAST_HORIZON_HOURS,
            predicted_pm25=predicted_pm25,
            unit="µg/m³",
            model_version=MODEL_VERSION,
        )

    @application.get("/forecast/latest", response_model=LatestForecastResponse)
    def latest_forecast(request: Request):
        try:
            artifact = load_pm25_artifact(
                request.app.state.pm25_artifact_path,
                TARGET_SENSOR_ID,
            )
            payload = _latest_prediction_request(artifact)
            predicted_pm25 = _predict(
                request.app.state.model,
                request.app.state.metadata,
                payload,
            )
        except OpenMeteoError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        target_start = payload.prediction_time + timedelta(
            hours=FORECAST_HORIZON_HOURS
        )
        return LatestForecastResponse(
            prediction_time=payload.prediction_time,
            target_interval_start=target_start,
            target_interval_end=target_start + timedelta(hours=1),
            forecast_horizon_hours=FORECAST_HORIZON_HOURS,
            predicted_pm25=predicted_pm25,
            unit="µg/m³",
            model_version=MODEL_VERSION,
            latest_completed_pm25=payload.history[-1].pm25,
            history_start=payload.history[0].event_time,
            history_end=payload.history[-1].event_time,
            data_mode="historical_artifact",
        )

    @application.get("/forecast/current", response_model=CurrentForecastResponse)
    def current_forecast(request: Request):
        current_time = now().astimezone(timezone.utc)
        try:
            artifact, retrieved_at = _load_current_artifact(request.app.state.current_pm25_artifact_path)
            payload = _current_prediction_request(artifact, current_time)
            predicted_pm25 = _predict(request.app.state.model, request.app.state.metadata, payload)
        except OpenMeteoError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        age_minutes = max(0.0, (current_time - payload.prediction_time).total_seconds() / 60)
        is_stale = age_minutes > FRESHNESS_THRESHOLD_HOURS * 60
        target_start = payload.prediction_time + timedelta(hours=FORECAST_HORIZON_HOURS)
        return CurrentForecastResponse(
            prediction_time=payload.prediction_time,
            target_interval_start=target_start,
            target_interval_end=target_start + timedelta(hours=1),
            forecast_horizon_hours=FORECAST_HORIZON_HOURS,
            predicted_pm25=predicted_pm25,
            unit="µg/m³",
            model_version=MODEL_VERSION,
            latest_completed_pm25=payload.history[-1].pm25,
            history_start=payload.history[0].event_time,
            history_end=payload.history[-1].event_time,
            data_mode="stale_openaq" if is_stale else "fresh_openaq",
            source_retrieved_at=retrieved_at,
            sensor_id=artifact["sensor_id"],
            freshness_status="stale" if is_stale else "fresh",
            age_minutes=age_minutes,
        )

    return application


app = create_app()
