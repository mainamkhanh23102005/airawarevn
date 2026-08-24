import hashlib
import json
import math
import sqlite3
import unicodedata
import uuid
from contextlib import closing
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

from scripts.modeling.features import FORECAST_HORIZON_HOURS, TARGET_COLUMN, V1_FEATURE_COLUMNS


FORECAST_NAMESPACE = uuid.UUID("e60f879b-90cf-51a4-8936-aad3c962371c")
SCHEMA_VERSION = 1
IDENTITY_FIELDS = (
    "sensor_id", "prediction_time", "target_interval_start", "target_interval_end",
    "forecast_horizon_hours", "model_version", "model_artifact_sha256", "feature_schema_sha256",
)


class ForecastIntegrityError(Exception):
    pass


@dataclass(frozen=True)
class InsertResult:
    status: str
    record: "ForecastRecord"


@dataclass(frozen=True)
class IssueResult:
    outcome: str
    record: "ForecastRecord | None" = None


def _utc(value, name, hour_aligned=False):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    value = value.astimezone(timezone.utc)
    if value.microsecond:
        raise ValueError(f"{name} must not contain fractional seconds")
    if hour_aligned and (value.minute or value.second):
        raise ValueError(f"{name} must align to an hour")
    return value


def _timestamp(value):
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_timestamp(value):
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _hash(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be lowercase SHA256")
    return value


@dataclass(frozen=True)
class ForecastRecord:
    sensor_id: int
    prediction_time: datetime
    target_interval_start: datetime
    target_interval_end: datetime
    forecast_horizon_hours: int
    model_version: str
    model_artifact_sha256: str
    feature_schema_sha256: str
    forecast_id: str
    predicted_pm25: float
    persistence_prediction: float
    unit: str
    artifact_version: int
    feature_configuration: str
    source_retrieved_at: datetime
    input_data_cutoff: datetime
    history_start: datetime
    history_end: datetime
    data_mode_at_issue: str
    freshness_status_at_issue: str
    source_age_minutes_at_issue: float
    issuance_mode: str
    issued_at: datetime

    @classmethod
    def create(cls, sensor_id, prediction_time, predicted_pm25, persistence_prediction, model_version,
               model_artifact_sha256, feature_schema_sha256, artifact_version, feature_configuration,
               source_retrieved_at, input_data_cutoff, history_start, history_end, data_mode_at_issue,
               freshness_status_at_issue, source_age_minutes_at_issue, issued_at, forecast_id=None,
               target_interval_start=None, target_interval_end=None, forecast_horizon_hours=6,
               unit="µg/m³", issuance_mode="scheduled"):
        if type(sensor_id) is not int or sensor_id <= 0:
            raise ValueError("sensor_id must be a positive integer")
        for name, value in {"model_version": model_version, "unit": unit, "feature_configuration": feature_configuration, "data_mode_at_issue": data_mode_at_issue, "freshness_status_at_issue": freshness_status_at_issue}.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be nonempty")
        prediction_time = _utc(prediction_time, "prediction_time", True)
        target_interval_start = _utc(target_interval_start or prediction_time + timedelta(hours=6), "target_interval_start", True)
        target_interval_end = _utc(target_interval_end or target_interval_start + timedelta(hours=1), "target_interval_end", True)
        values = {
            "source_retrieved_at": source_retrieved_at, "input_data_cutoff": input_data_cutoff,
            "history_start": history_start, "history_end": history_end, "issued_at": issued_at,
        }
        values = {key: _utc(value, key) for key, value in values.items()}
        if forecast_horizon_hours != 6 or target_interval_start != prediction_time + timedelta(hours=6) or target_interval_end != target_interval_start + timedelta(hours=1):
            raise ValueError("invalid target interval")
        if values["input_data_cutoff"] != prediction_time or values["history_end"] != prediction_time - timedelta(hours=1) or values["history_start"] != prediction_time - timedelta(hours=24):
            raise ValueError("invalid input history relationship")
        if issuance_mode != "scheduled":
            raise ValueError("issuance_mode must be scheduled")
        if not math.isfinite(predicted_pm25) or not math.isfinite(persistence_prediction):
            raise ValueError("predictions must be finite")
        if not math.isfinite(source_age_minutes_at_issue) or source_age_minutes_at_issue < 0:
            raise ValueError("source age must be nonnegative")
        model_version = unicodedata.normalize("NFC", model_version)
        feature_configuration = unicodedata.normalize("NFC", feature_configuration)
        record = cls(sensor_id, prediction_time, target_interval_start, target_interval_end, forecast_horizon_hours,
                     model_version, _hash(model_artifact_sha256, "model_artifact_sha256"),
                     _hash(feature_schema_sha256, "feature_schema_sha256"), forecast_id or "",
                     float(predicted_pm25), float(persistence_prediction), unicodedata.normalize("NFC", unit),
                     artifact_version, feature_configuration, values["source_retrieved_at"], values["input_data_cutoff"],
                     values["history_start"], values["history_end"], unicodedata.normalize("NFC", data_mode_at_issue),
                     unicodedata.normalize("NFC", freshness_status_at_issue), float(source_age_minutes_at_issue),
                     issuance_mode, values["issued_at"])
        expected_id = str(uuid.uuid5(FORECAST_NAMESPACE, canonical_identity_json(record)))
        if forecast_id is not None and forecast_id != expected_id:
            raise ForecastIntegrityError("forecast_id does not match natural identity")
        return cls(**{**asdict(record), "forecast_id": forecast_id or expected_id})

    def as_dict(self):
        return asdict(self)


def canonical_identity_json(record):
    payload = {"identity_serialization_version": 1}
    for name in IDENTITY_FIELDS:
        value = getattr(record, name)
        if isinstance(value, datetime):
            value = _timestamp(value)
        elif isinstance(value, str):
            value = unicodedata.normalize("NFC", value)
        payload[name] = value
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_feature_schema_json():
    return json.dumps({
        "calendar_timezone": "Asia/Ho_Chi_Minh", "feature_columns": V1_FEATURE_COLUMNS,
        "feature_configuration": "A2", "feature_schema_serialization_version": 1,
        "forecast_horizon_hours": FORECAST_HORIZON_HOURS, "raw_pm25_is_feature": False,
        "target_column": TARGET_COLUMN, "weather_is_feature": False,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def feature_schema_sha256():
    return hashlib.sha256(canonical_feature_schema_json().encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ForecastStore(Protocol):
    def initialize(self): ...
    def insert(self, record): ...
    def latest(self): ...


class SQLiteForecastStore:
    def __init__(self, path):
        self.path = Path(path)

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        declarations = []
        for field in fields(ForecastRecord):
            kind = "INTEGER" if field.name in {"sensor_id", "forecast_horizon_hours", "artifact_version"} else "REAL" if field.name in {"predicted_pm25", "persistence_prediction", "source_age_minutes_at_issue"} else "TEXT"
            declarations.append(f"{field.name} {kind} NOT NULL" + (" PRIMARY KEY" if field.name == "forecast_id" else ""))
        unique = ", UNIQUE (" + ", ".join(IDENTITY_FIELDS) + ")"
        with closing(self._connect()) as connection, connection:
            connection.execute("CREATE TABLE IF NOT EXISTS forecasts (" + ", ".join(declarations) + unique + ")")
            connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def _after_insert(self, connection, record):
        pass

    def _values(self, record):
        return tuple(_timestamp(value) if isinstance(value, datetime) else value for value in asdict(record).values())

    def insert(self, record):
        expected_id = str(uuid.uuid5(FORECAST_NAMESPACE, canonical_identity_json(record)))
        if record.forecast_id != expected_id:
            raise ForecastIntegrityError("forecast_id does not match natural identity")
        names = tuple(field.name for field in fields(ForecastRecord))
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = self._get_by_identity_connection(connection, record)
            if existing:
                if existing != record:
                    raise ForecastIntegrityError("immutable forecast conflict")
                return InsertResult("already_exists", existing)
            try:
                connection.execute(f"INSERT INTO forecasts ({', '.join(names)}) VALUES ({', '.join('?' for _ in names)})", self._values(record))
            except sqlite3.IntegrityError as error:
                raise ForecastIntegrityError("forecast identity conflict") from error
            self._after_insert(connection, record)
        return InsertResult("inserted", record)

    def _row(self, row):
        if row is None:
            return None
        payload = dict(row)
        for name in ("prediction_time", "target_interval_start", "target_interval_end", "source_retrieved_at", "input_data_cutoff", "history_start", "history_end", "issued_at"):
            payload[name] = _parse_timestamp(payload[name])
        return ForecastRecord(**payload)

    def _get_by_identity_connection(self, connection, record):
        where = " AND ".join(f"{name}=?" for name in IDENTITY_FIELDS)
        values = tuple(_timestamp(getattr(record, name)) if isinstance(getattr(record, name), datetime) else getattr(record, name) for name in IDENTITY_FIELDS)
        return self._row(connection.execute(f"SELECT * FROM forecasts WHERE {where}", values).fetchone())

    def get_by_id(self, forecast_id):
        with closing(self._connect()) as connection:
            return self._row(connection.execute("SELECT * FROM forecasts WHERE forecast_id=?", (forecast_id,)).fetchone())

    def get_by_identity(self, record):
        with closing(self._connect()) as connection:
            return self._get_by_identity_connection(connection, record)

    def latest(self):
        with closing(self._connect()) as connection:
            return self._row(connection.execute("SELECT * FROM forecasts ORDER BY issued_at DESC, rowid DESC LIMIT 1").fetchone())

    def count(self):
        with closing(self._connect()) as connection:
            return connection.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]


def _record_json(record):
    payload = asdict(record)
    for field in fields(record):
        if isinstance(payload[field.name], datetime):
            payload[field.name] = _timestamp(payload[field.name])
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _record_from_json(value):
    payload = json.loads(value)
    for name in ("prediction_time", "target_interval_start", "target_interval_end", "source_retrieved_at", "input_data_cutoff", "history_start", "history_end", "issued_at"):
        payload[name] = _parse_timestamp(payload[name])
    return ForecastRecord(**payload)


def issue_forecast(database_path, model_path, current_pm25_path, now=lambda: datetime.now(timezone.utc)):
    store = SQLiteForecastStore(database_path)
    try:
        from app.main import MODEL_VERSION, _current_prediction_request, _load_current_artifact, _predict, _validate_metadata
        from scripts.modeling.predict import load_artifact
        store.initialize()
        model, metadata = load_artifact(model_path)
        _validate_metadata(metadata)
        artifact, retrieved_at = _load_current_artifact(Path(current_pm25_path))
        issued_at = _utc(now(), "issued_at")
        payload = _current_prediction_request(artifact, issued_at)
        prediction = _predict(model, metadata, payload)
        age = max(0.0, (issued_at - payload.prediction_time).total_seconds() / 60)
        stale = age > 240
        record = ForecastRecord.create(sensor_id=artifact["sensor_id"], prediction_time=payload.prediction_time,
            predicted_pm25=prediction, persistence_prediction=payload.history[-1].pm25, model_version=MODEL_VERSION,
            model_artifact_sha256=sha256_file(model_path), feature_schema_sha256=feature_schema_sha256(),
            artifact_version=artifact["artifact_version"], feature_configuration=metadata["feature_configuration"],
            source_retrieved_at=retrieved_at, input_data_cutoff=payload.prediction_time,
            history_start=payload.history[0].event_time, history_end=payload.history[-1].event_time,
            data_mode_at_issue="stale_openaq" if stale else "fresh_openaq",
            freshness_status_at_issue="stale" if stale else "fresh", source_age_minutes_at_issue=age, issued_at=issued_at)
        existing = store.get_by_identity(record)
        if existing is not None:
            return IssueResult("already_exists", existing)
        result = store.insert(record)
        return IssueResult("issued" if result.status == "inserted" else "already_exists", result.record)
    except ForecastIntegrityError:
        return IssueResult("failed")
    except ValueError as error:
        if "no completed hourly intervals" in str(error) or "lacks contiguous" in str(error):
            return IssueResult("not_eligible")
        return IssueResult("failed")
    except Exception:
        return IssueResult("failed")
