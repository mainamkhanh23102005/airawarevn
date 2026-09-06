import hashlib
import importlib
import json
import math
import os
import sqlite3
import unicodedata
import uuid
from contextlib import closing
from dataclasses import asdict, dataclass, fields
from datetime import datetime, time, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

from scripts.modeling.features import FORECAST_HORIZON_HOURS, TARGET_COLUMN, V1_FEATURE_COLUMNS


FORECAST_NAMESPACE = uuid.UUID("e60f879b-90cf-51a4-8936-aad3c962371c")
EVALUATION_NAMESPACE = uuid.UUID("1e772429-52c1-5f08-a957-88d6a5f5abbd")
EVALUATION_RUN_NAMESPACE = uuid.UUID("3c03496f-c34f-5b71-a62a-3a3b6430c613")
CONSUMER_PUBLICATION_NAMESPACE = uuid.UUID("16ca2b67-725b-52f8-9a45-9c085fd6307d")
SCHEMA_VERSION = 9
IDENTITY_FIELDS = (
    "sensor_id", "prediction_time", "target_interval_start", "target_interval_end",
    "forecast_horizon_hours", "model_version", "model_artifact_sha256", "feature_schema_sha256",
)


class ForecastIntegrityError(Exception):
    pass


class LedgerDatabaseError(Exception):
    pass


class LedgerTransientError(LedgerDatabaseError):
    pass


class ForecastStoreConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class InsertResult:
    status: str
    record: "ForecastRecord"


@dataclass(frozen=True)
class IssueResult:
    outcome: str
    record: "ForecastRecord | None" = None


@dataclass(frozen=True)
class EvaluationRecord:
    evaluation_id: str
    forecast_id: str
    reconciliation_id: str
    sensor_id: int
    target_interval_start: datetime
    target_interval_end: datetime
    predicted_pm25: float
    persistence_prediction: float
    observed_pm25: float
    model_error: float
    persistence_error: float
    evaluated_at: datetime
    evaluation_policy_version: int = 1

    @classmethod
    def create(cls, forecast_id, reconciliation_id, sensor_id, target_interval_start, target_interval_end,
               predicted_pm25, persistence_prediction, observed_pm25, evaluated_at, evaluation_id=None,
               model_error=None, persistence_error=None, evaluation_policy_version=1):
        if not isinstance(forecast_id, str) or not forecast_id or not isinstance(reconciliation_id, str) or not reconciliation_id:
            raise ValueError("evaluation relationships must be nonempty")
        if type(sensor_id) is not int or sensor_id <= 0 or type(evaluation_policy_version) is not int or evaluation_policy_version <= 0:
            raise ValueError("invalid evaluation identity")
        target_interval_start = _utc(target_interval_start, "target_interval_start", True)
        target_interval_end = _utc(target_interval_end, "target_interval_end", True)
        evaluated_at = _utc(evaluated_at, "evaluated_at")
        if target_interval_end != target_interval_start + timedelta(hours=1):
            raise ValueError("invalid evaluation target interval")
        values = {"predicted_pm25": predicted_pm25, "persistence_prediction": persistence_prediction, "observed_pm25": observed_pm25}
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("evaluation pm25 values must be finite and nonnegative")
        model_error = float(predicted_pm25) - float(observed_pm25)
        persistence_error = float(persistence_prediction) - float(observed_pm25)
        record = cls(evaluation_id or "", forecast_id, reconciliation_id, sensor_id, target_interval_start, target_interval_end,
            float(predicted_pm25), float(persistence_prediction), float(observed_pm25), model_error, persistence_error,
            evaluated_at, evaluation_policy_version)
        expected_id = str(uuid.uuid5(EVALUATION_NAMESPACE, canonical_evaluation_identity_json(record)))
        if evaluation_id is not None and evaluation_id != expected_id:
            raise ForecastIntegrityError("evaluation_id does not match natural identity")
        return cls(**{**asdict(record), "evaluation_id": evaluation_id or expected_id})


@dataclass(frozen=True)
class EvaluationResult:
    status: str
    record: EvaluationRecord


@dataclass(frozen=True)
class EvaluationMetrics:
    count: int
    model_mae: float | None
    model_rmse: float | None
    persistence_mae: float | None
    persistence_rmse: float | None
    mae_improvement: float | None
    rmse_improvement: float | None
    mae_improvement_percent: float | None
    rmse_improvement_percent: float | None


@dataclass(frozen=True)
class EvaluationRunSnapshot:
    snapshot_id: str
    evaluation_policy_version: int
    model_version: str
    model_artifact_sha256: str
    feature_schema_sha256: str
    sensor_id: int
    target_interval_end_start: datetime
    target_interval_end_end: datetime
    evaluation_ids: tuple[str, ...]
    metrics: EvaluationMetrics
    created_at: datetime


@dataclass(frozen=True)
class EvaluationRunResult:
    status: str
    snapshot: EvaluationRunSnapshot


@dataclass(frozen=True)
class ConsumerPerformancePublication:
    publication_id: str
    publication_date: str
    range_start_utc: datetime
    range_end_utc: datetime
    model_version: str
    model_artifact_sha256: str
    feature_schema_sha256: str
    sensor_id: int
    evaluation_policy_version: int
    forecast_horizon_hours: int
    verified_count: int
    mature_issued_count: int
    available: bool
    reason: str | None
    model_mae: float | None
    persistence_mae: float | None
    mae_difference: float | None
    snapshot_id: str | None
    membership_sha256: str
    published_at: datetime


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


def canonical_evaluation_identity_json(record):
    payload = {
        "evaluation_identity_version": 1,
        "forecast_id": record.forecast_id,
        "reconciliation_id": record.reconciliation_id,
        "evaluation_policy_version": record.evaluation_policy_version,
    }
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


def _translate_sqlite_errors(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except sqlite3.OperationalError as error:
            message = str(error).lower()
            error_type = LedgerTransientError if "locked" in message or "busy" in message else LedgerDatabaseError
            raise error_type(str(error)) from error
        except sqlite3.Error as error:
            raise LedgerDatabaseError(str(error)) from error
    return wrapped


def _raise_libsql_error(error):
    message = str(error)
    lowered = message.lower()
    if "constraint failed" in lowered or "constraint violation" in lowered:
        raise sqlite3.IntegrityError(message) from error
    transient_markers = ("busy", "locked", "timeout", "temporarily unavailable", "try again", "conflict")
    error_type = LedgerTransientError if any(marker in lowered for marker in transient_markers) else LedgerDatabaseError
    raise error_type(message) from error


class _LibSQLRow:
    def __init__(self, columns, values):
        self._columns = tuple(columns)
        self._values = tuple(values)
        self._index = {name: index for index, name in enumerate(self._columns)}

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._values[self._index[key]]
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def keys(self):
        return self._columns


class _LibSQLCursor:
    def __init__(self, cursor, driver_error):
        self._cursor = cursor
        self._driver_error = driver_error
        description = cursor.description or ()
        self._columns = tuple(column[0] for column in description)

    def _row(self, row):
        return None if row is None else _LibSQLRow(self._columns, row)

    def fetchone(self):
        try:
            return self._row(self._cursor.fetchone())
        except (self._driver_error, ValueError) as error:
            _raise_libsql_error(error)

    def fetchall(self):
        try:
            return [self._row(row) for row in (self._cursor.fetchall() or [])]
        except (self._driver_error, ValueError) as error:
            _raise_libsql_error(error)

    def __iter__(self):
        return self

    def __next__(self):
        row = self.fetchone()
        if row is None:
            raise StopIteration
        return row


class _LibSQLConnection:
    def __init__(self, connection, driver_error, read_only=False):
        self._connection = connection
        self._driver_error = driver_error
        self._read_only = read_only

    def _guard_read_only(self, sql):
        if not self._read_only:
            return
        statement = sql.lstrip().upper()
        if statement.startswith("SELECT") or statement.startswith("EXPLAIN"):
            return
        if statement.startswith((
                "PRAGMA TABLE_LIST",
                "PRAGMA TABLE_INFO(",
                "PRAGMA INDEX_LIST(",
                "PRAGMA INDEX_INFO(",
                "PRAGMA FOREIGN_KEY_LIST(",
        )):
            return
        raise LedgerDatabaseError("libSQL forecast store is read-only")

    def execute(self, sql, parameters=None):
        self._guard_read_only(sql)
        try:
            cursor = self._connection.execute(sql) if parameters is None else self._connection.execute(sql, parameters)
            return _LibSQLCursor(cursor, self._driver_error)
        except (self._driver_error, ValueError) as error:
            _raise_libsql_error(error)

    def commit(self):
        try:
            return self._connection.commit()
        except (self._driver_error, ValueError) as error:
            _raise_libsql_error(error)

    def rollback(self):
        try:
            return self._connection.rollback()
        except (self._driver_error, ValueError) as error:
            _raise_libsql_error(error)

    def close(self):
        try:
            return self._connection.close()
        except (self._driver_error, ValueError) as error:
            _raise_libsql_error(error)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        return False


class ForecastStore(Protocol):
    def validate_existing(self): ...
    def initialize(self): ...
    def acquire_monitoring_lease(self, lease_name, owner_id, now, ttl_seconds): ...
    def release_monitoring_lease(self, lease_name, owner_id): ...
    def insert(self, record): ...
    def get_by_id(self, forecast_id): ...
    def get_by_identity(self, record): ...
    def latest(self): ...
    def eligible_forecasts(self, now, delay_minutes, limit=None): ...
    def forecasts_for_targets(self, targets): ...
    def persist_acquisition(self, batch): ...
    def settle(self, forecast, batch, candidates, reconciled_at, policy): ...
    def materialize_evaluation(self, forecast_id, evaluated_at=None): ...
    def pending_evaluation_forecast_ids(self, limit=None): ...
    def pending_evaluation_snapshot_forecast_ids(self, limit=None): ...
    def pending_evaluation_snapshot_windows(self, limit=None): ...
    def create_evaluation_run_snapshot(self, model_version, model_artifact_sha256, feature_schema_sha256, sensor_id,
                                       target_interval_end_start, target_interval_end_end, created_at,
                                       evaluation_policy_version=1): ...
    def find_evaluation_run_snapshot(self, model_version, model_artifact_sha256, feature_schema_sha256, sensor_id,
                                     target_interval_end_start, target_interval_end_end,
                                     evaluation_policy_version=1): ...
    def publish_consumer_performance(self, model_version, model_artifact_sha256, feature_schema_sha256, sensor_id,
                                     reference_time, published_at=None, evaluation_policy_version=1,
                                     forecast_horizon_hours=6, minimum_verified_count=48): ...
    def current_consumer_performance(self, model_version, model_artifact_sha256, feature_schema_sha256, sensor_id,
                                     evaluation_policy_version=1, forecast_horizon_hours=6): ...
    def get_evaluation(self, forecast_id): ...


class SQLiteForecastStore:
    def __init__(self, path, read_only=False):
        self.path = Path(path)
        self.read_only = read_only

    def _connect(self):
        database = f"{self.path.resolve().as_uri()}?mode=ro" if self.read_only else self.path
        connection = sqlite3.connect(database, timeout=30, uri=self.read_only)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _forecast_schema(self):
        declarations = []
        for field in fields(ForecastRecord):
            kind = "INTEGER" if field.name in {"sensor_id", "forecast_horizon_hours", "artifact_version"} else "REAL" if field.name in {"predicted_pm25", "persistence_prediction", "source_age_minutes_at_issue"} else "TEXT"
            declarations.append(f"{field.name} {kind} NOT NULL" + (" PRIMARY KEY" if field.name == "forecast_id" else ""))
        return "CREATE TABLE forecasts (" + ", ".join(declarations) + ", UNIQUE (" + ", ".join(IDENTITY_FIELDS) + "))"

    def _validate_forecasts_schema(self, connection):
        columns = {row[1]: row for row in connection.execute("PRAGMA table_info(forecasts)")}
        if set(columns) != {field.name for field in fields(ForecastRecord)} or columns.get("forecast_id", (None,) * 6)[5] != 1:
            raise RuntimeError("unsupported forecasts schema")
        indexes = list(connection.execute("PRAGMA index_list(forecasts)"))
        unique_columns = [tuple(row[2] for row in connection.execute(f"PRAGMA index_info({index[1]})")) for index in indexes if index[2]]
        if IDENTITY_FIELDS not in unique_columns:
            raise RuntimeError("unsupported forecasts schema")

    def _validate_m2_schema(self, connection):
        expected_columns = {
            "observation_acquisitions": {"acquisition_id", "sensor_id", "requested_interval_start", "requested_interval_end", "retrieved_at", "source_provider", "source_endpoint", "http_status", "raw_payload_sha256", "normalized_payload_sha256", "created_at"},
            "forecast_reconciliations": {"reconciliation_id", "forecast_id", "acquisition_id", "sensor_id", "target_interval_start", "target_interval_end", "observed_pm25", "unit", "observation_event_time", "observation_period_end", "source_record_ids_json", "source_retrieved_at", "raw_payload_sha256", "normalized_candidate_sha256", "reconciled_at", "truth_policy_version", "reconciliation_delay_minutes"},
            "observation_revisions": {"revision_id", "reconciliation_id", "acquisition_id", "observed_pm25", "unit", "source_record_ids_json", "source_retrieved_at", "raw_payload_sha256", "normalized_candidate_sha256", "detected_at"},
        }
        expected_primary = {
            "observation_acquisitions": "acquisition_id",
            "forecast_reconciliations": "reconciliation_id",
            "observation_revisions": "revision_id",
        }
        expected_unique = {
            "observation_acquisitions": {("acquisition_id",), ("sensor_id", "requested_interval_start", "requested_interval_end", "retrieved_at", "raw_payload_sha256")},
            "forecast_reconciliations": {("reconciliation_id",), ("forecast_id",)},
            "observation_revisions": {("revision_id",), ("reconciliation_id", "acquisition_id")},
        }
        expected_foreign = {
            "observation_acquisitions": set(),
            "forecast_reconciliations": {("acquisition_id", "observation_acquisitions", "acquisition_id"), ("forecast_id", "forecasts", "forecast_id")},
            "observation_revisions": {("acquisition_id", "observation_acquisitions", "acquisition_id"), ("reconciliation_id", "forecast_reconciliations", "reconciliation_id")},
        }
        for table, columns in expected_columns.items():
            info = list(connection.execute(f"PRAGMA table_info({table})"))
            primary = {row[1] for row in info if row[5]}
            if {row[1] for row in info} != columns or primary != {expected_primary[table]}:
                raise RuntimeError("unsupported schema")
            indexes = list(connection.execute(f"PRAGMA index_list({table})"))
            unique = {tuple(row[2] for row in connection.execute(f"PRAGMA index_info({index[1]})")) for index in indexes if index[2]}
            if not expected_unique[table] <= unique:
                raise RuntimeError("unsupported schema")
            foreign = {(row[3], row[2], row[4]) for row in connection.execute(f"PRAGMA foreign_key_list({table})")}
            if foreign != expected_foreign[table]:
                raise RuntimeError("unsupported schema")

    def _validate_m3_schema(self, connection):
        table = "evaluation_rows"
        columns = {"evaluation_id", "forecast_id", "reconciliation_id", "sensor_id", "target_interval_start",
            "target_interval_end", "predicted_pm25", "persistence_prediction", "observed_pm25", "model_error",
            "persistence_error", "evaluated_at", "evaluation_policy_version"}
        info = list(connection.execute(f"PRAGMA table_info({table})"))
        if ({row[1] for row in info} != columns or {row[1] for row in info if row[5]} != {"evaluation_id"}
                or any(not row[3] for row in info)):
            raise RuntimeError("unsupported schema")
        indexes = list(connection.execute(f"PRAGMA index_list({table})"))
        unique = {tuple(row[2] for row in connection.execute(f"PRAGMA index_info({index[1]})")) for index in indexes if index[2]}
        if not {("evaluation_id",), ("forecast_id",), ("reconciliation_id",)} <= unique:
            raise RuntimeError("unsupported schema")
        foreign = {(row[3], row[2], row[4]) for row in connection.execute(f"PRAGMA foreign_key_list({table})")}
        if foreign != {("forecast_id", "forecasts", "forecast_id"), ("reconciliation_id", "forecast_reconciliations", "reconciliation_id")}:
            raise RuntimeError("unsupported schema")

    def _create_m5_schema(self, connection):
        connection.execute("""CREATE INDEX evaluation_rows_current_window_idx ON evaluation_rows
            (sensor_id, evaluation_policy_version, target_interval_end, forecast_id)""")

    def _create_m6_schema(self, connection):
        connection.execute("""CREATE TABLE evaluation_cohort_cursors (
            evaluation_policy_version INTEGER NOT NULL, model_version TEXT NOT NULL,
            model_artifact_sha256 TEXT NOT NULL, feature_schema_sha256 TEXT NOT NULL,
            sensor_id INTEGER NOT NULL, latest_target_interval_end TEXT NOT NULL,
            latest_evaluated_at TEXT NOT NULL, evaluation_count INTEGER NOT NULL,
            PRIMARY KEY (evaluation_policy_version, model_version, model_artifact_sha256,
                feature_schema_sha256, sensor_id))""")
        connection.execute("""CREATE INDEX evaluation_cohort_cursors_pending_idx
            ON evaluation_cohort_cursors (evaluation_count, latest_target_interval_end,
                sensor_id, model_version, model_artifact_sha256, feature_schema_sha256)""")
        connection.execute("""CREATE INDEX evaluation_run_snapshots_identity_window_idx
            ON evaluation_run_snapshots (evaluation_policy_version, model_version,
                model_artifact_sha256, feature_schema_sha256, sensor_id,
                target_interval_end_start, target_interval_end_end)""")

    def _validate_m6_schema(self, connection):
        columns = {"evaluation_policy_version", "model_version", "model_artifact_sha256",
            "feature_schema_sha256", "sensor_id", "latest_target_interval_end",
            "latest_evaluated_at", "evaluation_count"}
        info = list(connection.execute("PRAGMA table_info(evaluation_cohort_cursors)"))
        if ({row[1] for row in info} != columns
                or {row[1] for row in info if row[5]} != {"evaluation_policy_version", "model_version",
                    "model_artifact_sha256", "feature_schema_sha256", "sensor_id"}
                or any(not row[3] for row in info)):
            raise RuntimeError("unsupported schema")
        expected_indexes = {
            "evaluation_rows_current_window_idx": ("sensor_id", "evaluation_policy_version", "target_interval_end", "forecast_id"),
            "evaluation_cohort_cursors_pending_idx": ("evaluation_count", "latest_target_interval_end", "sensor_id", "model_version", "model_artifact_sha256", "feature_schema_sha256"),
            "evaluation_run_snapshots_identity_window_idx": ("evaluation_policy_version", "model_version", "model_artifact_sha256", "feature_schema_sha256", "sensor_id", "target_interval_end_start", "target_interval_end_end"),
        }
        for name, expected_columns in expected_indexes.items():
            actual_columns = tuple(row[2] for row in connection.execute(f"PRAGMA index_info({name})"))
            if actual_columns != expected_columns:
                raise RuntimeError("unsupported schema")

    def _create_m3_schema(self, connection):
        connection.execute("""CREATE TABLE evaluation_rows (
            evaluation_id TEXT NOT NULL PRIMARY KEY, forecast_id TEXT NOT NULL UNIQUE, reconciliation_id TEXT NOT NULL UNIQUE,
            sensor_id INTEGER NOT NULL, target_interval_start TEXT NOT NULL, target_interval_end TEXT NOT NULL,
            predicted_pm25 REAL NOT NULL, persistence_prediction REAL NOT NULL, observed_pm25 REAL NOT NULL,
            model_error REAL NOT NULL, persistence_error REAL NOT NULL, evaluated_at TEXT NOT NULL,
            evaluation_policy_version INTEGER NOT NULL,
            FOREIGN KEY(forecast_id) REFERENCES forecasts(forecast_id),
            FOREIGN KEY(reconciliation_id) REFERENCES forecast_reconciliations(reconciliation_id))""")

    def _create_m4_schema(self, connection):
        connection.execute("""CREATE TABLE evaluation_run_snapshots (
            snapshot_id TEXT NOT NULL PRIMARY KEY, evaluation_policy_version INTEGER NOT NULL,
            model_version TEXT NOT NULL, model_artifact_sha256 TEXT NOT NULL, feature_schema_sha256 TEXT NOT NULL,
            sensor_id INTEGER NOT NULL, target_interval_end_start TEXT NOT NULL, target_interval_end_end TEXT NOT NULL,
            evaluation_ids_json TEXT NOT NULL, count INTEGER NOT NULL, model_mae REAL, model_rmse REAL,
            persistence_mae REAL, persistence_rmse REAL, mae_improvement REAL, rmse_improvement REAL,
            mae_improvement_percent REAL, rmse_improvement_percent REAL, created_at TEXT NOT NULL)""")

    def _validate_m4_schema(self, connection):
        columns = {"snapshot_id", "evaluation_policy_version", "model_version", "model_artifact_sha256",
            "feature_schema_sha256", "sensor_id", "target_interval_end_start", "target_interval_end_end",
            "evaluation_ids_json", "count", "model_mae", "model_rmse", "persistence_mae", "persistence_rmse",
            "mae_improvement", "rmse_improvement", "mae_improvement_percent", "rmse_improvement_percent", "created_at"}
        info = list(connection.execute("PRAGMA table_info(evaluation_run_snapshots)"))
        if ({row[1] for row in info} != columns or {row[1] for row in info if row[5]} != {"snapshot_id"}
                or any(not row[3] for row in info if row[1] not in {"model_mae", "model_rmse", "persistence_mae", "persistence_rmse", "mae_improvement", "rmse_improvement", "mae_improvement_percent", "rmse_improvement_percent"})):
            raise RuntimeError("unsupported schema")

    def _create_m8_schema(self, connection):
        connection.execute("""CREATE TABLE IF NOT EXISTS consumer_performance_publications (
            publication_id TEXT NOT NULL PRIMARY KEY, publication_date TEXT NOT NULL,
            range_start_utc TEXT NOT NULL, range_end_utc TEXT NOT NULL,
            model_version TEXT NOT NULL, model_artifact_sha256 TEXT NOT NULL,
            feature_schema_sha256 TEXT NOT NULL, sensor_id INTEGER NOT NULL,
            evaluation_policy_version INTEGER NOT NULL, forecast_horizon_hours INTEGER NOT NULL,
            verified_count INTEGER NOT NULL, mature_issued_count INTEGER NOT NULL,
            available INTEGER NOT NULL, reason TEXT, model_mae REAL, persistence_mae REAL,
            mae_difference REAL, snapshot_id TEXT, membership_sha256 TEXT NOT NULL,
            published_at TEXT NOT NULL,
            UNIQUE(publication_date, model_version, model_artifact_sha256, feature_schema_sha256,
                sensor_id, evaluation_policy_version, forecast_horizon_hours, membership_sha256,
                mature_issued_count),
            FOREIGN KEY(snapshot_id) REFERENCES evaluation_run_snapshots(snapshot_id))""")
        connection.execute("""CREATE INDEX IF NOT EXISTS consumer_performance_current_idx
            ON consumer_performance_publications (publication_date, model_version,
                model_artifact_sha256, feature_schema_sha256, sensor_id,
                evaluation_policy_version, forecast_horizon_hours, published_at)""")

    def _validate_m8_schema(self, connection):
        expected = {"publication_id", "publication_date", "range_start_utc", "range_end_utc",
            "model_version", "model_artifact_sha256", "feature_schema_sha256", "sensor_id",
            "evaluation_policy_version", "forecast_horizon_hours", "verified_count",
            "mature_issued_count", "available", "reason", "model_mae", "persistence_mae",
            "mae_difference", "snapshot_id", "membership_sha256", "published_at"}
        info = list(connection.execute("PRAGMA table_info(consumer_performance_publications)"))
        if {row[1] for row in info} != expected or {row[1] for row in info if row[5]} != {"publication_id"}:
            raise RuntimeError("unsupported schema")
        foreign = {(row[3], row[2], row[4]) for row in connection.execute(
            "PRAGMA foreign_key_list(consumer_performance_publications)")}
        if foreign != {("snapshot_id", "evaluation_run_snapshots", "snapshot_id")}:
            raise RuntimeError("unsupported schema")

    def _create_m9_schema(self, connection):
        connection.execute("""CREATE TABLE monitoring_leases (
            lease_name TEXT NOT NULL PRIMARY KEY, owner_id TEXT NOT NULL,
            expires_at TEXT NOT NULL, acquired_at TEXT NOT NULL)""")

    def _validate_m9_schema(self, connection):
        info = list(connection.execute("PRAGMA table_info(monitoring_leases)"))
        expected = {"lease_name", "owner_id", "expires_at", "acquired_at"}
        if ({row[1] for row in info} != expected
                or {row[1] for row in info if row[5]} != {"lease_name"}
                or any(not row[3] for row in info)):
            raise RuntimeError("unsupported schema")

    def _create_m2_schema(self, connection):
        connection.execute("""CREATE TABLE observation_acquisitions (
            acquisition_id TEXT PRIMARY KEY, sensor_id INTEGER NOT NULL,
            requested_interval_start TEXT NOT NULL, requested_interval_end TEXT NOT NULL,
            retrieved_at TEXT NOT NULL, source_provider TEXT NOT NULL, source_endpoint TEXT NOT NULL,
            http_status INTEGER NOT NULL, raw_payload_sha256 TEXT NOT NULL,
            normalized_payload_sha256 TEXT NOT NULL, created_at TEXT NOT NULL,
            UNIQUE(sensor_id, requested_interval_start, requested_interval_end, retrieved_at, raw_payload_sha256))""")
        connection.execute("""CREATE TABLE forecast_reconciliations (
            reconciliation_id TEXT PRIMARY KEY, forecast_id TEXT NOT NULL UNIQUE,
            acquisition_id TEXT NOT NULL, sensor_id INTEGER NOT NULL,
            target_interval_start TEXT NOT NULL, target_interval_end TEXT NOT NULL,
            observed_pm25 REAL NOT NULL, unit TEXT NOT NULL,
            observation_event_time TEXT NOT NULL, observation_period_end TEXT NOT NULL,
            source_record_ids_json TEXT NOT NULL, source_retrieved_at TEXT NOT NULL,
            raw_payload_sha256 TEXT NOT NULL, normalized_candidate_sha256 TEXT NOT NULL,
            reconciled_at TEXT NOT NULL, truth_policy_version INTEGER NOT NULL,
            reconciliation_delay_minutes INTEGER NOT NULL,
            FOREIGN KEY(forecast_id) REFERENCES forecasts(forecast_id),
            FOREIGN KEY(acquisition_id) REFERENCES observation_acquisitions(acquisition_id))""")
        connection.execute("""CREATE TABLE observation_revisions (
            revision_id TEXT PRIMARY KEY, reconciliation_id TEXT NOT NULL,
            acquisition_id TEXT NOT NULL, observed_pm25 REAL NOT NULL, unit TEXT NOT NULL,
            source_record_ids_json TEXT NOT NULL, source_retrieved_at TEXT NOT NULL,
            raw_payload_sha256 TEXT NOT NULL, normalized_candidate_sha256 TEXT NOT NULL,
            detected_at TEXT NOT NULL,
            UNIQUE(reconciliation_id, acquisition_id),
            FOREIGN KEY(reconciliation_id) REFERENCES forecast_reconciliations(reconciliation_id),
            FOREIGN KEY(acquisition_id) REFERENCES observation_acquisitions(acquisition_id))""")

    @_translate_sqlite_errors
    def validate_existing(self):
        with closing(self._connect()) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
            required = {"forecasts", "observation_acquisitions", "forecast_reconciliations", "observation_revisions", "evaluation_rows", "evaluation_run_snapshots", "evaluation_cohort_cursors", "consumer_performance_publications", "monitoring_leases"}
            if version != SCHEMA_VERSION or not required <= tables:
                raise RuntimeError("unsupported schema")
            self._validate_forecasts_schema(connection)
            self._validate_m2_schema(connection)
            self._validate_m3_schema(connection)
            self._validate_m4_schema(connection)
            self._validate_m6_schema(connection)
            self._validate_m8_schema(connection)
            self._validate_m9_schema(connection)

    @_translate_sqlite_errors
    def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
            if version > SCHEMA_VERSION:
                raise RuntimeError("unsupported schema version")
            if version == 0 and tables:
                raise RuntimeError("unsupported schema")
            if version == SCHEMA_VERSION:
                self._validate_forecasts_schema(connection)
                required = {"forecasts", "observation_acquisitions", "forecast_reconciliations", "observation_revisions", "evaluation_rows", "evaluation_run_snapshots", "evaluation_cohort_cursors", "consumer_performance_publications", "monitoring_leases"}
                if not required <= tables:
                    raise RuntimeError("unsupported schema")
                self._validate_m2_schema(connection)
                self._validate_m3_schema(connection)
                self._validate_m4_schema(connection)
                self._validate_m6_schema(connection)
                self._validate_m8_schema(connection)
                self._validate_m9_schema(connection)
                return
            connection.execute("BEGIN IMMEDIATE")
            try:
                if version == 0:
                    connection.execute(self._forecast_schema())
                    self._create_m2_schema(connection)
                    self._create_m3_schema(connection)
                elif version == 1:
                    self._validate_forecasts_schema(connection)
                    self._create_m2_schema(connection)
                    self._create_m3_schema(connection)
                elif version == 2:
                    self._validate_forecasts_schema(connection)
                    self._validate_m2_schema(connection)
                    self._create_m3_schema(connection)
                elif version == 3:
                    self._validate_forecasts_schema(connection)
                    self._validate_m2_schema(connection)
                    self._validate_m3_schema(connection)
                elif version == 4:
                    self._validate_forecasts_schema(connection)
                    self._validate_m2_schema(connection)
                    self._validate_m3_schema(connection)
                    self._validate_m4_schema(connection)
                    self._create_m5_schema(connection)
                    self._create_m6_schema(connection)
                    self._backfill_evaluation_cohort_cursors(connection)
                    self._create_m8_schema(connection)
                    self._validate_m8_schema(connection)
                    self._create_m9_schema(connection)
                    self._validate_m9_schema(connection)
                    connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                    connection.commit()
                    return
                elif version == 5:
                    self._validate_forecasts_schema(connection)
                    self._validate_m2_schema(connection)
                    self._validate_m3_schema(connection)
                    self._validate_m4_schema(connection)
                    expected_index = ("sensor_id", "evaluation_policy_version", "target_interval_end", "forecast_id")
                    actual_index = tuple(row[2] for row in connection.execute("PRAGMA index_info(evaluation_rows_current_window_idx)"))
                    if actual_index != expected_index:
                        raise RuntimeError("unsupported schema")
                    self._create_m6_schema(connection)
                    self._backfill_evaluation_cohort_cursors(connection)
                    self._create_m8_schema(connection)
                    self._validate_m8_schema(connection)
                    self._create_m9_schema(connection)
                    self._validate_m9_schema(connection)
                    connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                    connection.commit()
                    return
                elif version == 6:
                    self._validate_forecasts_schema(connection)
                    self._validate_m2_schema(connection)
                    self._validate_m3_schema(connection)
                    self._validate_m4_schema(connection)
                    self._validate_m6_schema(connection)
                    connection.execute("DELETE FROM evaluation_cohort_cursors")
                    self._backfill_evaluation_cohort_cursors(connection)
                    self._create_m8_schema(connection)
                    self._validate_m8_schema(connection)
                    self._create_m9_schema(connection)
                    self._validate_m9_schema(connection)
                    connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                    connection.commit()
                    return
                elif version == 7:
                    self._validate_forecasts_schema(connection)
                    self._validate_m2_schema(connection)
                    self._validate_m3_schema(connection)
                    self._validate_m4_schema(connection)
                    self._validate_m6_schema(connection)
                    self._create_m8_schema(connection)
                    self._validate_m8_schema(connection)
                    self._create_m9_schema(connection)
                    self._validate_m9_schema(connection)
                    connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                    connection.commit()
                    return
                elif version == 8:
                    self._validate_forecasts_schema(connection)
                    self._validate_m2_schema(connection)
                    self._validate_m3_schema(connection)
                    self._validate_m4_schema(connection)
                    self._validate_m6_schema(connection)
                    self._validate_m8_schema(connection)
                    if "monitoring_leases" in tables:
                        raise RuntimeError("unsupported schema")
                    self._create_m9_schema(connection)
                    self._validate_m9_schema(connection)
                    connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                    connection.commit()
                    return
                else:
                    raise RuntimeError("unsupported schema version")
                self._create_m4_schema(connection)
                self._create_m5_schema(connection)
                self._create_m6_schema(connection)
                self._create_m8_schema(connection)
                self._create_m9_schema(connection)
                self._backfill_evaluation_cohort_cursors(connection)
                connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    @_translate_sqlite_errors
    def acquire_monitoring_lease(self, lease_name, owner_id, now, ttl_seconds):
        if not isinstance(lease_name, str) or not lease_name:
            raise ValueError("lease_name must be nonempty")
        if not isinstance(owner_id, str) or not owner_id:
            raise ValueError("owner_id must be nonempty")
        now = _utc(now, "now")
        if type(ttl_seconds) is not int or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be a positive integer")
        expires_at = now + timedelta(seconds=ttl_seconds)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT owner_id, expires_at, acquired_at FROM monitoring_leases WHERE lease_name=?",
                    (lease_name,)).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO monitoring_leases(lease_name, owner_id, expires_at, acquired_at) VALUES (?,?,?,?)",
                        (lease_name, owner_id, _timestamp(expires_at), _timestamp(now)))
                    connection.commit()
                    return True
                if _parse_timestamp(row["expires_at"]) <= now:
                    connection.execute(
                        "UPDATE monitoring_leases SET owner_id=?, expires_at=?, acquired_at=? WHERE lease_name=?",
                        (owner_id, _timestamp(expires_at), _timestamp(now), lease_name))
                    connection.commit()
                    return True
                if row["owner_id"] == owner_id:
                    connection.execute(
                        "UPDATE monitoring_leases SET expires_at=? WHERE lease_name=? AND owner_id=?",
                        (_timestamp(expires_at), lease_name, owner_id))
                    connection.commit()
                    return True
                connection.commit()
                return False
            except Exception:
                connection.rollback()
                raise

    @_translate_sqlite_errors
    def release_monitoring_lease(self, lease_name, owner_id):
        if not isinstance(lease_name, str) or not lease_name:
            raise ValueError("lease_name must be nonempty")
        if not isinstance(owner_id, str) or not owner_id:
            raise ValueError("owner_id must be nonempty")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT owner_id FROM monitoring_leases WHERE lease_name=?", (lease_name,)).fetchone()
                if row is None or row["owner_id"] != owner_id:
                    connection.commit()
                    return False
                connection.execute(
                    "DELETE FROM monitoring_leases WHERE lease_name=? AND owner_id=?", (lease_name, owner_id))
                connection.commit()
                return True
            except Exception:
                connection.rollback()
                raise

    def _after_insert(self, connection, record):
        pass

    def _values(self, record):
        return tuple(_timestamp(value) if isinstance(value, datetime) else value for value in asdict(record).values())

    @_translate_sqlite_errors
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

    @_translate_sqlite_errors
    def get_by_id(self, forecast_id):
        with closing(self._connect()) as connection:
            return self._row(connection.execute("SELECT * FROM forecasts WHERE forecast_id=?", (forecast_id,)).fetchone())

    @_translate_sqlite_errors
    def get_by_identity(self, record):
        with closing(self._connect()) as connection:
            return self._get_by_identity_connection(connection, record)

    @_translate_sqlite_errors
    def latest(self):
        with closing(self._connect()) as connection:
            return self._row(connection.execute("SELECT * FROM forecasts ORDER BY issued_at DESC, rowid DESC LIMIT 1").fetchone())

    @_translate_sqlite_errors
    def eligible_forecasts(self, now, delay_minutes, limit=None):
        cutoff = _timestamp(_utc(now, "now") - timedelta(minutes=delay_minutes))
        sql = """SELECT forecasts.* FROM forecasts
            LEFT JOIN forecast_reconciliations USING(forecast_id)
            WHERE forecast_reconciliations.forecast_id IS NULL AND forecasts.target_interval_end<=?
            ORDER BY forecasts.target_interval_end ASC, forecasts.forecast_id ASC"""
        parameters = [cutoff]
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(limit)
        with closing(self._connect()) as connection:
            return [self._row(row) for row in connection.execute(sql, parameters)]

    @_translate_sqlite_errors
    def forecasts_for_targets(self, targets):
        if not targets:
            return []
        clauses, values = [], []
        for sensor_id, start, end in targets:
            clauses.append("(sensor_id=? AND target_interval_start=? AND target_interval_end=?)")
            values.extend((sensor_id, _timestamp(start), _timestamp(end)))
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT * FROM forecasts WHERE " + " OR ".join(clauses) + " ORDER BY target_interval_end, forecast_id", values)
            return [self._row(row) for row in rows]

    def count(self):
        with closing(self._connect()) as connection:
            return connection.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]

    def _acquisition_values(self, batch):
        return (batch.acquisition_id, batch.sensor_id, _timestamp(batch.requested_interval_start),
            _timestamp(batch.requested_interval_end), _timestamp(batch.retrieved_at), batch.source_provider,
            batch.source_endpoint, batch.http_status, batch.raw_payload_sha256, batch.normalized_payload_sha256,
            _timestamp(batch.created_at))

    def _persist_acquisition_connection(self, connection, batch):
        values = self._acquisition_values(batch)
        existing = connection.execute("SELECT * FROM observation_acquisitions WHERE acquisition_id=?", (batch.acquisition_id,)).fetchone()
        if existing is None:
            try:
                connection.execute("INSERT INTO observation_acquisitions VALUES (?,?,?,?,?,?,?,?,?,?,?)", values)
                return
            except sqlite3.IntegrityError:
                existing = connection.execute("SELECT * FROM observation_acquisitions WHERE acquisition_id=?", (batch.acquisition_id,)).fetchone()
        if existing is None or tuple(existing) != values:
            raise ForecastIntegrityError("immutable acquisition conflict")

    @_translate_sqlite_errors
    def persist_acquisition(self, batch):
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._persist_acquisition_connection(connection, batch)
            connection.commit()

    @_translate_sqlite_errors
    def settle(self, forecast, batch, candidates, reconciled_at, policy):
        from app.ground_truth_reconciler import ReconcileResult, _canonical_json, _identity, normalized_sha256, RECONCILIATION_NAMESPACE, REVISION_NAMESPACE
        value_text = candidates[0][1]
        value = float(value_text)
        ids = sorted((record.get("record_id") for record, _ in candidates), key=lambda item: (item is not None, str(item)))
        ids_json = _canonical_json([None if item is None else str(item) for item in ids])
        candidate_hash = normalized_sha256([record for record, _ in candidates])
        reconciliation_id = _identity(RECONCILIATION_NAMESPACE, {"reconciliation_identity_version": 1,
            "forecast_id": forecast.forecast_id, "truth_policy_version": policy.version})
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._persist_acquisition_connection(connection, batch)
            stored_forecast = connection.execute("SELECT * FROM forecasts WHERE forecast_id=?", (forecast.forecast_id,)).fetchone()
            stored_acquisition = connection.execute("SELECT * FROM observation_acquisitions WHERE acquisition_id=?", (batch.acquisition_id,)).fetchone()
            if stored_forecast is None or self._row(stored_forecast) != forecast:
                raise ForecastIntegrityError("forecast relationship conflict")
            request_start = _parse_timestamp(stored_acquisition["requested_interval_start"])
            request_end = _parse_timestamp(stored_acquisition["requested_interval_end"])
            if stored_acquisition["sensor_id"] != forecast.sensor_id or request_start > forecast.target_interval_start or request_end < forecast.target_interval_end:
                raise ForecastIntegrityError("acquisition relationship conflict")
            existing = connection.execute("SELECT * FROM forecast_reconciliations WHERE forecast_id=?", (forecast.forecast_id,)).fetchone()
            if existing is None:
                connection.execute("""INSERT INTO forecast_reconciliations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (reconciliation_id, forecast.forecast_id, batch.acquisition_id, forecast.sensor_id,
                     _timestamp(forecast.target_interval_start), _timestamp(forecast.target_interval_end), value, "µg/m³",
                     _timestamp(forecast.target_interval_start), _timestamp(forecast.target_interval_end), ids_json,
                     _timestamp(batch.retrieved_at), batch.raw_payload_sha256, candidate_hash, _timestamp(reconciled_at),
                     policy.version, policy.delay_minutes))
                connection.commit()
                return ReconcileResult("reconciled")
            if existing["observed_pm25"] == value:
                connection.commit()
                return ReconcileResult("already_reconciled")
            revision_id = _identity(REVISION_NAMESPACE, {"revision_identity_version": 1,
                "reconciliation_id": existing["reconciliation_id"], "acquisition_id": batch.acquisition_id})
            revision_values = (revision_id, existing["reconciliation_id"], batch.acquisition_id, value, "µg/m³", ids_json,
                _timestamp(batch.retrieved_at), batch.raw_payload_sha256, candidate_hash, _timestamp(reconciled_at))
            stored_revision = connection.execute("SELECT * FROM observation_revisions WHERE revision_id=?", (revision_id,)).fetchone()
            if stored_revision is None:
                try:
                    connection.execute("INSERT INTO observation_revisions VALUES (?,?,?,?,?,?,?,?,?,?)", revision_values)
                except sqlite3.IntegrityError:
                    stored_revision = connection.execute("SELECT * FROM observation_revisions WHERE revision_id=?", (revision_id,)).fetchone()
            if stored_revision is not None and tuple(stored_revision) != revision_values:
                raise ForecastIntegrityError("immutable revision conflict")
            connection.commit()
            return ReconcileResult("revision_detected")

    def _evaluation_values(self, record):
        return tuple(_timestamp(value) if isinstance(value, datetime) else value for value in asdict(record).values())

    def _update_evaluation_cohort_cursor(self, connection, record):
        forecast = connection.execute("""SELECT model_version, model_artifact_sha256,
            feature_schema_sha256 FROM forecasts WHERE forecast_id=?""", (record.forecast_id,)).fetchone()
        connection.execute("""INSERT INTO evaluation_cohort_cursors VALUES (?,?,?,?,?,?,?,1)
            ON CONFLICT(evaluation_policy_version, model_version, model_artifact_sha256,
                feature_schema_sha256, sensor_id) DO UPDATE SET
                latest_evaluated_at=CASE WHEN excluded.latest_target_interval_end>
                    evaluation_cohort_cursors.latest_target_interval_end OR (
                    excluded.latest_target_interval_end=evaluation_cohort_cursors.latest_target_interval_end
                    AND excluded.latest_evaluated_at>evaluation_cohort_cursors.latest_evaluated_at)
                    THEN excluded.latest_evaluated_at ELSE evaluation_cohort_cursors.latest_evaluated_at END,
                latest_target_interval_end=MAX(evaluation_cohort_cursors.latest_target_interval_end,
                    excluded.latest_target_interval_end),
                evaluation_count=evaluation_cohort_cursors.evaluation_count+1""",
            (record.evaluation_policy_version, forecast["model_version"], forecast["model_artifact_sha256"],
             forecast["feature_schema_sha256"], record.sensor_id, _timestamp(record.target_interval_end),
             _timestamp(record.evaluated_at)))

    def _backfill_evaluation_cohort_cursors(self, connection):
        connection.execute("""INSERT INTO evaluation_cohort_cursors
            SELECT evaluation_policy_version, model_version, model_artifact_sha256,
                feature_schema_sha256, sensor_id, target_interval_end, evaluated_at,
                evaluation_count
            FROM (
                SELECT evaluation_rows.evaluation_policy_version, forecasts.model_version,
                    forecasts.model_artifact_sha256, forecasts.feature_schema_sha256,
                    evaluation_rows.sensor_id, evaluation_rows.target_interval_end,
                    evaluation_rows.evaluated_at,
                    COUNT(*) OVER cohort AS evaluation_count,
                    ROW_NUMBER() OVER cohort_order AS row_number
                FROM evaluation_rows JOIN forecasts USING(forecast_id)
                WINDOW cohort AS (PARTITION BY evaluation_rows.evaluation_policy_version,
                    forecasts.model_version, forecasts.model_artifact_sha256,
                    forecasts.feature_schema_sha256, evaluation_rows.sensor_id),
                cohort_order AS (PARTITION BY evaluation_rows.evaluation_policy_version,
                    forecasts.model_version, forecasts.model_artifact_sha256,
                    forecasts.feature_schema_sha256, evaluation_rows.sensor_id
                    ORDER BY evaluation_rows.target_interval_end DESC,
                        evaluation_rows.evaluated_at DESC, evaluation_rows.evaluation_id DESC)
            ) WHERE row_number=1""")

    def _evaluation_row(self, row):
        if row is None:
            return None
        payload = dict(row)
        try:
            stored_errors = (payload["model_error"], payload["persistence_error"])
            for name in ("target_interval_start", "target_interval_end", "evaluated_at"):
                payload[name] = _parse_timestamp(payload[name])
            record = EvaluationRecord.create(**payload)
            if stored_errors != (record.model_error, record.persistence_error):
                raise ValueError("evaluation errors do not match predictions and truth")
            return record
        except (TypeError, ValueError, ForecastIntegrityError) as error:
            raise ForecastIntegrityError("invalid durable evaluation row") from error

    def _validate_durable_evaluation_relationship(self, connection, row):
        record = self._evaluation_row(row)
        self._validate_evaluation_relationship(connection, record)
        return record

    def _validate_evaluation_relationship(self, connection, record):
        forecast = connection.execute("SELECT * FROM forecasts WHERE forecast_id=?", (record.forecast_id,)).fetchone()
        reconciliation = connection.execute("SELECT * FROM forecast_reconciliations WHERE reconciliation_id=?", (record.reconciliation_id,)).fetchone()
        if forecast is None or reconciliation is None or reconciliation["forecast_id"] != record.forecast_id:
            raise ForecastIntegrityError("evaluation relationship conflict")
        expected = (forecast["sensor_id"], forecast["target_interval_start"], forecast["target_interval_end"],
            forecast["predicted_pm25"], forecast["persistence_prediction"], reconciliation["observed_pm25"])
        actual = (record.sensor_id, _timestamp(record.target_interval_start), _timestamp(record.target_interval_end),
            record.predicted_pm25, record.persistence_prediction, record.observed_pm25)
        if expected != actual:
            raise ForecastIntegrityError("evaluation relationship conflict")

    def _persist_evaluation_connection(self, connection, record):
        expected_id = str(uuid.uuid5(EVALUATION_NAMESPACE, canonical_evaluation_identity_json(record)))
        if record.evaluation_id != expected_id:
            raise ForecastIntegrityError("evaluation_id does not match natural identity")
        if (any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0
                for value in (record.predicted_pm25, record.persistence_prediction, record.observed_pm25))
                or any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
                       for value in (record.model_error, record.persistence_error))
                or record.model_error != record.predicted_pm25 - record.observed_pm25
                or record.persistence_error != record.persistence_prediction - record.observed_pm25):
            raise ForecastIntegrityError("evaluation value conflict")
        self._validate_evaluation_relationship(connection, record)
        existing = self._evaluation_row(connection.execute("SELECT * FROM evaluation_rows WHERE forecast_id=?", (record.forecast_id,)).fetchone())
        if existing is not None:
            if existing != record:
                raise ForecastIntegrityError("immutable evaluation conflict")
            return EvaluationResult("already_exists", existing)
        try:
            connection.execute("INSERT INTO evaluation_rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", self._evaluation_values(record))
        except sqlite3.IntegrityError as error:
            raise ForecastIntegrityError("evaluation identity conflict") from error
        self._update_evaluation_cohort_cursor(connection, record)
        return EvaluationResult("inserted", record)

    def persist_evaluation(self, record):
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            result = self._persist_evaluation_connection(connection, record)
            connection.commit()
            return result

    def _materialize_evaluation_connection(self, connection, forecast_id, evaluated_at=None):
        row = connection.execute("""SELECT forecasts.forecast_id, forecast_reconciliations.reconciliation_id,
            forecasts.sensor_id, forecasts.target_interval_start, forecasts.target_interval_end,
            forecasts.predicted_pm25, forecasts.persistence_prediction, forecast_reconciliations.observed_pm25,
            forecast_reconciliations.reconciled_at
            FROM forecasts JOIN forecast_reconciliations USING(forecast_id) WHERE forecasts.forecast_id=?""", (forecast_id,)).fetchone()
        if row is None:
            raise ForecastIntegrityError("missing initial reconciliation")
        record = EvaluationRecord.create(row["forecast_id"], row["reconciliation_id"], row["sensor_id"],
            _parse_timestamp(row["target_interval_start"]), _parse_timestamp(row["target_interval_end"]),
            row["predicted_pm25"], row["persistence_prediction"], row["observed_pm25"],
            evaluated_at or _parse_timestamp(row["reconciled_at"]))
        return self._persist_evaluation_connection(connection, record)

    @_translate_sqlite_errors
    def materialize_evaluation(self, forecast_id, evaluated_at=None):
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            result = self._materialize_evaluation_connection(connection, forecast_id, evaluated_at)
            connection.commit()
            return result

    @_translate_sqlite_errors
    def pending_evaluation_forecast_ids(self, limit=None):
        sql = """SELECT forecast_reconciliations.forecast_id FROM forecast_reconciliations
            LEFT JOIN evaluation_rows USING(forecast_id) WHERE evaluation_rows.forecast_id IS NULL
            ORDER BY forecast_reconciliations.target_interval_end ASC, forecast_reconciliations.forecast_id ASC"""
        parameters = []
        if limit is not None:
            if type(limit) is not int or limit <= 0:
                raise ValueError("limit must be a positive integer")
            sql += " LIMIT ?"
            parameters.append(limit)
        with closing(self._connect()) as connection:
            return [row[0] for row in connection.execute(sql, parameters)]

    @_translate_sqlite_errors
    def pending_evaluation_snapshot_forecast_ids(self, limit=None):
        sql = """SELECT evaluation_rows.forecast_id FROM evaluation_rows JOIN forecasts USING(forecast_id)
            WHERE NOT EXISTS (SELECT 1 FROM evaluation_run_snapshots WHERE evaluation_run_snapshots.evaluation_policy_version=evaluation_rows.evaluation_policy_version
                AND evaluation_run_snapshots.model_version=forecasts.model_version
                AND evaluation_run_snapshots.model_artifact_sha256=forecasts.model_artifact_sha256
                AND evaluation_run_snapshots.feature_schema_sha256=forecasts.feature_schema_sha256
                AND evaluation_run_snapshots.sensor_id=evaluation_rows.sensor_id
                AND evaluation_run_snapshots.target_interval_end_start=evaluation_rows.target_interval_end
                AND evaluation_run_snapshots.target_interval_end_end=strftime('%Y-%m-%dT%H:%M:%SZ', datetime(evaluation_rows.target_interval_end, '+1 hour')))
            ORDER BY evaluation_rows.target_interval_end ASC, evaluation_rows.forecast_id ASC"""
        parameters = []
        if limit is not None:
            if type(limit) is not int or limit <= 0:
                raise ValueError("limit must be a positive integer")
            sql += " LIMIT ?"
            parameters.append(limit)
        with closing(self._connect()) as connection:
            return [row[0] for row in connection.execute(sql, parameters)]

    @_translate_sqlite_errors
    def pending_evaluation_snapshot_windows(self, limit=None):
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("limit must be a positive integer")
        sql = """SELECT model_version, model_artifact_sha256, feature_schema_sha256, sensor_id,
                evaluation_policy_version, latest_target_interval_end, latest_evaluated_at
                FROM evaluation_cohort_cursors
                WHERE evaluation_count>=2
                ORDER BY latest_target_interval_end, sensor_id, model_version, model_artifact_sha256, feature_schema_sha256"""
        with closing(self._connect()) as connection:
            rows = connection.execute(sql).fetchall()
            windows = []
            for row in rows:
                cohort = connection.execute("""SELECT evaluation_rows.evaluation_id,
                    MIN(evaluation_rows.target_interval_end) OVER () AS start
                    FROM evaluation_rows JOIN forecasts USING(forecast_id)
                    WHERE evaluation_rows.evaluation_policy_version=? AND forecasts.model_version=?
                    AND forecasts.model_artifact_sha256=? AND forecasts.feature_schema_sha256=?
                    AND evaluation_rows.sensor_id=?
                    ORDER BY evaluation_rows.target_interval_end, evaluation_rows.evaluation_id""",
                    (row["evaluation_policy_version"], row["model_version"], row["model_artifact_sha256"],
                     row["feature_schema_sha256"], row["sensor_id"])).fetchall()
                evaluation_ids = tuple(item["evaluation_id"] for item in cohort)
                start = cohort[0]["start"]
                end = _timestamp(_parse_timestamp(row["latest_target_interval_end"]) + timedelta(hours=1))
                snapshots = connection.execute("""SELECT * FROM evaluation_run_snapshots
                    WHERE evaluation_policy_version=? AND model_version=? AND model_artifact_sha256=?
                    AND feature_schema_sha256=? AND sensor_id=? AND target_interval_end_start=?
                    AND target_interval_end_end=?""", (row["evaluation_policy_version"], row["model_version"],
                    row["model_artifact_sha256"], row["feature_schema_sha256"], row["sensor_id"], start, end)).fetchall()
                if any(self._snapshot_row(connection, snapshot).evaluation_ids == evaluation_ids for snapshot in snapshots):
                    continue
                windows.append((row["model_version"], row["model_artifact_sha256"], row["feature_schema_sha256"],
                    row["sensor_id"], _parse_timestamp(start), _parse_timestamp(row["latest_target_interval_end"]) + timedelta(hours=1),
                    _parse_timestamp(row["latest_evaluated_at"]), row["evaluation_policy_version"]))
                if limit is not None and len(windows) == limit:
                    break
        return windows


    def backfill_evaluations(self, limit=None):
        sql = """SELECT forecast_reconciliations.forecast_id FROM forecast_reconciliations
            LEFT JOIN evaluation_rows USING(forecast_id) WHERE evaluation_rows.forecast_id IS NULL
            ORDER BY forecast_reconciliations.target_interval_end ASC, forecast_reconciliations.forecast_id ASC"""
        parameters = []
        if limit is not None:
            if type(limit) is not int or limit <= 0:
                raise ValueError("limit must be a positive integer")
            sql += " LIMIT ?"
            parameters.append(limit)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            forecast_ids = [row[0] for row in connection.execute(sql, parameters)]
            results = [self._materialize_evaluation_connection(connection, forecast_id) for forecast_id in forecast_ids]
            connection.commit()
            return results

    def _evaluation_metrics(self, rows):
        count = len(rows)
        if not count:
            return EvaluationMetrics(0, None, None, None, None, None, None, None, None)
        model_errors = [row.model_error if isinstance(row, EvaluationRecord) else row["model_error"] for row in rows]
        persistence_errors = [row.persistence_error if isinstance(row, EvaluationRecord) else row["persistence_error"] for row in rows]
        model_mae = sum(abs(error) for error in model_errors) / count
        persistence_mae = sum(abs(error) for error in persistence_errors) / count
        model_rmse = math.sqrt(sum(error ** 2 for error in model_errors) / count)
        persistence_rmse = math.sqrt(sum(error ** 2 for error in persistence_errors) / count)
        mae_improvement = persistence_mae - model_mae
        rmse_improvement = persistence_rmse - model_rmse
        return EvaluationMetrics(count, model_mae, model_rmse, persistence_mae, persistence_rmse,
            mae_improvement, rmse_improvement,
            None if persistence_mae == 0 else mae_improvement / persistence_mae * 100,
            None if persistence_rmse == 0 else rmse_improvement / persistence_rmse * 100)

    def _evaluation_cohort_rows(self, connection, model_version, model_artifact_sha256, feature_schema_sha256,
                               sensor_id, target_interval_end_start, target_interval_end_end, evaluation_policy_version):
        if not isinstance(model_version, str) or not model_version or type(sensor_id) is not int or sensor_id <= 0:
            raise ValueError("invalid evaluation cohort")
        _hash(model_artifact_sha256, "model_artifact_sha256")
        _hash(feature_schema_sha256, "feature_schema_sha256")
        if type(evaluation_policy_version) is not int or evaluation_policy_version <= 0:
            raise ValueError("invalid evaluation policy version")
        start = _utc(target_interval_end_start, "target_interval_end_start", True)
        end = _utc(target_interval_end_end, "target_interval_end_end", True)
        if end <= start:
            raise ValueError("invalid evaluation window")
        rows = list(connection.execute("""SELECT evaluation_rows.* FROM evaluation_rows JOIN forecasts USING(forecast_id)
            WHERE forecasts.model_version=? AND forecasts.model_artifact_sha256=? AND forecasts.feature_schema_sha256=?
            AND evaluation_rows.sensor_id=? AND evaluation_rows.target_interval_end>=? AND evaluation_rows.target_interval_end<?
            AND evaluation_rows.evaluation_policy_version=? ORDER BY evaluation_rows.target_interval_end, evaluation_rows.evaluation_id""",
            (model_version, model_artifact_sha256, feature_schema_sha256, sensor_id, _timestamp(start), _timestamp(end), evaluation_policy_version)))
        for row in rows:
            self._validate_durable_evaluation_relationship(connection, row)
        return rows

    def evaluation_metrics(self, model_version, model_artifact_sha256, feature_schema_sha256, sensor_id,
                           target_interval_end_start, target_interval_end_end, evaluation_policy_version=1):
        with closing(self._connect()) as connection:
            rows = self._evaluation_cohort_rows(connection, model_version, model_artifact_sha256, feature_schema_sha256,
                sensor_id, target_interval_end_start, target_interval_end_end, evaluation_policy_version)
            return self._evaluation_metrics(rows)

    @_translate_sqlite_errors
    def create_evaluation_run_snapshot(self, model_version, model_artifact_sha256, feature_schema_sha256, sensor_id,
                                       target_interval_end_start, target_interval_end_end, created_at,
                                       evaluation_policy_version=1):
        created_at = _utc(created_at, "created_at")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                result = self._create_evaluation_run_snapshot(connection, model_version,
                    model_artifact_sha256, feature_schema_sha256, sensor_id,
                    target_interval_end_start, target_interval_end_end, created_at,
                    evaluation_policy_version)
                connection.commit()
                return result
            except Exception:
                connection.rollback()
                raise

    def _create_evaluation_run_snapshot(self, connection, model_version, model_artifact_sha256,
                                        feature_schema_sha256, sensor_id, target_interval_end_start,
                                        target_interval_end_end, created_at, evaluation_policy_version=1,
                                        rows=None):
        rows = rows if rows is not None else self._evaluation_cohort_rows(connection, model_version,
            model_artifact_sha256, feature_schema_sha256, sensor_id, target_interval_end_start,
            target_interval_end_end, evaluation_policy_version)
        start = _utc(target_interval_end_start, "target_interval_end_start", True)
        end = _utc(target_interval_end_end, "target_interval_end_end", True)
        evaluation_ids = tuple(row["evaluation_id"] for row in rows)
        identity = json.dumps({"snapshot_identity_version": 1, "evaluation_policy_version": evaluation_policy_version,
            "model_version": model_version, "model_artifact_sha256": model_artifact_sha256,
            "feature_schema_sha256": feature_schema_sha256, "sensor_id": sensor_id,
            "target_interval_end_start": _timestamp(start), "target_interval_end_end": _timestamp(end),
            "evaluation_ids": evaluation_ids}, sort_keys=True, separators=(",", ":"))
        snapshot_id = str(uuid.uuid5(EVALUATION_RUN_NAMESPACE, identity))
        existing = connection.execute("SELECT * FROM evaluation_run_snapshots WHERE snapshot_id=?", (snapshot_id,)).fetchone()
        metrics = self._evaluation_metrics(rows)
        if existing is not None:
            snapshot = self._snapshot_row(connection, existing)
            if snapshot.evaluation_ids != evaluation_ids or snapshot.metrics != metrics:
                raise ForecastIntegrityError("snapshot does not match selected evaluations")
            return EvaluationRunResult("already_exists", snapshot)
        connection.execute("INSERT INTO evaluation_run_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (snapshot_id, evaluation_policy_version, model_version, model_artifact_sha256, feature_schema_sha256, sensor_id,
            _timestamp(start), _timestamp(end), json.dumps(evaluation_ids, separators=(",", ":")), metrics.count,
            metrics.model_mae, metrics.model_rmse, metrics.persistence_mae, metrics.persistence_rmse,
            metrics.mae_improvement, metrics.rmse_improvement, metrics.mae_improvement_percent,
            metrics.rmse_improvement_percent, _timestamp(created_at)))
        return EvaluationRunResult("inserted", EvaluationRunSnapshot(snapshot_id, evaluation_policy_version, model_version,
            model_artifact_sha256, feature_schema_sha256, sensor_id, start, end, evaluation_ids, metrics, created_at))

    def _snapshot_row(self, connection, row):
        try:
            evaluation_ids = json.loads(row["evaluation_ids_json"])
            if (not isinstance(evaluation_ids, list) or any(not isinstance(value, str) or not value for value in evaluation_ids)
                    or len(evaluation_ids) != len(set(evaluation_ids)) or row["count"] != len(evaluation_ids)
                    or type(row["count"]) is not int or row["count"] < 0):
                raise ValueError("invalid snapshot provenance")
            start = _parse_timestamp(row["target_interval_end_start"])
            end = _parse_timestamp(row["target_interval_end_end"])
            if end <= start:
                raise ValueError("invalid snapshot window")
            rows = []
            for evaluation_id in evaluation_ids:
                evaluation = connection.execute("SELECT * FROM evaluation_rows WHERE evaluation_id=?", (evaluation_id,)).fetchone()
                if evaluation is None:
                    raise ValueError("missing snapshot evaluation")
                record = self._validate_durable_evaluation_relationship(connection, evaluation)
                forecast = connection.execute("SELECT model_version, model_artifact_sha256, feature_schema_sha256 FROM forecasts WHERE forecast_id=?", (record.forecast_id,)).fetchone()
                if (record.evaluation_policy_version != row["evaluation_policy_version"] or record.sensor_id != row["sensor_id"]
                        or not start <= record.target_interval_end < end or forecast is None
                        or (forecast["model_version"], forecast["model_artifact_sha256"], forecast["feature_schema_sha256"])
                        != (row["model_version"], row["model_artifact_sha256"], row["feature_schema_sha256"])):
                    raise ValueError("snapshot evaluation provenance conflict")
                rows.append(record)
            values = (row["model_mae"], row["model_rmse"], row["persistence_mae"], row["persistence_rmse"],
                row["mae_improvement"], row["rmse_improvement"], row["mae_improvement_percent"], row["rmse_improvement_percent"])
            if any(value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)) for value in values):
                raise ValueError("invalid snapshot metrics")
            metrics = self._evaluation_metrics(rows)
            if tuple(values) != tuple(metrics.__dict__.values())[1:]:
                raise ValueError("snapshot metrics conflict")
            identity = json.dumps({"snapshot_identity_version": 1, "evaluation_policy_version": row["evaluation_policy_version"],
                "model_version": row["model_version"], "model_artifact_sha256": row["model_artifact_sha256"],
                "feature_schema_sha256": row["feature_schema_sha256"], "sensor_id": row["sensor_id"],
                "target_interval_end_start": _timestamp(start), "target_interval_end_end": _timestamp(end),
                "evaluation_ids": tuple(evaluation_ids)}, sort_keys=True, separators=(",", ":"))
            if row["snapshot_id"] != str(uuid.uuid5(EVALUATION_RUN_NAMESPACE, identity)):
                raise ValueError("snapshot_id does not match provenance")
            return EvaluationRunSnapshot(row["snapshot_id"], row["evaluation_policy_version"], row["model_version"],
                row["model_artifact_sha256"], row["feature_schema_sha256"], row["sensor_id"], start, end,
                tuple(evaluation_ids), metrics, _parse_timestamp(row["created_at"]))
        except (TypeError, ValueError, ForecastIntegrityError, json.JSONDecodeError) as error:
            raise ForecastIntegrityError("invalid durable snapshot") from error

    @_translate_sqlite_errors
    def find_evaluation_run_snapshot(self, model_version, model_artifact_sha256, feature_schema_sha256, sensor_id,
                                     target_interval_end_start, target_interval_end_end, evaluation_policy_version=1):
        if not isinstance(model_version, str) or not model_version or type(sensor_id) is not int or sensor_id <= 0:
            raise ValueError("invalid evaluation cohort")
        _hash(model_artifact_sha256, "model_artifact_sha256")
        _hash(feature_schema_sha256, "feature_schema_sha256")
        if type(evaluation_policy_version) is not int or evaluation_policy_version <= 0:
            raise ValueError("invalid evaluation policy version")
        start = _utc(target_interval_end_start, "target_interval_end_start", True)
        end = _utc(target_interval_end_end, "target_interval_end_end", True)
        if end <= start:
            raise ValueError("invalid evaluation window")
        with closing(self._connect()) as connection:
            rows = list(connection.execute("""SELECT * FROM evaluation_run_snapshots
                WHERE evaluation_policy_version=? AND model_version=? AND model_artifact_sha256=?
                AND feature_schema_sha256=? AND sensor_id=? AND target_interval_end_start=?
                AND target_interval_end_end=? ORDER BY created_at, snapshot_id""", (evaluation_policy_version, model_version,
                model_artifact_sha256, feature_schema_sha256, sensor_id, _timestamp(start), _timestamp(end))))
            incompatible = connection.execute("""SELECT 1 FROM evaluation_rows JOIN forecasts USING(forecast_id)
                WHERE evaluation_rows.sensor_id=? AND evaluation_rows.evaluation_policy_version=?
                AND evaluation_rows.target_interval_end>=? AND evaluation_rows.target_interval_end<?
                AND (forecasts.model_version<>? OR forecasts.model_artifact_sha256<>? OR forecasts.feature_schema_sha256<>?)
                LIMIT 1""", (sensor_id, evaluation_policy_version, _timestamp(start), _timestamp(end),
                model_version, model_artifact_sha256, feature_schema_sha256)).fetchone()
            if incompatible is not None:
                raise ValueError("ambiguous evaluation cohort")
            if not rows:
                return None
            snapshots = [self._snapshot_row(connection, row) for row in rows]
            current_evaluation_ids = tuple(row["evaluation_id"] for row in self._evaluation_cohort_rows(
                connection, model_version, model_artifact_sha256, feature_schema_sha256, sensor_id,
                start, end, evaluation_policy_version))
            for snapshot in snapshots:
                if snapshot.evaluation_ids == current_evaluation_ids:
                    return snapshot
            return None

    def _consumer_publication_row(self, row):
        if row is None:
            return None
        return ConsumerPerformancePublication(
            row["publication_id"], row["publication_date"], _parse_timestamp(row["range_start_utc"]),
            _parse_timestamp(row["range_end_utc"]), row["model_version"], row["model_artifact_sha256"],
            row["feature_schema_sha256"], row["sensor_id"], row["evaluation_policy_version"],
            row["forecast_horizon_hours"], row["verified_count"], row["mature_issued_count"],
            bool(row["available"]), row["reason"], row["model_mae"], row["persistence_mae"],
            row["mae_difference"], row["snapshot_id"], row["membership_sha256"],
            _parse_timestamp(row["published_at"]))

    @_translate_sqlite_errors
    def publish_consumer_performance(self, model_version, model_artifact_sha256,
                                     feature_schema_sha256, sensor_id, reference_time,
                                     published_at=None, evaluation_policy_version=1,
                                     forecast_horizon_hours=6, minimum_verified_count=48):
        reference = _utc(reference_time, "reference_time")
        published = _utc(published_at or reference, "published_at")
        ict = ZoneInfo("Asia/Ho_Chi_Minh")
        publication_date = reference.astimezone(ict).date()
        end = datetime.combine(publication_date, time.min, ict).astimezone(timezone.utc)
        start = end - timedelta(days=30)
        _hash(model_artifact_sha256, "model_artifact_sha256")
        _hash(feature_schema_sha256, "feature_schema_sha256")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                rows = self._evaluation_cohort_rows(connection, model_version, model_artifact_sha256,
                    feature_schema_sha256, sensor_id, start, end, evaluation_policy_version)
                evaluation_ids = tuple(row["evaluation_id"] for row in rows)
                mature_count = connection.execute("""SELECT COUNT(*) FROM forecasts
                WHERE model_version=? AND model_artifact_sha256=? AND feature_schema_sha256=?
                AND sensor_id=? AND forecast_horizon_hours=? AND issuance_mode='scheduled'
                AND target_interval_end>=? AND target_interval_end<? AND target_interval_end<=?""",
                    (model_version, model_artifact_sha256, feature_schema_sha256, sensor_id,
                     forecast_horizon_hours, _timestamp(start), _timestamp(end),
                     _timestamp(reference - timedelta(minutes=120)))).fetchone()[0]
                verified_count = len(rows)
                if verified_count > mature_count:
                    raise ForecastIntegrityError("verified count exceeds mature issued count")
                snapshot = None
                if verified_count:
                    snapshot = self._create_evaluation_run_snapshot(connection, model_version,
                        model_artifact_sha256, feature_schema_sha256, sensor_id, start, end,
                        published, evaluation_policy_version, rows).snapshot
                membership = hashlib.sha256(json.dumps(evaluation_ids, separators=(",", ":")).encode()).hexdigest()
                available = verified_count >= minimum_verified_count
                metrics = self._evaluation_metrics(rows) if available else None
                identity = json.dumps({"consumer_publication_identity_version": 1,
            "publication_date": publication_date.isoformat(), "model_version": model_version,
            "model_artifact_sha256": model_artifact_sha256,
            "feature_schema_sha256": feature_schema_sha256, "sensor_id": sensor_id,
            "evaluation_policy_version": evaluation_policy_version,
            "forecast_horizon_hours": forecast_horizon_hours, "membership_sha256": membership,
            "mature_issued_count": mature_count}, sort_keys=True, separators=(",", ":"))
                publication_id = str(uuid.uuid5(CONSUMER_PUBLICATION_NAMESPACE, identity))
                values = (publication_id, publication_date.isoformat(), _timestamp(start), _timestamp(end),
            model_version, model_artifact_sha256, feature_schema_sha256, sensor_id,
            evaluation_policy_version, forecast_horizon_hours, verified_count, mature_count,
            int(available), None if available else "insufficient_history",
            metrics.model_mae if metrics else None, metrics.persistence_mae if metrics else None,
            metrics.mae_improvement if metrics else None, snapshot.snapshot_id if snapshot else None,
            membership, _timestamp(published))
                existing = connection.execute("SELECT * FROM consumer_performance_publications WHERE publication_id=?",
                    (publication_id,)).fetchone()
                if existing is None:
                    connection.execute("INSERT INTO consumer_performance_publications VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", values)
                    existing = connection.execute("SELECT * FROM consumer_performance_publications WHERE publication_id=?",
                        (publication_id,)).fetchone()
                elif tuple(existing)[:-1] != values[:-1]:
                    raise ForecastIntegrityError("publication does not match selected evaluations")
                connection.commit()
                return self._consumer_publication_row(existing)
            except Exception:
                connection.rollback()
                raise

    @_translate_sqlite_errors
    def current_consumer_performance(self, model_version, model_artifact_sha256,
                                     feature_schema_sha256, sensor_id,
                                     evaluation_policy_version=1, forecast_horizon_hours=6):
        with closing(self._connect()) as connection:
            row = connection.execute("""SELECT * FROM consumer_performance_publications
                WHERE model_version=? AND model_artifact_sha256=? AND feature_schema_sha256=?
                AND sensor_id=? AND evaluation_policy_version=? AND forecast_horizon_hours=?
                ORDER BY publication_date DESC, published_at DESC, rowid DESC LIMIT 1""",
                (model_version, model_artifact_sha256, feature_schema_sha256, sensor_id,
                 evaluation_policy_version, forecast_horizon_hours)).fetchone()
            return self._consumer_publication_row(row)

    @_translate_sqlite_errors
    def get_evaluation(self, forecast_id):
        with closing(self._connect()) as connection:
            return self._evaluation_row(connection.execute("SELECT * FROM evaluation_rows WHERE forecast_id=?", (forecast_id,)).fetchone())

    def count_evaluations(self):
        with closing(self._connect()) as connection:
            return connection.execute("SELECT COUNT(*) FROM evaluation_rows").fetchone()[0]

    def get_reconciliation(self, forecast_id):
        from app.ground_truth_reconciler import ReconciliationRecord
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM forecast_reconciliations WHERE forecast_id=?", (forecast_id,)).fetchone()
        if row is None:
            return None
        return ReconciliationRecord(row["reconciliation_id"], row["forecast_id"], row["acquisition_id"],
            row["observed_pm25"], _parse_timestamp(row["source_retrieved_at"]))

    def count_acquisitions(self):
        with closing(self._connect()) as connection:
            return connection.execute("SELECT COUNT(*) FROM observation_acquisitions").fetchone()[0]

    def count_revisions(self):
        with closing(self._connect()) as connection:
            return connection.execute("SELECT COUNT(*) FROM observation_revisions").fetchone()[0]


class LibSQLForecastStore(SQLiteForecastStore):
    def __init__(self, database_url, auth_token, read_only=False):
        self.database_url = database_url
        self.auth_token = auth_token
        self.read_only = read_only

    @staticmethod
    def _load_driver():
        try:
            return importlib.import_module("libsql")
        except ImportError as error:
            raise ForecastStoreConfigurationError(
                "AIRAWARE_LEDGER_BACKEND='libsql' requires the libsql package"
            ) from error

    def _connect(self):
        driver = self._load_driver()
        try:
            raw = driver.connect(
                database=self.database_url,
                auth_token=self.auth_token,
                timeout=30,
                isolation_level=None,
            )
            if not self.read_only:
                raw.execute("PRAGMA foreign_keys=ON")
        except (driver.Error, ValueError) as error:
            _raise_libsql_error(error)
        return _LibSQLConnection(raw, driver.Error, read_only=self.read_only)

    def _remote_tables(self, connection):
        rows = connection.execute("PRAGMA table_list").fetchall()
        return {row["name"] for row in rows
                if row["schema"] == "main" and row["type"] == "table" and not row["name"].startswith("sqlite_")}

    def _remote_schema_versions(self, connection):
        return tuple(row[0] for row in connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version").fetchall())

    def _validate_remote_schema(self, connection, expected_version=SCHEMA_VERSION):
        required = {"schema_migrations", "forecasts", "observation_acquisitions", "forecast_reconciliations",
            "observation_revisions", "evaluation_rows", "evaluation_run_snapshots", "evaluation_cohort_cursors",
            "consumer_performance_publications"}
        if expected_version >= 9:
            required.add("monitoring_leases")
        tables = self._remote_tables(connection)
        if not required <= tables:
            raise RuntimeError("unsupported schema")
        versions = self._remote_schema_versions(connection)
        if versions and max(versions) > SCHEMA_VERSION:
            raise RuntimeError("unsupported schema version")
        if versions != (expected_version,):
            raise RuntimeError("unsupported schema")
        self._validate_forecasts_schema(connection)
        self._validate_m2_schema(connection)
        self._validate_m3_schema(connection)
        self._validate_m4_schema(connection)
        self._validate_m6_schema(connection)
        self._validate_m8_schema(connection)
        if expected_version >= 9:
            self._validate_m9_schema(connection)

    def validate_existing(self):
        with closing(self._connect()) as connection:
            self._validate_remote_schema(connection)

    def initialize(self):
        if self.read_only:
            raise LedgerDatabaseError("libSQL forecast store is read-only")
        with closing(self._connect()) as connection:
            tables = self._remote_tables(connection)
            if tables:
                if "schema_migrations" not in tables:
                    raise RuntimeError("unsupported schema")
                versions = self._remote_schema_versions(connection)
                if versions == (SCHEMA_VERSION,):
                    self._validate_remote_schema(connection)
                    return
                if versions != (8,) or "monitoring_leases" in tables:
                    self._validate_remote_schema(connection)
                    return
                self._validate_remote_schema(connection, expected_version=8)
                connection.execute("BEGIN IMMEDIATE")
                try:
                    self._create_m9_schema(connection)
                    self._validate_m9_schema(connection)
                    connection.execute("DELETE FROM schema_migrations WHERE version=?", (8,))
                    connection.execute("INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                        (SCHEMA_VERSION, _timestamp(datetime.now(timezone.utc))))
                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise
                self._validate_remote_schema(connection)
                return
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute("""CREATE TABLE schema_migrations (
                    version INTEGER NOT NULL PRIMARY KEY, applied_at TEXT NOT NULL)""")
                connection.execute(self._forecast_schema())
                self._create_m2_schema(connection)
                self._create_m3_schema(connection)
                self._create_m4_schema(connection)
                self._create_m5_schema(connection)
                self._create_m6_schema(connection)
                self._create_m8_schema(connection)
                self._create_m9_schema(connection)
                connection.execute("INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (SCHEMA_VERSION, _timestamp(datetime.now(timezone.utc))))
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        self.validate_existing()

    def latest(self):
        with closing(self._connect()) as connection:
            return self._row(connection.execute(
                "SELECT * FROM forecasts ORDER BY issued_at DESC, forecast_id DESC LIMIT 1").fetchone())

    def current_consumer_performance(self, model_version, model_artifact_sha256,
                                     feature_schema_sha256, sensor_id,
                                     evaluation_policy_version=1, forecast_horizon_hours=6):
        with closing(self._connect()) as connection:
            row = connection.execute("""SELECT * FROM consumer_performance_publications
                WHERE model_version=? AND model_artifact_sha256=? AND feature_schema_sha256=?
                AND sensor_id=? AND evaluation_policy_version=? AND forecast_horizon_hours=?
                ORDER BY publication_date DESC, published_at DESC, publication_id DESC LIMIT 1""",
                (model_version, model_artifact_sha256, feature_schema_sha256, sensor_id,
                 evaluation_policy_version, forecast_horizon_hours)).fetchone()
            return self._consumer_publication_row(row)


def create_forecast_store(path, read_only=False):
    selected_backend = os.environ.get("AIRAWARE_LEDGER_BACKEND", "sqlite").strip().lower()
    if selected_backend == "sqlite":
        return SQLiteForecastStore(path, read_only=read_only)
    if selected_backend == "libsql":
        database_url = os.environ.get("AIRAWARE_TURSO_DATABASE_URL", "").strip()
        auth_token = os.environ.get("AIRAWARE_TURSO_AUTH_TOKEN", "").strip()
        missing = [name for name, value in (
            ("AIRAWARE_TURSO_DATABASE_URL", database_url), ("AIRAWARE_TURSO_AUTH_TOKEN", auth_token)) if not value]
        if missing:
            raise ForecastStoreConfigurationError(
                "libsql backend requires " + ", ".join(missing)
            )
        return LibSQLForecastStore(database_url, auth_token, read_only=read_only)
    raise ForecastStoreConfigurationError(
        f"unsupported AIRAWARE_LEDGER_BACKEND={selected_backend!r}; supported backends: sqlite, libsql"
    )


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
    store = create_forecast_store(database_path)
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
