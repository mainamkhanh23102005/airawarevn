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
SCHEMA_VERSION = 2
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
                required = {"forecasts", "observation_acquisitions", "forecast_reconciliations", "observation_revisions"}
                if not required <= tables:
                    raise RuntimeError("unsupported schema")
                self._validate_m2_schema(connection)
                return
            connection.execute("BEGIN IMMEDIATE")
            try:
                if version == 0:
                    connection.execute(self._forecast_schema())
                elif version == 1:
                    self._validate_forecasts_schema(connection)
                else:
                    raise RuntimeError("unsupported schema version")
                self._create_m2_schema(connection)
                connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                connection.commit()
            except Exception:
                connection.rollback()
                raise

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

    def persist_acquisition(self, batch):
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._persist_acquisition_connection(connection, batch)
            connection.commit()

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
