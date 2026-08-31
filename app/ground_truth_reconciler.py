import hashlib
import json
import math
import unicodedata
import uuid
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation

from app.forecast_ledger import ForecastIntegrityError, _hash, _parse_timestamp, _timestamp, _utc


ACQUISITION_NAMESPACE = uuid.UUID("4153ba6e-a333-57b1-8dfe-8b58604bdc30")
RECONCILIATION_NAMESPACE = uuid.UUID("76b6950a-b1ef-5e10-bd94-025e1b0c92c6")
REVISION_NAMESPACE = uuid.UUID("14ed017a-e290-5afc-9cf5-abcbd5c536cc")
CANONICAL_UNIT = "µg/m³"


def _canonical_json(payload):
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _identity(namespace, payload):
    return str(uuid.uuid5(namespace, _canonical_json(payload)))


def canonical_decimal(value, unit):
    if isinstance(value, bool):
        raise ValueError("PM2.5 must be numeric")
    try:
        value = value if isinstance(value, Decimal) else Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError("PM2.5 must be numeric") from error
    if not value.is_finite() or value.is_signed() or value < 0:
        raise ValueError("PM2.5 must be finite and nonnegative")
    if unit in {"µg/m³", "ug/m3", "μg/m³", "µg/m3"}:
        converted = value
    elif unit == "mg/m³":
        digits = value.as_tuple()
        converted = Decimal((digits.sign, digits.digits, digits.exponent + 3))
    else:
        raise ValueError("unsupported PM2.5 unit")
    text = format(converted, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _record(record):
    if record.get("malformed"):
        return {"malformed": True, "sensor_id": record.get("sensor_id"),
            "event_time": _timestamp(record["event_time"]) if isinstance(record.get("event_time"), datetime) else None,
            "period_end_utc": _timestamp(record["period_end_utc"]) if isinstance(record.get("period_end_utc"), datetime) else None,
            "value_decimal": str(record["value_decimal"]) if "value_decimal" in record else None,
            "unit": record.get("unit"), "record_id": None if record.get("record_id") is None else str(record["record_id"])}
    event_time = _utc(record["event_time"], "event_time", True)
    period_end = _utc(record["period_end_utc"], "period_end_utc", True)
    decimal_text = canonical_decimal(record["value_decimal"], record.get("unit"))
    record_id = record.get("record_id")
    return {
        "sensor_id": record["sensor_id"], "event_time": _timestamp(event_time),
        "period_end_utc": _timestamp(period_end), "pm25_ug_m3": decimal_text,
        "unit": CANONICAL_UNIT, "record_id": None if record_id is None else unicodedata.normalize("NFC", str(record_id)),
    }


def canonical_normalized_json(records):
    normalized = [_record(record) for record in records]
    normalized.sort(key=lambda item: (item.get("malformed", False), str(item.get("sensor_id")),
        item.get("event_time") or "", item.get("period_end_utc") or "", item.get("pm25_ug_m3") or "",
        item.get("value_decimal") or "", item.get("unit") or "", (item.get("record_id") is not None, item.get("record_id") or "")))
    return _canonical_json({"normalized_serialization_version": 1, "records": normalized})


def normalized_sha256(records):
    return hashlib.sha256(canonical_normalized_json(records).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TruthPolicy:
    version: int = 1
    delay_minutes: int = 120

    def __post_init__(self):
        if (self.version, self.delay_minutes) != (1, 120):
            raise ValueError("unsupported truth policy")


@dataclass(frozen=True)
class AcquisitionBatch:
    acquisition_id: str
    sensor_id: int
    requested_interval_start: datetime
    requested_interval_end: datetime
    retrieved_at: datetime
    source_provider: str
    source_endpoint: str
    http_status: int
    raw_payload_sha256: str
    normalized_payload_sha256: str
    created_at: datetime
    records: tuple

    @classmethod
    def create(cls, sensor_id, requested_interval_start, requested_interval_end, retrieved_at,
               source_endpoint, http_status, raw_payload_sha256, records, created_at,
               source_provider="OpenAQ", acquisition_id=None, normalized_payload_sha256=None):
        if type(sensor_id) is not int or sensor_id <= 0:
            raise ValueError("sensor_id must be a positive integer")
        if type(http_status) is not int or not 200 <= http_status < 300:
            raise ValueError("http_status must be successful")
        start = _utc(requested_interval_start, "requested_interval_start")
        end = _utc(requested_interval_end, "requested_interval_end")
        if not isinstance(retrieved_at, datetime) or not isinstance(created_at, datetime):
            raise ValueError("acquisition timestamps must be datetimes")
        retrieved = _utc(retrieved_at.replace(microsecond=0), "retrieved_at")
        created = _utc(created_at.replace(microsecond=0), "created_at")
        records = tuple(records)
        digest = normalized_payload_sha256 or normalized_sha256(records)
        payload = {"acquisition_identity_version": 1, "sensor_id": sensor_id,
            "requested_interval_start": _timestamp(start), "requested_interval_end": _timestamp(end),
            "retrieved_at": _timestamp(retrieved), "raw_payload_sha256": _hash(raw_payload_sha256, "raw_payload_sha256")}
        expected = _identity(ACQUISITION_NAMESPACE, payload)
        if acquisition_id is not None and acquisition_id != expected:
            raise ValueError("acquisition_id does not match identity")
        return cls(expected, sensor_id, start, end, retrieved, source_provider, source_endpoint, http_status,
            raw_payload_sha256, _hash(digest, "normalized_payload_sha256"), created, records)


@dataclass(frozen=True)
class ReconciliationRecord:
    reconciliation_id: str
    forecast_id: str
    acquisition_id: str
    observed_pm25: float
    source_retrieved_at: datetime


@dataclass(frozen=True)
class ReconcileResult:
    status: str
    reason_code: str | None = None
    forecast_id: str | None = None


@dataclass(frozen=True)
class RequestPlan:
    sensor_id: int
    start: datetime
    end: datetime
    forecasts: tuple


@dataclass(frozen=True)
class RunResult:
    results: tuple
    counts: dict


class SourceTransportError(Exception):
    pass


class SourceParseError(Exception):
    pass


class GroundTruthReconciler:
    def __init__(self, store, now, policy=TruthPolicy(), source=None, limit=None):
        self.store = store
        self.now = now
        self.policy = policy
        self.source = source
        self.limit = limit

    @staticmethod
    def plan_requests(forecasts):
        grouped = {}
        for forecast in forecasts:
            grouped.setdefault(forecast.sensor_id, {}).setdefault(
                (forecast.target_interval_start, forecast.target_interval_end), []).append(forecast)
        plans = []
        for sensor_id in sorted(grouped):
            intervals = sorted(grouped[sensor_id])
            current_start = current_end = None
            current_forecasts = []
            for start, end in intervals:
                rows = sorted(grouped[sensor_id][(start, end)], key=lambda item: item.forecast_id)
                if current_end is None or start > current_end:
                    if current_end is not None:
                        plans.append(RequestPlan(sensor_id, current_start, current_end, tuple(current_forecasts)))
                    current_start, current_end, current_forecasts = start, end, list(rows)
                else:
                    current_end = max(current_end, end)
                    current_forecasts.extend(rows)
            if current_end is not None:
                plans.append(RequestPlan(sensor_id, current_start, current_end, tuple(current_forecasts)))
        return tuple(plans)

    def _run_forecasts(self, forecasts, now):
        results = []
        for plan in self.plan_requests(forecasts):
            try:
                batch = self.source.acquire(plan.sensor_id, plan.start, plan.end)
            except SourceTransportError:
                results.extend(ReconcileResult("failed", "transport_error", item.forecast_id) for item in plan.forecasts)
                continue
            except SourceParseError:
                results.extend(ReconcileResult("failed", "parse_error", item.forecast_id) for item in plan.forecasts)
                continue
            target_keys = [(item.sensor_id, item.target_interval_start, item.target_interval_end) for item in plan.forecasts]
            relevant = self.store.forecasts_for_targets(target_keys)
            batches = batch if isinstance(batch, tuple) else (batch,)
            plan_results = {}
            for acquired in batches:
                for forecast in relevant:
                    try:
                        result = self.reconcile(forecast, acquired, now)
                    except ForecastIntegrityError:
                        result = ReconcileResult("failed", "database_error")
                    previous = plan_results.get(forecast.forecast_id)
                    if previous is None or result.status in {"reconciled", "revision_detected"}:
                        plan_results[forecast.forecast_id] = ReconcileResult(result.status, result.reason_code, forecast.forecast_id)
            results.extend(plan_results.values())
        results.sort(key=lambda item: next((index for index, forecast in enumerate(forecasts) if forecast.forecast_id == item.forecast_id), len(forecasts)))
        counts = {}
        for result in results:
            counts[result.status] = counts.get(result.status, 0) + 1
        return RunResult(tuple(results), counts)

    def run(self):
        now = _utc(self.now().replace(microsecond=0), "now")
        forecasts = self.store.eligible_forecasts(now, self.policy.delay_minutes, self.limit)
        return self._run_forecasts(forecasts, now)

    def run_for_targets(self, targets):
        now = _utc(self.now().replace(microsecond=0), "now")
        return self._run_forecasts(self.store.forecasts_for_targets(targets), now)

    def _candidates(self, forecast, batch):
        matching = []
        invalid_relevant = False
        for record in batch.records:
            sensor_id = record.get("sensor_id")
            if type(sensor_id) is int and sensor_id != forecast.sensor_id:
                continue
            if record.get("malformed"):
                if isinstance(record.get("event_time"), datetime) and record["event_time"] != forecast.target_interval_start:
                    continue
                invalid_relevant = True
                continue
            if type(sensor_id) is not int:
                invalid_relevant = True
                continue
            try:
                start = _utc(record["event_time"], "event_time")
                if start != forecast.target_interval_start:
                    continue
                end = _utc(record["period_end_utc"], "period_end_utc")
                if end != forecast.target_interval_end:
                    continue
                if start.minute or start.second or end.minute or end.second or end - start != timedelta(hours=1):
                    continue
                decimal_text = canonical_decimal(record.get("value_decimal"), record.get("unit"))
                if not math.isfinite(float(decimal_text)):
                    raise ValueError("PM2.5 must fit SQLite REAL")
                matching.append((record, decimal_text))
            except (KeyError, TypeError, ValueError):
                invalid_relevant = True
        if invalid_relevant and matching:
            return None, "ambiguous"
        if invalid_relevant:
            return None, "invalid_source"
        if not matching:
            return None, "not_available"
        values = {value for _, value in matching}
        if len(values) != 1:
            return None, "ambiguous"
        return matching, None

    def reconcile(self, forecast, batch, reconciliation_now=None):
        maturity = forecast.target_interval_end + timedelta(minutes=self.policy.delay_minutes)
        reconciliation_now = reconciliation_now or _utc(self.now().replace(microsecond=0), "now")
        if reconciliation_now < maturity:
            return ReconcileResult("pending", "not_mature")
        if batch.retrieved_at < maturity:
            return ReconcileResult("pending", "not_available")
        try:
            candidates, reason = self._candidates(forecast, batch)
            if reason:
                self.store.persist_acquisition(batch)
                return ReconcileResult("pending", reason)
            return self.store.settle(forecast, batch, candidates, reconciliation_now, self.policy)
        except sqlite3.Error:
            return ReconcileResult("failed", "database_error")
