from dataclasses import dataclass
from datetime import timedelta
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

from app.forecast_ledger import ForecastIntegrityError, ForecastStore, LedgerDatabaseError


@dataclass(frozen=True)
class EvaluationMonitoringResult:
    results: tuple
    counts: dict
    snapshot_results: tuple = ()
    consumer_publication: object | None = None


def _snapshot_evaluation(store: ForecastStore, record):
    forecast = store.get_by_id(record.forecast_id)
    return store.create_evaluation_run_snapshot(forecast.model_version, forecast.model_artifact_sha256,
        forecast.feature_schema_sha256, record.sensor_id, record.target_interval_end,
        record.target_interval_end + timedelta(hours=1), record.evaluated_at)


def materialize_available_evaluations(store: ForecastStore, limit=None, consumer_cohort=None, now=None):
    results = []
    snapshot_results = []
    for forecast_id in store.pending_evaluation_forecast_ids(limit):
        try:
            result = store.materialize_evaluation(forecast_id)
        except ForecastIntegrityError:
            results.append((forecast_id, "failed", "integrity_error"))
        except LedgerDatabaseError:
            results.append((forecast_id, "failed", "database_error"))
        else:
            results.append((forecast_id, result.status, None))
    for forecast_id in store.pending_evaluation_snapshot_forecast_ids(limit):
        try:
            snapshot = _snapshot_evaluation(store, store.get_evaluation(forecast_id))
        except ForecastIntegrityError:
            snapshot_results.append((forecast_id, "failed", "integrity_error"))
        except LedgerDatabaseError:
            snapshot_results.append((forecast_id, "failed", "database_error"))
        else:
            snapshot_results.append((forecast_id, snapshot.status, snapshot.snapshot.snapshot_id))
    for model_version, artifact, schema, sensor_id, start, end, created_at, policy_version in store.pending_evaluation_snapshot_windows(limit):
        try:
            snapshot = store.create_evaluation_run_snapshot(model_version, artifact, schema, sensor_id,
                start, end, created_at, policy_version)
        except ForecastIntegrityError:
            snapshot_results.append((None, "failed", "integrity_error"))
        except LedgerDatabaseError:
            snapshot_results.append((None, "failed", "database_error"))
        else:
            snapshot_results.append((None, snapshot.status, snapshot.snapshot.snapshot_id))
    counts = {}
    for _, status, _ in results:
        counts[status] = counts.get(status, 0) + 1
    publication = None
    if consumer_cohort is not None:
        reference = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        ict = ZoneInfo("Asia/Ho_Chi_Minh")
        local = reference.astimezone(ict)
        if local.timetz().replace(tzinfo=None) >= time(3, 15):
            publication = store.publish_consumer_performance(
                consumer_cohort["model_version"], consumer_cohort["model_artifact_sha256"],
                consumer_cohort["feature_schema_sha256"], consumer_cohort["sensor_id"],
                reference, reference,
                consumer_cohort.get("evaluation_policy_version", 1),
                consumer_cohort.get("forecast_horizon_hours", 6))
    return EvaluationMonitoringResult(tuple(results), counts, tuple(snapshot_results), publication)
