import json
import os
from datetime import datetime, timezone
from pathlib import Path

from app.evaluation_monitor import materialize_available_evaluations
from app.forecast_ledger import (create_forecast_store, feature_schema_sha256,
                                 issue_forecast, sha256_file)
from app.main import MODEL_VERSION
from scripts.modeling.features import FORECAST_HORIZON_HOURS
from scripts.openmeteo import TARGET_SENSOR_ID
from scripts.reconcile_ground_truth import run_reconciliation


DEFAULT_MODEL_PATH = Path(".artifacts/models/airaware_v1.joblib")
DEFAULT_CURRENT_PM25_PATH = Path(".artifacts/live/current_pm25.json")
DEFAULT_MONITORING_LEASE_NAME = "production-monitoring"
DEFAULT_MONITORING_LEASE_TTL_SECONDS = 3600


class MonitoringLeaseUnavailable(RuntimeError):
    pass


def _run_monitoring_work(database, raw_directory, api_key, model_path, current_pm25_path, now, store=None):
    issue = issue_forecast(database, model_path, current_pm25_path, now=now)
    reconciliation = run_reconciliation(database, raw_directory, api_key, now=now)
    if store is None:
        store = create_forecast_store(database)
        store.initialize()
    reference = now().astimezone(timezone.utc)
    if Path(model_path).is_file():
        evaluation = materialize_available_evaluations(store, consumer_cohort={
            "model_version": MODEL_VERSION,
            "model_artifact_sha256": sha256_file(model_path),
            "feature_schema_sha256": feature_schema_sha256(),
            "sensor_id": TARGET_SENSOR_ID,
            "evaluation_policy_version": 1,
            "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
        }, now=reference)
    else:
        evaluation = materialize_available_evaluations(store)
    return issue, reconciliation, evaluation


def run_monitoring_cycle(database, raw_directory, api_key, model_path=DEFAULT_MODEL_PATH,
                         current_pm25_path=DEFAULT_CURRENT_PM25_PATH, now=lambda: datetime.now(timezone.utc),
                         lease_owner_id=None, lease_ttl_seconds=DEFAULT_MONITORING_LEASE_TTL_SECONDS):
    if lease_owner_id is None:
        return _run_monitoring_work(database, raw_directory, api_key, model_path, current_pm25_path, now)

    store = create_forecast_store(database)
    store.initialize()
    if not store.acquire_monitoring_lease(
            DEFAULT_MONITORING_LEASE_NAME, lease_owner_id, now(), lease_ttl_seconds):
        raise MonitoringLeaseUnavailable("monitoring cycle skipped: production-monitoring lease is already held")

    work_failed = False
    try:
        return _run_monitoring_work(
            database, raw_directory, api_key, model_path, current_pm25_path, now, store=store)
    except BaseException:
        work_failed = True
        raise
    finally:
        try:
            released = store.release_monitoring_lease(DEFAULT_MONITORING_LEASE_NAME, lease_owner_id)
            if not released and not work_failed:
                raise MonitoringLeaseUnavailable(
                    "monitoring cycle lost production-monitoring lease before release")
        except BaseException:
            if not work_failed:
                raise


def main():
    database = os.environ.get("AIRAWARE_FORECAST_LEDGER_PATH")
    raw_directory = os.environ.get("AIRAWARE_RECONCILIATION_RAW_DIRECTORY")
    api_key = os.environ.get("OPENAQ_API_KEY")
    if not database or not raw_directory or not api_key:
        raise SystemExit("AIRAWARE_FORECAST_LEDGER_PATH, AIRAWARE_RECONCILIATION_RAW_DIRECTORY, and OPENAQ_API_KEY are required")
    issue, reconciliation, evaluation = run_monitoring_cycle(database, raw_directory, api_key,
        os.environ.get("AIRAWARE_MODEL_PATH", DEFAULT_MODEL_PATH),
        os.environ.get("AIRAWARE_CURRENT_PM25_ARTIFACT_PATH", DEFAULT_CURRENT_PM25_PATH))
    print(json.dumps({"issue": issue.outcome, "reconciliation": reconciliation.counts,
        "evaluation": evaluation.counts, "snapshots": len(evaluation.snapshot_results)},
        sort_keys=True, separators=(",", ":")))
    return 0 if issue.outcome != "failed" and not evaluation.counts.get("failed") and not reconciliation.counts.get("failed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
