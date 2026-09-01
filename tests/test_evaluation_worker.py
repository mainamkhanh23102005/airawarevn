import asyncio
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from app import main
from app.evaluation_monitor import materialize_available_evaluations
from app.forecast_ledger import ForecastRecord, SQLiteForecastStore
from app.ground_truth_reconciler import AcquisitionBatch, GroundTruthReconciler


UTC = timezone.utc


class EvaluationWorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = SQLiteForecastStore(Path(self.directory.name) / "ledger.sqlite3")
        self.store.initialize()
        prediction_time = datetime(2026, 1, 1, tzinfo=UTC)
        self.forecast = ForecastRecord.create(sensor_id=9, prediction_time=prediction_time,
            predicted_pm25=20, persistence_prediction=18, model_version="v1",
            model_artifact_sha256="a" * 64, feature_schema_sha256="b" * 64,
            artifact_version=1, feature_configuration="A2", source_retrieved_at=prediction_time,
            input_data_cutoff=prediction_time, history_start=prediction_time - timedelta(hours=24),
            history_end=prediction_time - timedelta(hours=1), data_mode_at_issue="fresh_openaq",
            freshness_status_at_issue="fresh", source_age_minutes_at_issue=0, issued_at=prediction_time)
        self.store.insert(self.forecast)

    def tearDown(self):
        self.directory.cleanup()

    def _settle(self):
        batch = AcquisitionBatch.create(sensor_id=9, requested_interval_start=self.forecast.target_interval_start,
            requested_interval_end=self.forecast.target_interval_end,
            retrieved_at=self.forecast.target_interval_end + timedelta(hours=2),
            source_endpoint="/sensors/9/hours", http_status=200, raw_payload_sha256="c" * 64,
            normalized_payload_sha256="d" * 64, created_at=self.forecast.target_interval_end + timedelta(hours=2),
            records=[{"sensor_id": 9, "event_time": self.forecast.target_interval_start,
                "period_end_utc": self.forecast.target_interval_end, "value_decimal": Decimal("12.5"),
                "unit": "µg/m³", "record_id": 1}])
        GroundTruthReconciler(self.store, now=lambda: self.forecast.target_interval_end + timedelta(hours=2)).reconcile(self.forecast, batch)

    def test_worker_materializes_only_initial_mature_truth_idempotently(self):
        self.assertEqual(materialize_available_evaluations(self.store).counts, {})
        self._settle()
        first = materialize_available_evaluations(self.store)
        repeated = materialize_available_evaluations(self.store)
        self.assertEqual(first.counts, {"inserted": 1})
        self.assertEqual(repeated.counts, {})
        self.assertEqual(self.store.count_evaluations(), 1)

    def test_worker_isolates_one_recoverable_materialization_failure(self):
        self._settle()
        with patch.object(self.store, "materialize_evaluation", side_effect=sqlite3.OperationalError("busy")):
            result = materialize_available_evaluations(self.store)
        self.assertEqual(result.results, ((self.forecast.forecast_id, "failed", "database_error"),))
        self.assertEqual(self.store.count_evaluations(), 0)

    def test_worker_creates_idempotent_single_window_snapshot_for_traffic_independent_reporting(self):
        self._settle()
        first = materialize_available_evaluations(self.store)
        repeated = materialize_available_evaluations(self.store)
        snapshot = self.store.find_evaluation_run_snapshot("v1", "a" * 64, "b" * 64, 9,
            self.forecast.target_interval_end, self.forecast.target_interval_end + timedelta(hours=1))
        self.assertEqual(first.snapshot_results, ((self.forecast.forecast_id, "inserted", snapshot.snapshot_id),))
        self.assertEqual(repeated.snapshot_results, ())
        self.assertEqual(snapshot.evaluation_ids, (self.store.get_evaluation(self.forecast.forecast_id).evaluation_id,))

    def test_worker_creates_current_compatible_multi_hour_snapshot_without_reporting_traffic(self):
        self._settle()
        materialize_available_evaluations(self.store)
        later = ForecastRecord.create(sensor_id=9, prediction_time=self.forecast.prediction_time + timedelta(hours=1),
            predicted_pm25=22, persistence_prediction=20, model_version="v1", model_artifact_sha256="a" * 64,
            feature_schema_sha256="b" * 64, artifact_version=1, feature_configuration="A2",
            source_retrieved_at=self.forecast.prediction_time + timedelta(hours=1),
            input_data_cutoff=self.forecast.prediction_time + timedelta(hours=1),
            history_start=self.forecast.prediction_time - timedelta(hours=23),
            history_end=self.forecast.prediction_time, data_mode_at_issue="fresh_openaq",
            freshness_status_at_issue="fresh", source_age_minutes_at_issue=0,
            issued_at=self.forecast.prediction_time + timedelta(hours=1))
        self.store.insert(later)
        batch = AcquisitionBatch.create(sensor_id=9, requested_interval_start=later.target_interval_start,
            requested_interval_end=later.target_interval_end, retrieved_at=later.target_interval_end + timedelta(hours=2),
            source_endpoint="/sensors/9/hours", http_status=200, raw_payload_sha256="e" * 64,
            normalized_payload_sha256="f" * 64, created_at=later.target_interval_end + timedelta(hours=2), records=[{
                "sensor_id": 9, "event_time": later.target_interval_start, "period_end_utc": later.target_interval_end,
                "value_decimal": Decimal("14"), "unit": "µg/m³", "record_id": 2}])
        GroundTruthReconciler(self.store, now=lambda: later.target_interval_end + timedelta(hours=2)).reconcile(later, batch)
        materialize_available_evaluations(self.store)
        snapshot = self.store.find_evaluation_run_snapshot("v1", "a" * 64, "b" * 64, 9,
            later.target_interval_end - timedelta(hours=1), later.target_interval_end + timedelta(hours=1))
        self.assertEqual(snapshot.metrics.count, 2)
        self.assertEqual(len(snapshot.evaluation_ids), 2)
        self.assertEqual(materialize_available_evaluations(self.store).snapshot_results, ())

    def test_worker_retries_missing_aggregate_snapshot_without_new_evaluations(self):
        self._settle()
        materialize_available_evaluations(self.store)
        later = ForecastRecord.create(sensor_id=9, prediction_time=self.forecast.prediction_time + timedelta(hours=1),
            predicted_pm25=22, persistence_prediction=20, model_version="v1", model_artifact_sha256="a" * 64,
            feature_schema_sha256="b" * 64, artifact_version=1, feature_configuration="A2",
            source_retrieved_at=self.forecast.prediction_time + timedelta(hours=1),
            input_data_cutoff=self.forecast.prediction_time + timedelta(hours=1),
            history_start=self.forecast.prediction_time - timedelta(hours=23), history_end=self.forecast.prediction_time,
            data_mode_at_issue="fresh_openaq", freshness_status_at_issue="fresh", source_age_minutes_at_issue=0,
            issued_at=self.forecast.prediction_time + timedelta(hours=1))
        self.store.insert(later)
        batch = AcquisitionBatch.create(sensor_id=9, requested_interval_start=later.target_interval_start,
            requested_interval_end=later.target_interval_end, retrieved_at=later.target_interval_end + timedelta(hours=2),
            source_endpoint="/sensors/9/hours", http_status=200, raw_payload_sha256="e" * 64,
            normalized_payload_sha256="f" * 64, created_at=later.target_interval_end + timedelta(hours=2), records=[{
                "sensor_id": 9, "event_time": later.target_interval_start, "period_end_utc": later.target_interval_end,
                "value_decimal": Decimal("14"), "unit": "µg/m³", "record_id": 2}])
        GroundTruthReconciler(self.store, now=lambda: later.target_interval_end + timedelta(hours=2)).reconcile(later, batch)
        self.store.materialize_evaluation(later.forecast_id)
        aggregate = self.store.pending_evaluation_snapshot_windows()[0]
        with patch.object(self.store, "pending_evaluation_snapshot_forecast_ids", return_value=[]), patch.object(
                self.store, "pending_evaluation_snapshot_windows", return_value=[aggregate]), patch.object(
                self.store, "create_evaluation_run_snapshot", side_effect=sqlite3.OperationalError("busy")):
            failed = materialize_available_evaluations(self.store)
        self.assertEqual(failed.counts, {})
        self.assertEqual(failed.snapshot_results, ((None, "failed", "database_error"),))
        retried = materialize_available_evaluations(self.store)
        self.assertEqual(retried.counts, {})
        snapshot = self.store.find_evaluation_run_snapshot("v1", "a" * 64, "b" * 64, 9,
            later.target_interval_end - timedelta(hours=1), later.target_interval_end + timedelta(hours=1))
        self.assertEqual(snapshot.metrics.count, 2)


if __name__ == "__main__":
    unittest.main()
