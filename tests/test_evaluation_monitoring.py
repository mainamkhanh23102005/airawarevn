import tempfile
import unittest
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from app.evaluation_monitor import materialize_available_evaluations
from app.forecast_ledger import ForecastRecord, SQLiteForecastStore
from app.ground_truth_reconciler import AcquisitionBatch, GroundTruthReconciler


UTC = timezone.utc
ICT = ZoneInfo("Asia/Ho_Chi_Minh")
ARTIFACT = "a" * 64
SCHEMA = "b" * 64


class EvaluationMonitoringTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = SQLiteForecastStore(Path(self.directory.name) / "ledger.sqlite3")
        self.store.initialize()
        self.start = datetime(2026, 1, 1, 6, tzinfo=UTC)

    def tearDown(self):
        self.directory.cleanup()

    def _cohort(self, model_version="v1", sensor_id=9):
        return {"model_version": model_version, "model_artifact_sha256": ARTIFACT,
                "feature_schema_sha256": SCHEMA, "sensor_id": sensor_id,
                "evaluation_policy_version": 1, "forecast_horizon_hours": 6}

    def _evaluate(self, prediction_time, predicted, persistence, observed, model_version="v1", artifact=ARTIFACT, schema=SCHEMA, sensor_id=9):
        forecast = ForecastRecord.create(sensor_id=sensor_id, prediction_time=prediction_time,
            predicted_pm25=predicted, persistence_prediction=persistence, model_version=model_version,
            model_artifact_sha256=artifact, feature_schema_sha256=schema, artifact_version=1,
            feature_configuration="A2", source_retrieved_at=prediction_time, input_data_cutoff=prediction_time,
            history_start=prediction_time - timedelta(hours=24), history_end=prediction_time - timedelta(hours=1),
            data_mode_at_issue="fresh_openaq", freshness_status_at_issue="fresh", source_age_minutes_at_issue=0,
            issued_at=prediction_time)
        self.store.insert(forecast)
        batch = AcquisitionBatch.create(sensor_id=sensor_id, requested_interval_start=forecast.target_interval_start,
            requested_interval_end=forecast.target_interval_end, retrieved_at=forecast.target_interval_end + timedelta(hours=2),
            source_endpoint="/sensors/9/hours", http_status=200, raw_payload_sha256=(str(sensor_id) * 64)[:64],
            normalized_payload_sha256="f" * 64, created_at=forecast.target_interval_end + timedelta(hours=2), records=[{
                "sensor_id": sensor_id, "event_time": forecast.target_interval_start,
                "period_end_utc": forecast.target_interval_end, "value_decimal": Decimal(str(observed)),
                "unit": "µg/m³", "record_id": sensor_id}])
        result = GroundTruthReconciler(self.store, now=lambda: forecast.target_interval_end + timedelta(hours=2)).reconcile(forecast, batch)
        self.assertEqual(result.status, "reconciled")
        return self.store.materialize_evaluation(forecast.forecast_id).record

    def test_metrics_use_exact_filtered_deterministic_immutable_rows(self):
        first = self._evaluate(self.start, 12, 14, 10)
        second = self._evaluate(self.start + timedelta(hours=1), 8, 7, 10)
        self._evaluate(self.start + timedelta(hours=2), 1, 1, 1, model_version="v2")
        metrics = self.store.evaluation_metrics("v1", ARTIFACT, SCHEMA, 9, first.target_interval_end, second.target_interval_end + timedelta(hours=1))
        self.assertEqual(metrics.count, 2)
        self.assertEqual(metrics.model_mae, 2)
        self.assertEqual(metrics.persistence_mae, 3.5)
        self.assertAlmostEqual(metrics.model_rmse, 2)
        self.assertAlmostEqual(metrics.persistence_rmse, (25 / 2) ** 0.5)
        self.assertEqual(metrics.mae_improvement, 1.5)
        self.assertAlmostEqual(metrics.mae_improvement_percent, 1.5 / 3.5 * 100)

    def test_empty_and_zero_baseline_metrics_have_required_nulls(self):
        empty = self.store.evaluation_metrics("v1", ARTIFACT, SCHEMA, 9, self.start, self.start + timedelta(hours=1))
        self.assertEqual(empty.count, 0)
        self.assertTrue(all(value is None for value in empty.__dict__.values() if value != 0))
        row = self._evaluate(self.start, 1, 10, 10)
        metrics = self.store.evaluation_metrics("v1", ARTIFACT, SCHEMA, 9, row.target_interval_end, row.target_interval_end + timedelta(hours=1))
        self.assertIsNone(metrics.mae_improvement_percent)
        self.assertIsNone(metrics.rmse_improvement_percent)

    def test_snapshot_is_idempotent_and_independent_of_later_rows(self):
        first = self._evaluate(self.start, 12, 14, 10)
        end = first.target_interval_end + timedelta(hours=1)
        created = datetime(2026, 2, 1, tzinfo=UTC)
        initial = self.store.create_evaluation_run_snapshot("v1", ARTIFACT, SCHEMA, 9, first.target_interval_end, end, created)
        repeated = self.store.create_evaluation_run_snapshot("v1", ARTIFACT, SCHEMA, 9, first.target_interval_end, end, created + timedelta(days=1))
        self.assertEqual((initial.status, repeated.status), ("inserted", "already_exists"))
        self.assertEqual(initial.snapshot, repeated.snapshot)
        self._evaluate(self.start + timedelta(hours=1), 1, 1, 1)
        self.assertEqual(initial.snapshot.evaluation_ids, (first.evaluation_id,))
        self.assertEqual(initial.snapshot.metrics.count, 1)

    def test_rejects_invalid_or_mixed_cohort_filters(self):
        row = self._evaluate(self.start, 12, 14, 10)
        with self.assertRaisesRegex(ValueError, "window"):
            self.store.evaluation_metrics("v1", ARTIFACT, SCHEMA, 9, row.target_interval_end, row.target_interval_end)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.store.evaluation_metrics("v1", "A" * 64, SCHEMA, 9, row.target_interval_end, row.target_interval_end + timedelta(hours=1))

    def test_empty_snapshot_lookup_ignores_incompatible_rows_outside_requested_window(self):
        row = self._evaluate(self.start, 12, 14, 10, model_version="v2")
        snapshot = self.store.find_evaluation_run_snapshot("v1", ARTIFACT, SCHEMA, 9,
            row.target_interval_end + timedelta(hours=1), row.target_interval_end + timedelta(hours=2))
        self.assertIsNone(snapshot)

    def test_snapshot_lookup_rejects_in_window_incompatible_cohort_even_when_snapshot_exists(self):
        row = self._evaluate(self.start, 12, 14, 10)
        self.store.create_evaluation_run_snapshot("v1", ARTIFACT, SCHEMA, 9,
            row.target_interval_end, row.target_interval_end + timedelta(hours=1), self.start)
        self._evaluate(self.start, 12, 14, 10, model_version="v2", artifact="c" * 64, schema="d" * 64)
        with self.assertRaisesRegex(ValueError, "ambiguous evaluation cohort"):
            self.store.find_evaluation_run_snapshot("v1", ARTIFACT, SCHEMA, 9,
                row.target_interval_end, row.target_interval_end + timedelta(hours=1))

    def test_empty_snapshot_lookup_rejects_in_window_cohort_with_multiple_different_identity_fields(self):
        row = self._evaluate(self.start, 12, 14, 10, model_version="v2", artifact="c" * 64, schema="d" * 64)
        with self.assertRaisesRegex(ValueError, "ambiguous evaluation cohort"):
            self.store.find_evaluation_run_snapshot("v1", ARTIFACT, SCHEMA, 9,
                row.target_interval_end, row.target_interval_end + timedelta(hours=1))

    def test_tampered_durable_evaluation_values_fail_closed(self):
        row = self._evaluate(self.start, 12, 14, 10)
        for column, value in (("predicted_pm25", -1), ("model_error", 99)):
            with self.subTest(column=column), self.store._connect() as connection:
                connection.execute(f"UPDATE evaluation_rows SET {column}=? WHERE evaluation_id=?", (value, row.evaluation_id))
                connection.commit()
            with self.assertRaisesRegex(Exception, "invalid durable evaluation row"):
                self.store.evaluation_metrics("v1", ARTIFACT, SCHEMA, 9, row.target_interval_end, row.target_interval_end + timedelta(hours=1))
            with self.store._connect() as connection:
                connection.execute(f"UPDATE evaluation_rows SET {column}=? WHERE evaluation_id=?", (12 if column == "predicted_pm25" else 2, row.evaluation_id))
                connection.commit()

    def test_tampered_snapshot_provenance_and_metrics_fail_closed(self):
        row = self._evaluate(self.start, 12, 14, 10)
        self.store.create_evaluation_run_snapshot("v1", ARTIFACT, SCHEMA, 9, row.target_interval_end,
            row.target_interval_end + timedelta(hours=1), self.start)
        for column, value in (("evaluation_ids_json", '[]'), ("model_mae", 99)):
            with self.subTest(column=column), self.store._connect() as connection:
                connection.execute(f"UPDATE evaluation_run_snapshots SET {column}=?", (value,))
                connection.commit()
            with self.assertRaisesRegex(Exception, "invalid durable snapshot"):
                self.store.find_evaluation_run_snapshot("v1", ARTIFACT, SCHEMA, 9, row.target_interval_end,
                    row.target_interval_end + timedelta(hours=1))
            with self.store._connect() as connection:
                if column == "evaluation_ids_json":
                    connection.execute("UPDATE evaluation_run_snapshots SET evaluation_ids_json=?", (f'["{row.evaluation_id}"]',))
                else:
                    connection.execute("UPDATE evaluation_run_snapshots SET model_mae=?", (2,))
                connection.commit()

    def test_late_initial_evaluation_replaces_canonical_aggregate_snapshot_for_all_positions(self):
        for initial_offsets, late_offset in (((1, 2), 0), ((0, 2), 1), ((0, 1), 2)):
            with self.subTest(late_offset=late_offset):
                directory = tempfile.TemporaryDirectory()
                store = SQLiteForecastStore(Path(directory.name) / "ledger.sqlite3")
                store.initialize()
                records = {}
                for offset in initial_offsets:
                    original_store, self.store = self.store, store
                    records[offset] = self._evaluate(self.start + timedelta(hours=offset), 12, 14, 10)
                    self.store = original_store
                initial_window = store.pending_evaluation_snapshot_windows()[0]
                initial = store.create_evaluation_run_snapshot(*initial_window[:6], initial_window[6], initial_window[7]).snapshot
                original_store, self.store = self.store, store
                records[late_offset] = self._evaluate(self.start + timedelta(hours=late_offset), 12, 14, 10)
                self.store = original_store
                replacement_window = store.pending_evaluation_snapshot_windows()[0]
                replacement = store.create_evaluation_run_snapshot(*replacement_window[:6], replacement_window[6], replacement_window[7]).snapshot
                start = min(record.target_interval_end for record in records.values())
                end = max(record.target_interval_end for record in records.values()) + timedelta(hours=1)
                canonical = store.find_evaluation_run_snapshot("v1", ARTIFACT, SCHEMA, 9, start, end)
                self.assertEqual(replacement.evaluation_ids, tuple(records[offset].evaluation_id for offset in sorted(records)))
                self.assertEqual(canonical, replacement)
                self.assertNotEqual(initial.snapshot_id, replacement.snapshot_id)
                self.assertEqual(store.pending_evaluation_snapshot_windows(), [])
                revised = ForecastRecord.create(sensor_id=9, prediction_time=self.start + timedelta(hours=3),
                    predicted_pm25=12, persistence_prediction=14, model_version="v1", model_artifact_sha256=ARTIFACT,
                    feature_schema_sha256=SCHEMA, artifact_version=1, feature_configuration="A2",
                    source_retrieved_at=self.start + timedelta(hours=3), input_data_cutoff=self.start + timedelta(hours=3),
                    history_start=self.start - timedelta(hours=21), history_end=self.start + timedelta(hours=2),
                    data_mode_at_issue="fresh_openaq", freshness_status_at_issue="fresh", source_age_minutes_at_issue=0,
                    issued_at=self.start + timedelta(hours=3))
                store.insert(revised)
                batch = AcquisitionBatch.create(sensor_id=9, requested_interval_start=revised.target_interval_start,
                    requested_interval_end=revised.target_interval_end, retrieved_at=revised.target_interval_end + timedelta(hours=2),
                    source_endpoint="/sensors/9/hours", http_status=200, raw_payload_sha256="c" * 64,
                    normalized_payload_sha256="f" * 64, created_at=revised.target_interval_end + timedelta(hours=2), records=[{
                    "sensor_id": 9, "event_time": revised.target_interval_start, "period_end_utc": revised.target_interval_end,
                    "value_decimal": Decimal("10"), "unit": "µg/m³", "record_id": 99}])
                GroundTruthReconciler(store, now=lambda: revised.target_interval_end + timedelta(hours=2)).reconcile(revised, batch)
                revision = AcquisitionBatch.create(sensor_id=9, requested_interval_start=revised.target_interval_start,
                    requested_interval_end=revised.target_interval_end, retrieved_at=revised.target_interval_end + timedelta(hours=3),
                    source_endpoint="/sensors/9/hours", http_status=200, raw_payload_sha256="d" * 64,
                    normalized_payload_sha256="e" * 64, created_at=revised.target_interval_end + timedelta(hours=3), records=[{
                    "sensor_id": 9, "event_time": revised.target_interval_start, "period_end_utc": revised.target_interval_end,
                    "value_decimal": Decimal("99"), "unit": "µg/m³", "record_id": 100}])
                self.assertEqual(GroundTruthReconciler(store, now=lambda: revised.target_interval_end + timedelta(hours=3)).reconcile(revised, revision).status, "revision_detected")
                self.assertEqual(store.pending_evaluation_snapshot_windows(), [])
                self.assertEqual(store.find_evaluation_run_snapshot("v1", ARTIFACT, SCHEMA, 9, start, end), replacement)
                with store._connect() as connection:
                    self.assertEqual(connection.execute("SELECT COUNT(*) FROM evaluation_run_snapshots").fetchone()[0], 2)
                directory.cleanup()

    def test_snapshot_lookup_returns_current_membership_after_later_evaluation(self):
        first = self._evaluate(self.start, 12, 14, 10)
        end = first.target_interval_end + timedelta(hours=2)
        initial = self.store.create_evaluation_run_snapshot("v1", ARTIFACT, SCHEMA, 9, first.target_interval_end,
            end, self.start).snapshot
        second = self._evaluate(self.start + timedelta(hours=1), 8, 7, 10)
        replacement = self.store.create_evaluation_run_snapshot("v1", ARTIFACT, SCHEMA, 9, first.target_interval_end,
            end, self.start + timedelta(hours=1)).snapshot
        snapshot = self.store.find_evaluation_run_snapshot("v1", ARTIFACT, SCHEMA, 9, first.target_interval_end, end)
        self.assertNotEqual(initial.snapshot_id, replacement.snapshot_id)
        self.assertEqual(snapshot, replacement)
        self.assertEqual(snapshot.evaluation_ids, (first.evaluation_id, second.evaluation_id))

    def test_snapshot_lookup_rejects_malformed_candidate_among_valid_snapshots(self):
        row = self._evaluate(self.start, 12, 14, 10)
        self.store.create_evaluation_run_snapshot("v1", ARTIFACT, SCHEMA, 9, row.target_interval_end,
            row.target_interval_end + timedelta(hours=1), self.start)
        with self.store._connect() as connection:
            connection.execute("""INSERT INTO evaluation_run_snapshots
                SELECT 'tampered-duplicate', evaluation_policy_version, model_version, model_artifact_sha256,
                feature_schema_sha256, sensor_id, target_interval_end_start, target_interval_end_end,
                evaluation_ids_json, count, model_mae, model_rmse, persistence_mae, persistence_rmse,
                mae_improvement, rmse_improvement, mae_improvement_percent, rmse_improvement_percent, created_at
                FROM evaluation_run_snapshots""")
            connection.commit()
        with self.assertRaisesRegex(Exception, "invalid durable snapshot"):
            self.store.find_evaluation_run_snapshot("v1", ARTIFACT, SCHEMA, 9, row.target_interval_end,
                row.target_interval_end + timedelta(hours=1))

    def test_materialize_gate_suppresses_publication_before_0315(self):
        result = materialize_available_evaluations(self.store, consumer_cohort=self._cohort(),
            now=datetime(2026, 1, 2, 3, 14, 59, tzinfo=ICT))
        self.assertIsNone(result.consumer_publication)

    def test_materialize_gate_publishes_at_exact_0315(self):
        result = materialize_available_evaluations(self.store, consumer_cohort=self._cohort(),
            now=datetime(2026, 1, 2, 3, 15, 0, tzinfo=ICT))
        self.assertIsNotNone(result.consumer_publication)

    def test_materialize_gate_publishes_after_0315(self):
        result = materialize_available_evaluations(self.store, consumer_cohort=self._cohort(),
            now=datetime(2026, 1, 2, 3, 15, 1, tzinfo=ICT))
        self.assertIsNotNone(result.consumer_publication)

    def test_materialize_publication_date_uses_ict_date_not_utc_date(self):
        result = materialize_available_evaluations(self.store, consumer_cohort=self._cohort(),
            now=datetime(2026, 1, 1, 20, 30, 0, tzinfo=UTC))
        publication = result.consumer_publication
        self.assertIsNotNone(publication)
        self.assertEqual(publication.publication_date, "2026-01-02")
        self.assertNotEqual(publication.publication_date, "2026-01-01")


if __name__ == "__main__":
    unittest.main()
