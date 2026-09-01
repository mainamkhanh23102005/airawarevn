import math
import sqlite3
import tempfile
import threading
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from app.forecast_ledger import EvaluationRecord, ForecastIntegrityError, ForecastRecord, SQLiteForecastStore
from app.ground_truth_reconciler import AcquisitionBatch, GroundTruthReconciler


UTC = timezone.utc


class EvaluationPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.database = Path(self.directory.name) / "ledger.sqlite3"
        prediction_time = datetime(2026, 1, 1, tzinfo=UTC)
        self.forecast = ForecastRecord.create(
            sensor_id=9, prediction_time=prediction_time, predicted_pm25=20,
            persistence_prediction=18, model_version="v1", model_artifact_sha256="a" * 64,
            feature_schema_sha256="b" * 64, artifact_version=1, feature_configuration="A2",
            source_retrieved_at=prediction_time, input_data_cutoff=prediction_time,
            history_start=prediction_time - timedelta(hours=24),
            history_end=prediction_time - timedelta(hours=1), data_mode_at_issue="fresh_openaq",
            freshness_status_at_issue="fresh", source_age_minutes_at_issue=0, issued_at=prediction_time)

    def tearDown(self):
        self.directory.cleanup()

    def _settled_store(self, value=Decimal("12.5")):
        store = SQLiteForecastStore(self.database)
        store.initialize()
        store.insert(self.forecast)
        start = self.forecast.target_interval_start
        batch = AcquisitionBatch.create(sensor_id=9, requested_interval_start=start,
            requested_interval_end=start + timedelta(hours=1), retrieved_at=start + timedelta(hours=3),
            source_endpoint="/sensors/9/hours", http_status=200, raw_payload_sha256="c" * 64,
            normalized_payload_sha256="f" * 64, created_at=start + timedelta(hours=3), records=[{
                "sensor_id": 9, "event_time": start, "period_end_utc": start + timedelta(hours=1),
                "value_decimal": value, "unit": "µg/m³", "record_id": 1}])
        result = GroundTruthReconciler(store, now=lambda: start + timedelta(hours=3)).reconcile(self.forecast, batch)
        self.assertEqual(result.status, "reconciled")
        return store

    def test_v2_migration_preserves_existing_rows_and_rolls_back_on_failure(self):
        store = SQLiteForecastStore(self.database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with store._connect() as connection:
            connection.execute(store._forecast_schema())
            store._create_m2_schema(connection)
            connection.execute("PRAGMA user_version=2")
        store.insert(self.forecast)
        with sqlite3.connect(self.database) as connection:
            before = connection.execute("SELECT * FROM forecasts").fetchone()
        store.initialize()
        with sqlite3.connect(self.database) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 7)
            self.assertEqual(connection.execute("SELECT * FROM forecasts").fetchone(), before)
        failed = Path(self.directory.name) / "failed.sqlite3"
        failed_store = SQLiteForecastStore(failed)
        with failed_store._connect() as connection:
            connection.execute(failed_store._forecast_schema())
            failed_store._create_m2_schema(connection)
            connection.execute("PRAGMA user_version=2")
        from unittest.mock import patch
        with patch.object(failed_store, "_create_m3_schema", side_effect=RuntimeError("boom")):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                failed_store.initialize()
        with sqlite3.connect(failed) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 2)
            self.assertIsNone(connection.execute("SELECT name FROM sqlite_master WHERE name='evaluation_rows'").fetchone())

    def test_m3_schema_rejects_nullable_or_nonunique_evaluation_relationships(self):
        changes = (
            ("evaluation_id TEXT NOT NULL PRIMARY KEY", "evaluation_id TEXT PRIMARY KEY"),
            ("forecast_id TEXT NOT NULL UNIQUE", "forecast_id TEXT UNIQUE"),
            ("forecast_id TEXT NOT NULL UNIQUE", "forecast_id TEXT NOT NULL"),
            ("reconciliation_id TEXT NOT NULL UNIQUE", "reconciliation_id TEXT UNIQUE"),
        )
        for index, change in enumerate(changes):
            with self.subTest(change=change):
                path = Path(self.directory.name) / f"m3-schema-{index}.sqlite3"
                store = SQLiteForecastStore(path)
                store.initialize()
                with sqlite3.connect(path) as connection:
                    schema = connection.execute("SELECT sql FROM sqlite_master WHERE name='evaluation_rows'").fetchone()[0]
                    connection.execute("PRAGMA foreign_keys=OFF")
                    connection.execute("ALTER TABLE evaluation_rows RENAME TO old_evaluation_rows")
                    connection.execute(schema.replace(change[0], change[1]))
                    connection.execute("DROP TABLE old_evaluation_rows")
                with self.assertRaisesRegex(RuntimeError, "unsupported schema"):
                    store.initialize()

    def test_aggregate_candidate_uses_cohort_cursor_and_snapshot_identity_index(self):
        store = SQLiteForecastStore(self.database)
        store.initialize()
        with store._connect() as connection:
            plan = connection.execute("""EXPLAIN QUERY PLAN SELECT model_version
                FROM evaluation_cohort_cursors
                WHERE evaluation_count>=2 AND NOT EXISTS (
                    SELECT 1 FROM evaluation_run_snapshots
                    WHERE evaluation_run_snapshots.evaluation_policy_version=
                        evaluation_cohort_cursors.evaluation_policy_version
                    AND evaluation_run_snapshots.model_version=evaluation_cohort_cursors.model_version
                    AND evaluation_run_snapshots.model_artifact_sha256=
                        evaluation_cohort_cursors.model_artifact_sha256
                    AND evaluation_run_snapshots.feature_schema_sha256=
                        evaluation_cohort_cursors.feature_schema_sha256
                    AND evaluation_run_snapshots.sensor_id=evaluation_cohort_cursors.sensor_id
                    AND evaluation_run_snapshots.target_interval_end_start=strftime('%Y-%m-%dT%H:%M:%SZ',
                        datetime(evaluation_cohort_cursors.latest_target_interval_end, '-1 hour'))
                    AND evaluation_run_snapshots.target_interval_end_end=strftime('%Y-%m-%dT%H:%M:%SZ',
                        datetime(evaluation_cohort_cursors.latest_target_interval_end, '+1 hour')))""").fetchall()
        details = " ".join(row[3] for row in plan)
        self.assertNotIn("evaluation_rows", details)
        self.assertIn("evaluation_run_snapshots_identity_window_idx", details)

    def test_fresh_m3_schema_rejects_null_evaluation_id(self):
        store = SQLiteForecastStore(self.database)
        store.initialize()
        with store._connect() as connection:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute("""INSERT INTO evaluation_rows VALUES (
                    NULL, 'forecast', 'reconciliation', 9, '2026-01-01T00:00:00+00:00',
                    '2026-01-01T01:00:00+00:00', 20, 18, 12.5, 7.5, 5.5,
                    '2026-01-01T03:00:00+00:00', 1)""")

    def test_persist_evaluation_rejects_mismatched_forecast_and_reconciliation(self):
        store = self._settled_store()
        second = ForecastRecord.create(**{**self.forecast.as_dict(), "forecast_id": None, "model_version": "v2"})
        store.insert(second)
        start = second.target_interval_start
        batch = AcquisitionBatch.create(sensor_id=9, requested_interval_start=start,
            requested_interval_end=start + timedelta(hours=1), retrieved_at=start + timedelta(hours=4),
            source_endpoint="/sensors/9/hours", http_status=200, raw_payload_sha256="d" * 64,
            normalized_payload_sha256="e" * 64, created_at=start + timedelta(hours=4), records=[{
                "sensor_id": 9, "event_time": start, "period_end_utc": start + timedelta(hours=1),
                "value_decimal": Decimal("13"), "unit": "µg/m³", "record_id": 2}])
        GroundTruthReconciler(store, now=lambda: start + timedelta(hours=4)).reconcile(second, batch)
        first = store.materialize_evaluation(self.forecast.forecast_id).record
        second_reconciliation = store.get_reconciliation(second.forecast_id)
        mismatched = EvaluationRecord.create(first.forecast_id, second_reconciliation.reconciliation_id,
            first.sensor_id, first.target_interval_start, first.target_interval_end, first.predicted_pm25,
            first.persistence_prediction, first.observed_pm25, first.evaluated_at)
        with self.assertRaisesRegex(ForecastIntegrityError, "evaluation relationship conflict"):
            store.persist_evaluation(mismatched)

    def test_fresh_schema_and_materialization_are_immutable_and_idempotent(self):
        store = self._settled_store()
        with store._connect() as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 7)
            self.assertIsNotNone(connection.execute("SELECT name FROM sqlite_master WHERE name='evaluation_rows'").fetchone())
        first = store.materialize_evaluation(self.forecast.forecast_id)
        self.assertEqual((first.status, first.record.observed_pm25, first.record.model_error, first.record.persistence_error),
            ("inserted", 12.5, 7.5, 5.5))
        self.assertEqual(store.materialize_evaluation(self.forecast.forecast_id).status, "already_exists")
        with self.assertRaises(ForecastIntegrityError):
            store.persist_evaluation(replace(first.record, observed_pm25=13))
        self.assertEqual(store.get_evaluation(self.forecast.forecast_id), first.record)

    def test_backfill_rolls_back_all_rows_when_later_materialization_fails(self):
        store = self._settled_store()
        second = ForecastRecord.create(**{**self.forecast.as_dict(), "forecast_id": None, "model_version": "v2"})
        store.insert(second)
        start = second.target_interval_start
        batch = AcquisitionBatch.create(sensor_id=9, requested_interval_start=start,
            requested_interval_end=start + timedelta(hours=1), retrieved_at=start + timedelta(hours=4),
            source_endpoint="/sensors/9/hours", http_status=200, raw_payload_sha256="d" * 64,
            normalized_payload_sha256="e" * 64, created_at=start + timedelta(hours=4), records=[{
                "sensor_id": 9, "event_time": start, "period_end_utc": start + timedelta(hours=1),
                "value_decimal": Decimal("13"), "unit": "µg/m³", "record_id": 2}])
        GroundTruthReconciler(store, now=lambda: start + timedelta(hours=4)).reconcile(second, batch)
        ordered = sorted((self.forecast, second), key=lambda record: (record.target_interval_end, record.forecast_id))
        with store._connect() as connection:
            connection.execute("CREATE TRIGGER fail_later_evaluation BEFORE INSERT ON evaluation_rows "
                f"WHEN NEW.forecast_id = '{ordered[1].forecast_id}' BEGIN SELECT RAISE(ABORT, 'boom'); END")
        with self.assertRaisesRegex(ForecastIntegrityError, "evaluation identity conflict"):
            store.backfill_evaluations()
        self.assertEqual(store.count_evaluations(), 0)

    def test_backfill_order_limit_and_initial_truth_isolation(self):
        store = self._settled_store()
        second = ForecastRecord.create(**{**self.forecast.as_dict(), "forecast_id": None, "model_version": "v2"})
        store.insert(second)
        start = second.target_interval_start
        batch = AcquisitionBatch.create(sensor_id=9, requested_interval_start=start, requested_interval_end=start + timedelta(hours=1),
            retrieved_at=start + timedelta(hours=4), source_endpoint="/sensors/9/hours", http_status=200,
            raw_payload_sha256="d" * 64, normalized_payload_sha256="e" * 64, created_at=start + timedelta(hours=4), records=[{
                "sensor_id": 9, "event_time": start, "period_end_utc": start + timedelta(hours=1),
                "value_decimal": Decimal("13"), "unit": "µg/m³", "record_id": 2}])
        GroundTruthReconciler(store, now=lambda: start + timedelta(hours=4)).reconcile(second, batch)
        expected = sorted((self.forecast, second), key=lambda record: (record.target_interval_end, record.forecast_id))
        self.assertEqual([item.record.forecast_id for item in store.backfill_evaluations(limit=1)], [expected[0].forecast_id])
        self.assertEqual([item.record.forecast_id for item in store.backfill_evaluations()], [expected[1].forecast_id])
        revision = replace(batch, acquisition_id=None, retrieved_at=batch.retrieved_at + timedelta(hours=1), raw_payload_sha256="f" * 64,
            records=tuple({**row, "value_decimal": Decimal("99")} for row in batch.records))
        revision = AcquisitionBatch.create(**revision.__dict__)
        self.assertEqual(GroundTruthReconciler(store, now=lambda: start + timedelta(hours=5)).reconcile(second, revision).status, "revision_detected")
        self.assertEqual(store.get_evaluation(second.forecast_id).observed_pm25, 13.0)

    def test_persist_evaluation_rejects_direct_negative_and_nonfinite_pm25_values(self):
        store = self._settled_store()
        row = store.materialize_evaluation(self.forecast.forecast_id).record
        for field, value in (("predicted_pm25", -1), ("persistence_prediction", -1), ("observed_pm25", -1),
                             ("predicted_pm25", math.nan), ("persistence_prediction", math.inf), ("observed_pm25", -math.inf)):
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ForecastIntegrityError, "evaluation value conflict"):
                store.persist_evaluation(replace(row, **{field: value}))

    def test_evaluation_record_rejects_bad_values_and_relationship_mismatch(self):
        store = self._settled_store()
        row = store.materialize_evaluation(self.forecast.forecast_id).record
        for changes in ({"observed_pm25": -1}, {"model_error": math.inf}, {"sensor_id": 10},
                        {"target_interval_end": row.target_interval_end + timedelta(hours=1)}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, ForecastIntegrityError)):
                store.persist_evaluation(replace(row, **changes))

    def test_concurrent_materialization_creates_one_row(self):
        store = self._settled_store()
        statuses, errors = [], []
        def materialize():
            try: statuses.append(store.materialize_evaluation(self.forecast.forecast_id).status)
            except Exception as error: errors.append(error)
        threads = [threading.Thread(target=materialize) for _ in range(6)]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        self.assertFalse(errors)
        self.assertEqual(statuses.count("inserted"), 1)
        self.assertEqual(statuses.count("already_exists"), 5)


if __name__ == "__main__":
    unittest.main()
