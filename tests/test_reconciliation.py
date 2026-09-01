import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, localcontext
from pathlib import Path
from unittest.mock import patch

from app.forecast_ledger import ForecastIntegrityError, ForecastRecord, SQLiteForecastStore
from app.ground_truth_reconciler import (
    ACQUISITION_NAMESPACE,
    RECONCILIATION_NAMESPACE,
    REVISION_NAMESPACE,
    AcquisitionBatch,
    GroundTruthReconciler,
    ReconcileResult,
    TruthPolicy,
    canonical_decimal,
    canonical_normalized_json,
    normalized_sha256,
    SourceParseError,
    SourceTransportError,
)

UTC = timezone.utc


class ReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.database = Path(self.directory.name) / "ledger.sqlite3"
        self.prediction_time = datetime(2026, 1, 1, tzinfo=UTC)
        self.forecast = ForecastRecord.create(
            sensor_id=9, prediction_time=self.prediction_time, predicted_pm25=20,
            persistence_prediction=18, model_version="v1", model_artifact_sha256="a" * 64,
            feature_schema_sha256="b" * 64, artifact_version=1, feature_configuration="A2",
            source_retrieved_at=self.prediction_time, input_data_cutoff=self.prediction_time,
            history_start=self.prediction_time - timedelta(hours=24),
            history_end=self.prediction_time - timedelta(hours=1), data_mode_at_issue="fresh_openaq",
            freshness_status_at_issue="fresh", source_age_minutes_at_issue=0,
            issued_at=self.prediction_time,
        )

    def tearDown(self):
        self.directory.cleanup()

    def test_fresh_database_has_exact_v2_tables_and_foreign_keys(self):
        store = SQLiteForecastStore(self.database)
        store.initialize()
        with store._connect() as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 7)
            self.assertEqual(connection.execute("PRAGMA foreign_keys").fetchone()[0], 1)
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertTrue({"forecasts", "observation_acquisitions", "forecast_reconciliations", "observation_revisions"} <= tables)

    def _create_v1(self):
        store = SQLiteForecastStore(self.database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with store._connect() as connection:
            connection.execute(store._forecast_schema())
            connection.execute("PRAGMA user_version=1")
        return store

    def test_v1_migration_preserves_forecast_schema_and_values(self):
        store = self._create_v1()
        store.insert(self.forecast)
        with sqlite3.connect(self.database) as connection:
            before_sql = connection.execute("SELECT sql FROM sqlite_master WHERE name='forecasts'").fetchone()[0]
            before_row = connection.execute("SELECT * FROM forecasts").fetchone()
        store.initialize()
        with sqlite3.connect(self.database) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 7)
            self.assertEqual(connection.execute("SELECT sql FROM sqlite_master WHERE name='forecasts'").fetchone()[0], before_sql)
            self.assertEqual(connection.execute("SELECT * FROM forecasts").fetchone(), before_row)

    def test_malformed_v2_columns_unique_constraints_and_foreign_keys_fail_closed(self):
        corruptions = [
            "ALTER TABLE observation_acquisitions DROP COLUMN created_at",
            "DROP INDEX sqlite_autoindex_forecast_reconciliations_2",
            "PRAGMA foreign_keys=OFF",
        ]
        for index, statement in enumerate(corruptions):
            path = Path(self.directory.name) / f"schema-{index}.sqlite3"
            store = SQLiteForecastStore(path); store.initialize()
            with sqlite3.connect(path) as connection:
                if index == 0:
                    connection.execute(statement)
                elif index == 1:
                    connection.execute("ALTER TABLE forecast_reconciliations RENAME TO old_reconciliations")
                    connection.execute("CREATE TABLE forecast_reconciliations AS SELECT * FROM old_reconciliations")
                    connection.execute("DROP TABLE old_reconciliations")
                else:
                    connection.execute("ALTER TABLE observation_revisions RENAME TO old_revisions")
                    connection.execute("CREATE TABLE observation_revisions AS SELECT * FROM old_revisions")
                    connection.execute("DROP TABLE old_revisions")
            with self.assertRaisesRegex(RuntimeError, "unsupported schema"):
                store.initialize()

    def test_v2_reopen_rejects_wrong_primary_key_identity(self):
        replacements = {
            "observation_acquisitions": ("acquisition_id TEXT PRIMARY KEY", "acquisition_id TEXT UNIQUE", "sensor_id INTEGER NOT NULL", "sensor_id INTEGER NOT NULL PRIMARY KEY"),
            "forecast_reconciliations": ("reconciliation_id TEXT PRIMARY KEY", "reconciliation_id TEXT UNIQUE", "sensor_id INTEGER NOT NULL", "sensor_id INTEGER NOT NULL PRIMARY KEY"),
            "observation_revisions": ("revision_id TEXT PRIMARY KEY", "revision_id TEXT UNIQUE", "unit TEXT NOT NULL", "unit TEXT NOT NULL PRIMARY KEY"),
        }
        for index, (table, changes) in enumerate(replacements.items()):
            path = Path(self.directory.name) / f"pk-{index}.sqlite3"
            store = SQLiteForecastStore(path); store.initialize()
            with sqlite3.connect(path) as connection:
                sql = connection.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()[0]
                connection.execute("PRAGMA foreign_keys=OFF")
                connection.execute(f"ALTER TABLE {table} RENAME TO old_{table}")
                connection.execute(sql.replace(f"CREATE TABLE {table}", f"CREATE TABLE {table}").replace(changes[0], changes[1]).replace(changes[2], changes[3]))
                connection.execute(f"DROP TABLE old_{table}")
            with self.assertRaisesRegex(RuntimeError, "unsupported schema"):
                store.initialize()

    def test_partial_v2_fails_closed(self):
        store = SQLiteForecastStore(self.database); store.initialize()
        with sqlite3.connect(self.database) as connection:
            connection.execute("DROP TABLE observation_revisions")
        with self.assertRaisesRegex(RuntimeError, "unsupported schema"):
            store.initialize()

    def test_partial_v0_and_future_versions_fail_without_mutation(self):
        with sqlite3.connect(self.database) as connection:
            connection.execute("CREATE TABLE stray(value TEXT)")
        with self.assertRaisesRegex(RuntimeError, "unsupported schema"):
            SQLiteForecastStore(self.database).initialize()
        with sqlite3.connect(self.database) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 0)
            connection.execute("PRAGMA user_version=8")
        with self.assertRaisesRegex(RuntimeError, "unsupported schema version"):
            SQLiteForecastStore(self.database).initialize()
        with sqlite3.connect(self.database) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 8)

    def test_v6_cursor_migration_rebuilds_legacy_incomplete_backfill(self):
        store = SQLiteForecastStore(self.database)
        store.initialize()
        with sqlite3.connect(self.database) as connection:
            connection.execute("DELETE FROM evaluation_cohort_cursors")
            connection.execute("PRAGMA user_version=6")
        store.initialize()
        with sqlite3.connect(self.database) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 7)

    def test_v5_migration_rejects_missing_current_window_index(self):
        store = SQLiteForecastStore(self.database)
        store.initialize()
        with sqlite3.connect(self.database) as connection:
            connection.execute("DROP INDEX evaluation_rows_current_window_idx")
            connection.execute("DROP TABLE evaluation_cohort_cursors")
            connection.execute("PRAGMA user_version=5")
        with self.assertRaisesRegex(RuntimeError, "unsupported schema"):
            store.initialize()
        with sqlite3.connect(self.database) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 5)
            self.assertIsNone(connection.execute("SELECT name FROM sqlite_master WHERE name='evaluation_cohort_cursors'").fetchone())

    def test_failed_migration_rolls_back_tables_and_version(self):
        store = self._create_v1()
        with patch.object(store, "_create_m2_schema", side_effect=RuntimeError("boom")):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                store.initialize()
        with sqlite3.connect(self.database) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)
            self.assertIsNone(connection.execute("SELECT name FROM sqlite_master WHERE name='observation_acquisitions'").fetchone())

    def test_decimal_and_hash_fixtures_are_exact(self):
        fixtures = [("1", "1"), ("1.2300", "1.23"), ("1e-3", "0.001"), ("0", "0")]
        for source, expected in fixtures:
            self.assertEqual(canonical_decimal(Decimal(source), "µg/m³"), expected)
        self.assertEqual(canonical_decimal(Decimal("0.0012300"), "mg/m³"), "1.23")
        for bad in (Decimal("-0"), Decimal("-1"), Decimal("NaN"), Decimal("Infinity")):
            with self.assertRaises(ValueError):
                canonical_decimal(bad, "µg/m³")
        records = [{"sensor_id": 9, "event_time": datetime(2026, 1, 1, tzinfo=UTC),
                    "period_end_utc": datetime(2026, 1, 1, 1, tzinfo=UTC), "value_decimal": Decimal("1.2300"),
                    "unit": "µg/m³", "record_id": 7}]
        canonical = canonical_normalized_json(records)
        expected = '{"normalized_serialization_version":1,"records":[{"event_time":"2026-01-01T00:00:00Z","period_end_utc":"2026-01-01T01:00:00Z","pm25_ug_m3":"1.23","record_id":"7","sensor_id":9,"unit":"µg/m³"}]}'
        self.assertEqual(canonical, expected)
        self.assertEqual(normalized_sha256(records), "38259658d124f21e825b400951ff23999b3d3e622617b916843733bc23f82087")

    def test_mg_conversion_and_hash_ignore_decimal_context_precision(self):
        records = [{"sensor_id": 9, "event_time": datetime(2026, 1, 1, tzinfo=UTC),
            "period_end_utc": datetime(2026, 1, 1, 1, tzinfo=UTC),
            "value_decimal": Decimal("0.12345678901234567890123456789"), "unit": "mg/m³", "record_id": 1}]
        expected_json = '{"normalized_serialization_version":1,"records":[{"event_time":"2026-01-01T00:00:00Z","period_end_utc":"2026-01-01T01:00:00Z","pm25_ug_m3":"123.45678901234567890123456789","record_id":"1","sensor_id":9,"unit":"µg/m³"}]}'
        with localcontext() as context:
            context.prec = 6
            self.assertEqual(canonical_normalized_json(records), expected_json)
            self.assertEqual(normalized_sha256(records), "4264280ca1afb44b93e8306f2a2c57be48eb6eded2ac1731fdb5ad695fa8ab60")

    def _batch(self, value=Decimal("12.50"), retrieved_at=None, records=None):
        start = self.forecast.target_interval_start
        rows = records if records is not None else [{"sensor_id": 9, "event_time": start,
            "period_end_utc": start + timedelta(hours=1), "value_decimal": value,
            "unit": "µg/m³", "record_id": 1}]
        return AcquisitionBatch.create(sensor_id=9, requested_interval_start=start,
            requested_interval_end=start + timedelta(hours=1), retrieved_at=retrieved_at or start + timedelta(hours=3),
            source_endpoint="/sensors/9/hours", http_status=200, raw_payload_sha256="c" * 64, records=rows,
            normalized_payload_sha256="f" * 64, created_at=start + timedelta(hours=3))

    def _reconciler(self, now=None):
        store = SQLiteForecastStore(self.database); store.initialize(); store.insert(self.forecast)
        return store, GroundTruthReconciler(store, now=lambda: now or self.forecast.target_interval_end + timedelta(hours=2))

    def test_frozen_namespace_and_acquisition_uuid_fixture(self):
        self.assertEqual(str(ACQUISITION_NAMESPACE), "4153ba6e-a333-57b1-8dfe-8b58604bdc30")
        self.assertEqual(str(RECONCILIATION_NAMESPACE), "76b6950a-b1ef-5e10-bd94-025e1b0c92c6")
        self.assertEqual(str(REVISION_NAMESPACE), "14ed017a-e290-5afc-9cf5-abcbd5c536cc")
        self.assertEqual(self._batch().acquisition_id, "ce735da2-ea30-5d8e-b758-910b4cf2e1fd")

    def test_naive_relevant_timestamp_is_invalid_source(self):
        start = self.forecast.target_interval_start
        row = {"sensor_id": 9, "event_time": start.replace(tzinfo=None),
            "period_end_utc": start + timedelta(hours=1), "value_decimal": Decimal("1"),
            "unit": "µg/m³", "record_id": 1}
        store, reconciler = self._reconciler()
        result = reconciler.reconcile(self.forecast, self._batch(records=[row]))
        self.assertEqual((result.status, result.reason_code), ("pending", "invalid_source"))

    def test_acquisition_times_with_microseconds_are_canonicalized_to_seconds(self):
        start = self.forecast.target_interval_start
        batch = AcquisitionBatch.create(sensor_id=9, requested_interval_start=start,
            requested_interval_end=start + timedelta(hours=1), retrieved_at=start + timedelta(hours=3, microseconds=123456),
            source_endpoint="/sensors/9/hours", http_status=200, raw_payload_sha256="c" * 64,
            records=[], normalized_payload_sha256="f" * 64, created_at=start + timedelta(hours=3, microseconds=654321))
        self.assertEqual(batch.retrieved_at.microsecond, 0)
        self.assertEqual(batch.created_at.microsecond, 0)

    def test_non_success_http_status_is_rejected_without_writes(self):
        start = self.forecast.target_interval_start
        for status in (True, 500):
            with self.subTest(status=status), self.assertRaises(ValueError):
                AcquisitionBatch.create(sensor_id=9, requested_interval_start=start,
                    requested_interval_end=start + timedelta(hours=1), retrieved_at=start + timedelta(hours=3),
                    source_endpoint="/sensors/9/hours", http_status=status, raw_payload_sha256="c" * 64,
                    records=[], normalized_payload_sha256="f" * 64, created_at=start + timedelta(hours=3))
        store = SQLiteForecastStore(self.database); store.initialize()
        self.assertEqual((store.count_acquisitions(), store.count_revisions()), (0, 0))

    def test_policy_and_exact_matching_validation(self):
        with self.assertRaises(ValueError):
            TruthPolicy(1, 119)
        store, reconciler = self._reconciler(self.forecast.target_interval_end + timedelta(minutes=119, seconds=59))
        self.assertEqual(reconciler.reconcile(self.forecast, self._batch()).reason_code, "not_mature")
        reconciler = GroundTruthReconciler(store, now=lambda: self.forecast.target_interval_end + timedelta(minutes=120))
        self.assertEqual(reconciler.reconcile(self.forecast, self._batch()).status, "reconciled")
        self.assertEqual(store.get_reconciliation(self.forecast.forecast_id).observed_pm25, 12.5)

    def test_invalid_exact_candidates_never_settle(self):
        start = self.forecast.target_interval_start
        invalid = [
            {"sensor_id": 8, "event_time": start, "period_end_utc": start + timedelta(hours=1), "value_decimal": Decimal("1"), "unit": "µg/m³", "record_id": 1},
            {"sensor_id": 9, "event_time": start + timedelta(minutes=1), "period_end_utc": start + timedelta(hours=1), "value_decimal": Decimal("1"), "unit": "µg/m³", "record_id": 1},
            {"sensor_id": 9, "event_time": start, "period_end_utc": start + timedelta(minutes=30), "value_decimal": Decimal("1"), "unit": "µg/m³", "record_id": 1},
        ]
        for row in invalid:
            with self.subTest(row=row):
                path = Path(self.directory.name) / str(uuid.uuid4())
                store = SQLiteForecastStore(path); store.initialize(); store.insert(self.forecast)
                result = GroundTruthReconciler(store, now=lambda: start + timedelta(hours=3)).reconcile(self.forecast, self._batch(records=[row]))
                self.assertEqual((result.status, result.reason_code), ("pending", "not_available"))

    def test_invalid_and_conflicting_duplicate_sets_are_ambiguous(self):
        start = self.forecast.target_interval_start
        base = {"sensor_id": 9, "event_time": start, "period_end_utc": start + timedelta(hours=1), "unit": "µg/m³"}
        cases = [
            [{**base, "value_decimal": Decimal("1"), "record_id": 1}, {**base, "value_decimal": True, "record_id": 2}],
            [{**base, "value_decimal": Decimal("1"), "record_id": 1}, {**base, "value_decimal": Decimal("2"), "record_id": 2}],
        ]
        for rows in cases:
            path = Path(self.directory.name) / str(uuid.uuid4())
            store = SQLiteForecastStore(path); store.initialize(); store.insert(self.forecast)
            result = GroundTruthReconciler(store, now=lambda: start + timedelta(hours=3)).reconcile(self.forecast, self._batch(records=rows))
            self.assertEqual((result.status, result.reason_code), ("pending", "ambiguous"))

    def test_proven_wrong_hour_ignores_malformed_end_and_keeps_valid_target(self):
        start = self.forecast.target_interval_start
        rows = [
            {"sensor_id": 9, "event_time": start - timedelta(hours=1), "period_end_utc": "bad"},
            {"sensor_id": 9, "event_time": start, "period_end_utc": start + timedelta(hours=1),
             "value_decimal": Decimal("12.5"), "unit": "µg/m³", "record_id": 1},
        ]
        store, reconciler = self._reconciler()
        self.assertEqual(reconciler.reconcile(self.forecast, self._batch(records=rows)).status, "reconciled")

    def test_malformed_unprovable_row_is_invalid_source(self):
        store, reconciler = self._reconciler()
        result = reconciler.reconcile(self.forecast, self._batch(records=[{"value_decimal": Decimal("1")}]))
        self.assertEqual((result.status, result.reason_code), ("pending", "invalid_source"))

    def test_concurrent_initial_settlement_creates_one_canonical_row(self):
        store, reconciler = self._reconciler()
        statuses, errors = [], []
        def settle():
            try:
                statuses.append(reconciler.reconcile(self.forecast, self._batch()).status)
            except Exception as error:
                errors.append(error)
        threads = [threading.Thread(target=settle) for _ in range(6)]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        self.assertFalse(errors)
        self.assertEqual(statuses.count("reconciled"), 1)
        self.assertEqual(statuses.count("already_reconciled"), 5)

    def test_acquisition_conflict_rolls_back_but_exact_retry_is_idempotent(self):
        store, reconciler = self._reconciler()
        first = self._batch()
        self.assertEqual(reconciler.reconcile(self.forecast, first).status, "reconciled")
        conflict = replace(first, source_endpoint="/conflicting")
        with self.assertRaises(ForecastIntegrityError):
            reconciler.reconcile(self.forecast, conflict)
        self.assertEqual(store.count_revisions(), 0)
        self.assertEqual(reconciler.reconcile(self.forecast, first).status, "already_reconciled")

    def test_high_precision_identical_later_observation_is_not_revision(self):
        store, reconciler = self._reconciler()
        value = Decimal("0.1234567890123456789")
        first = self._batch(value=value)
        self.assertEqual(reconciler.reconcile(self.forecast, first).status, "reconciled")
        later = AcquisitionBatch.create(**{**first.__dict__, "acquisition_id": None,
            "retrieved_at": first.retrieved_at + timedelta(hours=1), "raw_payload_sha256": "d" * 64})
        self.assertEqual(reconciler.reconcile(self.forecast, later).status, "already_reconciled")
        self.assertEqual(store.count_revisions(), 0)

    def test_request_range_must_contain_target(self):
        store, reconciler = self._reconciler()
        base = self._batch()
        containing = AcquisitionBatch.create(**{**base.__dict__, "acquisition_id": None,
            "requested_interval_start": base.requested_interval_start - timedelta(hours=1),
            "requested_interval_end": base.requested_interval_end + timedelta(hours=1)})
        self.assertEqual(reconciler.reconcile(self.forecast, containing).status, "reconciled")
        for start, end in ((base.requested_interval_start + timedelta(minutes=1), base.requested_interval_end),
                           (base.requested_interval_start, base.requested_interval_end - timedelta(minutes=1))):
            path = Path(self.directory.name) / str(uuid.uuid4())
            candidate_store = SQLiteForecastStore(path); candidate_store.initialize(); candidate_store.insert(self.forecast)
            candidate = AcquisitionBatch.create(**{**base.__dict__, "acquisition_id": None,
                "requested_interval_start": start, "requested_interval_end": end})
            with self.assertRaises(ForecastIntegrityError):
                GroundTruthReconciler(candidate_store, now=lambda: self.forecast.target_interval_end + timedelta(hours=2)).reconcile(self.forecast, candidate)

    def test_settlement_rejects_mismatched_acquisition_relationship(self):
        store, reconciler = self._reconciler()
        batch = replace(self._batch(), sensor_id=8)
        with self.assertRaises(ForecastIntegrityError):
            reconciler.reconcile(self.forecast, batch)
        self.assertIsNone(store.get_reconciliation(self.forecast.forecast_id))

    def test_malformed_relevance_uses_sensor_identity_before_timestamps(self):
        store, reconciler = self._reconciler()
        irrelevant = {"sensor_id": 999, "event_time": "bad", "value_decimal": Decimal("1")}
        result = reconciler.reconcile(self.forecast, self._batch(records=[irrelevant]))
        self.assertEqual((result.status, result.reason_code), ("pending", "not_available"))
        missing_sensor = {"event_time": self.forecast.target_interval_start,
            "period_end_utc": self.forecast.target_interval_end, "value_decimal": Decimal("1"), "unit": "µg/m³"}
        result = reconciler.reconcile(self.forecast, self._batch(records=[missing_sensor]))
        self.assertEqual((result.status, result.reason_code), ("pending", "invalid_source"))

    def test_successful_pending_outcomes_persist_acquisition_only(self):
        cases = [
            ([{"sensor_id": 8, "event_time": self.forecast.target_interval_start,
               "period_end_utc": self.forecast.target_interval_end, "value_decimal": Decimal("1"), "unit": "µg/m³"}], "not_available"),
            ([{"sensor_id": 9, "event_time": self.forecast.target_interval_start,
               "period_end_utc": self.forecast.target_interval_end, "value_decimal": Decimal("1"), "unit": "µg/m³"},
              {"sensor_id": 9, "event_time": self.forecast.target_interval_start,
               "period_end_utc": self.forecast.target_interval_end, "value_decimal": Decimal("2"), "unit": "µg/m³"}], "ambiguous"),
            ([{"value_decimal": Decimal("1")}], "invalid_source"),
        ]
        for records, reason in cases:
            path = Path(self.directory.name) / str(uuid.uuid4())
            store = SQLiteForecastStore(path); store.initialize(); store.insert(self.forecast)
            reconciler = GroundTruthReconciler(store, now=lambda: self.forecast.target_interval_end + timedelta(hours=2))
            batch = self._batch(records=records)
            result = reconciler.reconcile(self.forecast, batch)
            self.assertEqual((result.status, result.reason_code), ("pending", reason))
            self.assertEqual(store.count_acquisitions(), 1)
            self.assertIsNone(store.get_reconciliation(self.forecast.forecast_id))
            self.assertEqual(store.count_revisions(), 0)
            self.assertEqual(reconciler.reconcile(self.forecast, batch).reason_code, reason)
            self.assertEqual(store.count_acquisitions(), 1)

    def test_concurrent_revision_is_single_and_conflicting_revision_fails(self):
        store, reconciler = self._reconciler()
        self.assertEqual(reconciler.reconcile(self.forecast, self._batch()).status, "reconciled")
        changed = self._batch(value=Decimal("13"), retrieved_at=self.forecast.target_interval_end + timedelta(hours=3))
        statuses, errors = [], []
        def revise():
            try: statuses.append(reconciler.reconcile(self.forecast, changed).status)
            except Exception as error: errors.append(error)
        threads = [threading.Thread(target=revise) for _ in range(6)]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        self.assertFalse(errors)
        self.assertEqual(store.count_revisions(), 1)
        with store._connect() as connection:
            connection.execute("UPDATE observation_revisions SET raw_payload_sha256=?", ("0" * 64,))
        with self.assertRaises(ForecastIntegrityError):
            reconciler.reconcile(self.forecast, changed)

    def test_database_failure_returns_frozen_classification_without_partial_rows(self):
        store, reconciler = self._reconciler()
        with patch.object(store, "settle", side_effect=sqlite3.OperationalError("boom")):
            result = reconciler.reconcile(self.forecast, self._batch())
        self.assertEqual((result.status, result.reason_code), ("failed", "database_error"))
        self.assertEqual(store.count_acquisitions(), 0)
        self.assertIsNone(store.get_reconciliation(self.forecast.forecast_id))

    def test_frozen_reconciliation_revision_ids_and_reopen_persistence(self):
        store, reconciler = self._reconciler()
        before = store.get_by_id(self.forecast.forecast_id)
        self.assertEqual(reconciler.reconcile(self.forecast, self._batch()).status, "reconciled")
        changed = self._batch(value=Decimal("13"), retrieved_at=self.forecast.target_interval_end + timedelta(hours=3))
        changed = AcquisitionBatch.create(**{**changed.__dict__, "acquisition_id": None, "raw_payload_sha256": "d" * 64})
        self.assertEqual(reconciler.reconcile(self.forecast, changed).status, "revision_detected")
        with store._connect() as connection:
            reconciliation = connection.execute("SELECT * FROM forecast_reconciliations").fetchone()
            revision = connection.execute("SELECT * FROM observation_revisions").fetchone()
        self.assertEqual(reconciliation["reconciliation_id"], "74de0370-015b-514d-bf9b-33cdf98816b0")
        self.assertEqual(revision["revision_id"], "b7afe44f-df56-5751-90ff-65ada1aa404c")
        self.assertEqual((reconciliation["truth_policy_version"], reconciliation["reconciliation_delay_minutes"]), (1, 120))
        reopened = SQLiteForecastStore(self.database); reopened.initialize()
        self.assertEqual(reopened.get_by_id(self.forecast.forecast_id), before)
        self.assertEqual(reopened.get_reconciliation(self.forecast.forecast_id).observed_pm25, 12.5)
        self.assertEqual(reopened.count_revisions(), 1)

    def test_equal_duplicates_are_order_independent_and_preserve_sorted_ids(self):
        start = self.forecast.target_interval_start
        rows = [{"sensor_id": 9, "event_time": start, "period_end_utc": start + timedelta(hours=1),
                 "value_decimal": Decimal("12.50"), "unit": "µg/m³", "record_id": record_id}
                for record_id in ("z", None, "a")]
        store, reconciler = self._reconciler()
        self.assertEqual(reconciler.reconcile(self.forecast, self._batch(records=list(reversed(rows)))).status, "reconciled")
        with store._connect() as connection:
            stored = connection.execute("SELECT source_record_ids_json, normalized_candidate_sha256 FROM forecast_reconciliations").fetchone()
        self.assertEqual(stored["source_record_ids_json"], '[null,"a","z"]')
        self.assertEqual(stored["normalized_candidate_sha256"], normalized_sha256(rows))

    def test_wrong_unit_is_invalid_source_and_offset_equivalent_utc_matches(self):
        start = self.forecast.target_interval_start
        wrong = {"sensor_id": 9, "event_time": start, "period_end_utc": start + timedelta(hours=1),
            "value_decimal": Decimal("1"), "unit": "ppm", "record_id": 1}
        store, reconciler = self._reconciler()
        self.assertEqual(reconciler.reconcile(self.forecast, self._batch(records=[wrong])).reason_code, "invalid_source")
        offset = timezone(timedelta(hours=7))
        equivalent = {"sensor_id": 9, "event_time": start.astimezone(offset),
            "period_end_utc": (start + timedelta(hours=1)).astimezone(offset), "value_decimal": Decimal("12.5"),
            "unit": "µg/m³", "record_id": 1}
        self.assertEqual(reconciler.reconcile(self.forecast, self._batch(records=[equivalent])).status, "reconciled")

    def test_eligible_selection_order_limit_and_settled_exclusion(self):
        store = SQLiteForecastStore(self.database); store.initialize()
        forecasts = []
        for model_version, offset in (("z", 1), ("a", 0), ("b", 0)):
            prediction = self.prediction_time + timedelta(hours=offset)
            record = ForecastRecord.create(**{**self.forecast.as_dict(), "forecast_id": None,
                "model_version": model_version, "prediction_time": prediction,
                "target_interval_start": prediction + timedelta(hours=6),
                "target_interval_end": prediction + timedelta(hours=7), "input_data_cutoff": prediction,
                "history_start": prediction - timedelta(hours=24), "history_end": prediction - timedelta(hours=1)})
            store.insert(record); forecasts.append(record)
        now = self.forecast.target_interval_end + timedelta(hours=2)
        selected = store.eligible_forecasts(now, 120, limit=2)
        expected = sorted((row for row in forecasts if row.target_interval_end + timedelta(minutes=120) <= now),
            key=lambda row: (row.target_interval_end, row.forecast_id))[:2]
        self.assertEqual(selected, expected)
        reconciler = GroundTruthReconciler(store, now=lambda: now)
        reconciler.reconcile(expected[0], self._batch())
        self.assertNotIn(expected[0], store.eligible_forecasts(now, 120))

    def test_grouping_is_deterministic_and_reuses_shared_adjacent_ranges(self):
        forecasts = []
        for sensor, offset, model in ((9, 0, "a"), (9, 0, "b"), (9, 1, "c"), (9, 3, "d"), (10, 0, "e")):
            prediction = self.prediction_time + timedelta(hours=offset)
            forecasts.append(ForecastRecord.create(**{**self.forecast.as_dict(), "forecast_id": None,
                "sensor_id": sensor, "model_version": model, "prediction_time": prediction,
                "target_interval_start": prediction + timedelta(hours=6),
                "target_interval_end": prediction + timedelta(hours=7), "input_data_cutoff": prediction,
                "history_start": prediction - timedelta(hours=24), "history_end": prediction - timedelta(hours=1)}))
        plans = GroundTruthReconciler.plan_requests(list(reversed(forecasts)))
        self.assertEqual([(plan.sensor_id, plan.start, plan.end, len(plan.forecasts)) for plan in plans], [
            (9, self.forecast.target_interval_start, self.forecast.target_interval_end + timedelta(hours=1), 3),
            (9, self.forecast.target_interval_start + timedelta(hours=3), self.forecast.target_interval_end + timedelta(hours=3), 1),
            (10, self.forecast.target_interval_start, self.forecast.target_interval_end, 1),
        ])

    def test_provider_neutral_run_orders_results_and_maps_source_failures(self):
        store = SQLiteForecastStore(self.database); store.initialize(); store.insert(self.forecast)
        now = self.forecast.target_interval_end + timedelta(hours=2)
        class Source:
            def __init__(self, error=None): self.error, self.calls = error, []
            def acquire(self, sensor_id, start, end):
                self.calls.append((sensor_id, start, end))
                if self.error: raise self.error("failure")
                return self_batch
        self_batch = self._batch()
        source = Source()
        run = GroundTruthReconciler(store, now=lambda: now, source=source).run()
        self.assertEqual([item.status for item in run.results], ["reconciled"])
        self.assertEqual(run.counts, {"reconciled": 1})
        self.assertEqual(source.calls, [(9, self.forecast.target_interval_start, self.forecast.target_interval_end)])
        for error, reason in ((SourceTransportError, "transport_error"), (SourceParseError, "parse_error")):
            path = Path(self.directory.name) / str(uuid.uuid4())
            failed_store = SQLiteForecastStore(path); failed_store.initialize(); failed_store.insert(self.forecast)
            result = GroundTruthReconciler(failed_store, now=lambda: now, source=Source(error)).run().results[0]
            self.assertEqual((result.status, result.reason_code), ("failed", reason))
            self.assertEqual(failed_store.count_acquisitions(), 0)

    def test_run_isolates_integrity_failure_and_continues_later_plan(self):
        store = SQLiteForecastStore(self.database); store.initialize(); store.insert(self.forecast)
        second = ForecastRecord.create(**{**self.forecast.as_dict(), "forecast_id": None, "sensor_id": 10,
            "prediction_time": self.prediction_time + timedelta(hours=1),
            "target_interval_start": self.forecast.target_interval_start + timedelta(hours=1),
            "target_interval_end": self.forecast.target_interval_end + timedelta(hours=1),
            "input_data_cutoff": self.prediction_time + timedelta(hours=1),
            "history_start": self.prediction_time - timedelta(hours=23), "history_end": self.prediction_time})
        store.insert(second)
        conflict = self._batch()
        store.persist_acquisition(conflict)
        valid = AcquisitionBatch.create(sensor_id=10, requested_interval_start=second.target_interval_start,
            requested_interval_end=second.target_interval_end, retrieved_at=second.target_interval_end + timedelta(hours=3),
            source_endpoint="/sensors/10/hours", http_status=200, raw_payload_sha256="d" * 64,
            records=[{"sensor_id": 10, "event_time": second.target_interval_start,
                "period_end_utc": second.target_interval_end, "value_decimal": Decimal("12.5"), "unit": "µg/m³"}],
            created_at=second.target_interval_end + timedelta(hours=3))
        class Source:
            def acquire(self, sensor_id, start, end):
                return replace(conflict, source_endpoint="/conflict") if sensor_id == 9 else valid
        run = GroundTruthReconciler(store, now=lambda: second.target_interval_end + timedelta(hours=3), source=Source()).run()
        self.assertEqual([(item.forecast_id, item.status, item.reason_code) for item in run.results], [
            (self.forecast.forecast_id, "failed", "database_error"), (second.forecast_id, "reconciled", None)])
        self.assertIsNotNone(store.get_reconciliation(second.forecast_id))

    def test_run_does_not_swallow_unexpected_reconcile_exception(self):
        store = SQLiteForecastStore(self.database); store.initialize(); store.insert(self.forecast)
        reconciler = GroundTruthReconciler(store, now=lambda: self.forecast.target_interval_end + timedelta(hours=2),
            source=type("Source", (), {"acquire": lambda _, sensor_id, start, end: self._batch()})())
        with patch.object(reconciler, "reconcile", side_effect=RuntimeError("unexpected")), self.assertRaisesRegex(RuntimeError, "unexpected"):
            reconciler.run()

    def test_settlement_database_failure_rolls_back_acquisition_transaction(self):
        store, reconciler = self._reconciler()
        before = (store.count_acquisitions(), store.count_revisions(), store.get_reconciliation(self.forecast.forecast_id))
        with store._connect() as connection:
            connection.execute("CREATE TRIGGER fail_reconciliation BEFORE INSERT ON forecast_reconciliations BEGIN SELECT RAISE(ABORT, 'boom'); END")
        result = reconciler.reconcile(self.forecast, self._batch())
        self.assertEqual((result.status, result.reason_code), ("failed", "database_error"))
        self.assertEqual((store.count_acquisitions(), store.count_revisions(), store.get_reconciliation(self.forecast.forecast_id)), before)

    def test_run_reuses_acquisition_across_models_and_later_revision(self):
        store = SQLiteForecastStore(self.database); store.initialize(); store.insert(self.forecast)
        second = ForecastRecord.create(**{**self.forecast.as_dict(), "forecast_id": None, "model_version": "v2"})
        store.insert(second)
        now = self.forecast.target_interval_end + timedelta(hours=2)
        class Source:
            def __init__(self, batch): self.batch, self.calls = batch, 0
            def acquire(self, sensor_id, start, end): self.calls += 1; return self.batch
        source = Source(self._batch())
        first_run = GroundTruthReconciler(store, now=lambda: now, source=source).run()
        self.assertEqual([item.status for item in first_run.results], ["reconciled", "reconciled"])
        self.assertEqual((source.calls, store.count_acquisitions()), (1, 1))
        changed = self._batch(value=Decimal("13"), retrieved_at=self.forecast.target_interval_end + timedelta(hours=3))
        source = Source(changed)
        revision_run = GroundTruthReconciler(store, now=lambda: now + timedelta(hours=1), source=source).run_for_targets(
            [(9, self.forecast.target_interval_start, self.forecast.target_interval_end)])
        self.assertEqual([item.status for item in revision_run.results], ["revision_detected", "revision_detected"])
        self.assertEqual(store.count_revisions(), 2)

    def test_run_canonicalizes_fractional_clock_once_for_maturity_and_persistence(self):
        store = SQLiteForecastStore(self.database); store.initialize(); store.insert(self.forecast)
        now = self.forecast.target_interval_end + timedelta(hours=2, microseconds=123456)
        class Source:
            def acquire(self, sensor_id, start, end):
                return self._batch
        source = Source(); source._batch = self._batch()
        result = GroundTruthReconciler(store, now=lambda: now, source=source).run()
        self.assertEqual(result.results[0].status, "reconciled")
        with store._connect() as connection:
            self.assertEqual(connection.execute("SELECT reconciled_at FROM forecast_reconciliations").fetchone()[0],
                (now.replace(microsecond=0)).strftime("%Y-%m-%dT%H:%M:%SZ"))

    def test_finite_decimal_that_overflows_real_is_invalid_source_without_settlement(self):
        store, reconciler = self._reconciler()
        result = reconciler.reconcile(self.forecast, self._batch(value=Decimal("1E+400")))
        self.assertEqual((result.status, result.reason_code), ("pending", "invalid_source"))
        self.assertEqual(store.count_acquisitions(), 1)
        self.assertIsNone(store.get_reconciliation(self.forecast.forecast_id))
        self.assertEqual(store.count_revisions(), 0)

    def test_partial_page_failure_has_no_database_evidence(self):
        store = SQLiteForecastStore(self.database); store.initialize(); store.insert(self.forecast)
        class Source:
            def acquire(self, sensor_id, start, end):
                raise SourceTransportError("page two failed")
        run = GroundTruthReconciler(store, now=lambda: self.forecast.target_interval_end + timedelta(hours=2), source=Source()).run()
        self.assertEqual((run.results[0].status, run.results[0].reason_code), ("failed", "transport_error"))
        self.assertEqual(store.count_acquisitions(), 0)
        self.assertIsNone(store.get_reconciliation(self.forecast.forecast_id))
        self.assertEqual(store.count_revisions(), 0)

    def test_partial_page_parse_failure_has_no_database_evidence(self):
        store = SQLiteForecastStore(self.database); store.initialize(); store.insert(self.forecast)
        class Source:
            def acquire(self, sensor_id, start, end):
                raise SourceParseError("page two invalid")
        run = GroundTruthReconciler(store, now=lambda: self.forecast.target_interval_end + timedelta(hours=2), source=Source()).run()
        self.assertEqual((run.results[0].status, run.results[0].reason_code), ("failed", "parse_error"))
        self.assertEqual(store.count_acquisitions(), 0)
        self.assertIsNone(store.get_reconciliation(self.forecast.forecast_id))
        self.assertEqual(store.count_revisions(), 0)

    def test_cli_is_thin_machine_readable_and_has_no_delay_override(self):
        root = Path(__file__).resolve().parents[1]
        help_result = subprocess.run([sys.executable, "-m", "scripts.reconcile_ground_truth", "--help"],
            cwd=root, capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("--database", help_result.stdout)
        self.assertIn("--limit", help_result.stdout)
        self.assertNotIn("--delay", help_result.stdout)
        from scripts import reconcile_ground_truth as cli
        fake = type("Run", (), {"results": (ReconcileResult("pending", "not_available", "f"),),
            "counts": {"pending": 1}})()
        with patch.object(cli, "run_reconciliation", return_value=fake) as run, patch.dict(os.environ, {"OPENAQ_API_KEY": "secret"}):
            output = __import__("io").StringIO()
            with patch("sys.stdout", output):
                code = cli.main(["--database", str(self.database), "--raw-directory", str(Path(self.directory.name) / "raw")])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["counts"], {"pending": 1})
        run.assert_called_once()

    def test_retry_equal_and_changed_acquisitions_preserve_canonical_truth(self):
        store, reconciler = self._reconciler()
        first = self._batch()
        self.assertEqual(reconciler.reconcile(self.forecast, first).status, "reconciled")
        self.assertEqual(reconciler.reconcile(self.forecast, first).status, "already_reconciled")
        later_equal = replace(first, acquisition_id="", retrieved_at=first.retrieved_at + timedelta(hours=1), raw_payload_sha256="d" * 64)
        later_equal = AcquisitionBatch.create(**{**later_equal.__dict__, "acquisition_id": None})
        self.assertEqual(reconciler.reconcile(self.forecast, later_equal).status, "already_reconciled")
        changed = AcquisitionBatch.create(**{**later_equal.__dict__, "acquisition_id": None,
            "retrieved_at": later_equal.retrieved_at + timedelta(hours=1), "raw_payload_sha256": "e" * 64,
            "records": tuple({**row, "value_decimal": Decimal("13")} for row in later_equal.records)})
        self.assertEqual(reconciler.reconcile(self.forecast, changed).status, "revision_detected")
        self.assertEqual(store.get_reconciliation(self.forecast.forecast_id).observed_pm25, 12.5)
        self.assertEqual(store.count_revisions(), 1)


if __name__ == "__main__":
    unittest.main()
