import os
import sqlite3
import unittest
import uuid
from contextlib import closing
from dataclasses import fields, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import patch

from app.forecast_ledger import (
    ForecastIntegrityError,
    ForecastRecord,
    ForecastStoreConfigurationError,
    LedgerDatabaseError,
    LedgerTransientError,
    LibSQLForecastStore,
    SCHEMA_VERSION,
    create_forecast_store,
)
from app.ground_truth_reconciler import AcquisitionBatch, GroundTruthReconciler


UTC = timezone.utc


class _SQLiteLibSQLDriver:
    Error = sqlite3.Error

    def __init__(self):
        self.uri = f"file:libsql-adapter-{uuid.uuid4()}?mode=memory&cache=shared"
        self.keeper = sqlite3.connect(self.uri, uri=True)

    def connect(self, database, auth_token, timeout=30, isolation_level=None):
        return sqlite3.connect(self.uri, uri=True, timeout=timeout, isolation_level=isolation_level)

    def close(self):
        self.keeper.close()


class _DriverError(Exception):
    pass


class _FailingDriver:
    Error = _DriverError

    def __init__(self, message):
        self.message = message

    def connect(self, **kwargs):
        raise self.Error(self.message)


class _ValueErrorDriver:
    Error = _DriverError

    def connect(self, **kwargs):
        raise ValueError("Hrana stream error")


def _forecast(model_hash="a" * 64, prediction_time=None, issued_at=None):
    prediction_time = prediction_time or datetime(2026, 1, 1, tzinfo=UTC)
    return ForecastRecord.create(
        sensor_id=9,
        prediction_time=prediction_time,
        predicted_pm25=20,
        persistence_prediction=18,
        model_version="v1",
        model_artifact_sha256=model_hash,
        feature_schema_sha256="b" * 64,
        artifact_version=1,
        feature_configuration="A2",
        source_retrieved_at=prediction_time,
        input_data_cutoff=prediction_time,
        history_start=prediction_time - timedelta(hours=24),
        history_end=prediction_time - timedelta(hours=1),
        data_mode_at_issue="fresh_openaq",
        freshness_status_at_issue="fresh",
        source_age_minutes_at_issue=0,
        issued_at=issued_at or prediction_time,
    )


class LibSQLForecastStoreTests(unittest.TestCase):
    def setUp(self):
        self.driver = _SQLiteLibSQLDriver()
        self.driver_patch = patch.object(LibSQLForecastStore, "_load_driver", return_value=self.driver)
        self.driver_patch.start()
        self.store = LibSQLForecastStore("libsql://unit-test", "test-token")

    def tearDown(self):
        self.driver_patch.stop()
        self.driver.close()

    def test_factory_selects_libsql_and_passes_read_only(self):
        environment = {
            "AIRAWARE_LEDGER_BACKEND": "libsql",
            "AIRAWARE_TURSO_DATABASE_URL": "libsql://example.turso.io",
            "AIRAWARE_TURSO_AUTH_TOKEN": "secret",
        }
        with patch.dict(os.environ, environment, clear=True):
            store = create_forecast_store("ignored.sqlite3", read_only=True)
        self.assertIsInstance(store, LibSQLForecastStore)
        self.assertEqual(store.database_url, environment["AIRAWARE_TURSO_DATABASE_URL"])
        self.assertEqual(store.auth_token, environment["AIRAWARE_TURSO_AUTH_TOKEN"])
        self.assertTrue(store.read_only)

    def test_factory_requires_remote_url_and_token(self):
        for missing in ("AIRAWARE_TURSO_DATABASE_URL", "AIRAWARE_TURSO_AUTH_TOKEN"):
            environment = {
                "AIRAWARE_LEDGER_BACKEND": "libsql",
                "AIRAWARE_TURSO_DATABASE_URL": "libsql://example.turso.io",
                "AIRAWARE_TURSO_AUTH_TOKEN": "secret",
            }
            environment.pop(missing)
            with self.subTest(missing=missing), patch.dict(os.environ, environment, clear=True), self.assertRaisesRegex(
                    ForecastStoreConfigurationError, missing):
                create_forecast_store("ignored.sqlite3")

    def test_driver_errors_translate_to_backend_neutral_errors(self):
        with patch.object(LibSQLForecastStore, "_load_driver", return_value=_FailingDriver("database is busy")):
            with self.assertRaises(LedgerTransientError):
                self.store.latest()
        with patch.object(LibSQLForecastStore, "_load_driver", return_value=_FailingDriver("remote query failed")):
            with self.assertRaises(LedgerDatabaseError):
                self.store.latest()
        with patch.object(LibSQLForecastStore, "_load_driver", return_value=_ValueErrorDriver()):
            with self.assertRaises(LedgerDatabaseError):
                self.store.latest()

    def test_row_adapter_matches_sqlite_row_access_patterns(self):
        with closing(self.store._connect()) as connection:
            row = connection.execute("SELECT 7 AS number, 'air' AS label").fetchone()
        self.assertEqual(row[0], 7)
        self.assertEqual(row["label"], "air")
        self.assertEqual(tuple(row), (7, "air"))
        self.assertEqual(dict(row), {"number": 7, "label": "air"})

    def test_remote_schema_initializes_and_validates_idempotently(self):
        self.store.initialize()
        self.store.initialize()
        self.store.validate_existing()
        with closing(self.store._connect()) as connection:
            self.assertEqual(self.store._remote_schema_versions(connection), (SCHEMA_VERSION,))

    def test_remote_schema_rejects_future_and_partial_versions(self):
        self.store.initialize()
        with closing(self.store._connect()) as connection:
            connection.execute("INSERT INTO schema_migrations VALUES (?, ?)", (SCHEMA_VERSION + 1, "2026-01-01T00:00:00Z"))
            connection.commit()
        with self.assertRaisesRegex(RuntimeError, "unsupported schema version"):
            self.store.validate_existing()
        with closing(self.store._connect()) as connection:
            connection.execute("DELETE FROM schema_migrations WHERE version=?", (SCHEMA_VERSION + 1,))
            connection.execute("DELETE FROM schema_migrations WHERE version=?", (SCHEMA_VERSION,))
            connection.commit()
        with self.assertRaisesRegex(RuntimeError, "unsupported schema"):
            self.store.validate_existing()

    def test_remote_schema_rejects_missing_domain_table(self):
        self.store.initialize()
        with closing(self.store._connect()) as connection:
            connection.execute("DROP TABLE consumer_performance_publications")
            connection.commit()
        with self.assertRaisesRegex(RuntimeError, "unsupported schema"):
            self.store.validate_existing()

    def test_read_only_store_validates_and_reads_without_mutating(self):
        self.store.initialize()
        record = _forecast()
        self.store.insert(record)
        with closing(self.store._connect()) as connection:
            before = self.store._remote_schema_versions(connection)
        reader = LibSQLForecastStore("libsql://unit-test", "reader-token", read_only=True)
        reader.validate_existing()
        self.assertEqual(reader.latest(), record)
        with closing(self.store._connect()) as connection:
            self.assertEqual(self.store._remote_schema_versions(connection), before)
        with self.assertRaisesRegex(LedgerDatabaseError, "read-only"):
            reader.initialize()
        with self.assertRaisesRegex(LedgerDatabaseError, "read-only"):
            reader.insert(_forecast("c" * 64))
        with self.assertRaisesRegex(LedgerDatabaseError, "read-only"):
            reader.publish_consumer_performance(
                "v1", "a" * 64, "b" * 64, 9, datetime(2026, 1, 2, tzinfo=UTC), minimum_verified_count=1)
        with closing(reader._connect()) as connection, self.assertRaisesRegex(LedgerDatabaseError, "read-only"):
            connection.execute("PRAGMA foreign_keys(OFF)")

    def test_adapter_preserves_domain_write_contract(self):
        self.store.initialize()
        forecast = _forecast()
        self.assertEqual(self.store.insert(forecast).status, "inserted")
        batch = AcquisitionBatch.create(
            sensor_id=forecast.sensor_id,
            requested_interval_start=forecast.target_interval_start,
            requested_interval_end=forecast.target_interval_end,
            retrieved_at=forecast.target_interval_end + timedelta(hours=2),
            source_endpoint="/sensors/9/hours",
            http_status=200,
            raw_payload_sha256="d" * 64,
            normalized_payload_sha256="e" * 64,
            created_at=forecast.target_interval_end + timedelta(hours=2),
            records=[{
                "sensor_id": forecast.sensor_id,
                "event_time": forecast.target_interval_start,
                "period_end_utc": forecast.target_interval_end,
                "value_decimal": Decimal("12.5"),
                "unit": "µg/m³",
                "record_id": 1,
            }],
        )
        reconciler = GroundTruthReconciler(
            self.store, now=lambda: forecast.target_interval_end + timedelta(hours=2))
        self.assertEqual(reconciler.reconcile(forecast, batch).status, "reconciled")
        self.assertEqual(self.store.count_acquisitions(), 1)
        evaluation = self.store.materialize_evaluation(forecast.forecast_id).record
        snapshot = self.store.create_evaluation_run_snapshot(
            "v1", "a" * 64, "b" * 64, 9,
            evaluation.target_interval_end, evaluation.target_interval_end + timedelta(hours=1),
            forecast.target_interval_end + timedelta(hours=2)).snapshot
        self.assertEqual(snapshot.evaluation_ids, (evaluation.evaluation_id,))
        publication = self.store.publish_consumer_performance(
            "v1", "a" * 64, "b" * 64, 9,
            datetime(2026, 1, 2, 17, 0, tzinfo=UTC), minimum_verified_count=1)
        self.assertTrue(publication.available)

    def test_latest_uses_explicit_deterministic_tie_breaker(self):
        self.store.initialize()
        issued = datetime(2026, 1, 2, tzinfo=UTC)
        records = [_forecast("a" * 64, issued_at=issued), _forecast("c" * 64, issued_at=issued)]
        expected = max(records, key=lambda record: record.forecast_id)
        other = min(records, key=lambda record: record.forecast_id)
        self.store.insert(expected)
        self.store.insert(other)
        self.assertEqual(self.store.latest(), expected)

    def test_current_publication_uses_explicit_deterministic_tie_breaker(self):
        self.store.initialize()
        common = (
            "2026-01-02", "2025-12-02T17:00:00Z", "2026-01-01T17:00:00Z", "v1",
            "a" * 64, "b" * 64, 9, 1, 6, 0,
        )
        published_at = "2026-01-02T18:00:00Z"
        rows = [
            ("publication-a", *common, 0, 0, "insufficient_history", None, None, None, None, "c" * 64, published_at),
            ("publication-b", *common, 1, 0, "insufficient_history", None, None, None, None, "d" * 64, published_at),
        ]
        with closing(self.store._connect()) as connection:
            for row in rows:
                connection.execute(
                    "INSERT INTO consumer_performance_publications VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
            connection.commit()
        publication = self.store.current_consumer_performance("v1", "a" * 64, "b" * 64, 9)
        self.assertEqual(publication.publication_id, "publication-b")


TEST_TURSO_URL = os.environ.get("AIRAWARE_TEST_TURSO_DATABASE_URL", "").strip()
TEST_TURSO_TOKEN = os.environ.get("AIRAWARE_TEST_TURSO_AUTH_TOKEN", "").strip()
TEST_TURSO_READER_TOKEN = os.environ.get("AIRAWARE_TEST_TURSO_READER_AUTH_TOKEN", "").strip()


@unittest.skipUnless(TEST_TURSO_URL and TEST_TURSO_TOKEN, "real Turso test credentials unavailable")
class LibSQLForecastStoreIntegrationTests(unittest.TestCase):
    def test_real_turso_contract_on_dedicated_empty_database(self):
        store = LibSQLForecastStore(TEST_TURSO_URL, TEST_TURSO_TOKEN)
        with closing(store._connect()) as connection:
            existing_tables = store._remote_tables(connection)
        if existing_tables:
            self.fail("AIRAWARE_TEST_TURSO_DATABASE_URL must reference a dedicated empty integration database")

        store.initialize()
        store.initialize()
        store.validate_existing()
        with closing(store._connect()) as connection:
            self.assertEqual(store._remote_schema_versions(connection), (SCHEMA_VERSION,))
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute("INSERT INTO schema_migrations VALUES (?, ?)",
                    (SCHEMA_VERSION + 1, "2026-01-01T00:00:00Z"))
                with self.assertRaisesRegex(RuntimeError, "unsupported schema version"):
                    store._validate_remote_schema(connection)
            finally:
                connection.rollback()

        forecast = _forecast()
        self.assertEqual(store.insert(forecast).status, "inserted")
        self.assertEqual(store.insert(forecast).status, "already_exists")
        with self.assertRaises(ForecastIntegrityError):
            store.insert(replace(forecast, predicted_pm25=99))
        self.assertEqual(store.get_by_id(forecast.forecast_id), forecast)
        self.assertEqual(store.latest(), forecast)

        before = store.count()
        rollback_forecast = _forecast("c" * 64, prediction_time=forecast.prediction_time + timedelta(hours=1))
        with patch.object(store, "_after_insert", side_effect=RuntimeError("rollback probe")):
            with self.assertRaisesRegex(RuntimeError, "rollback probe"):
                store.insert(rollback_forecast)
        self.assertEqual(store.count(), before)

        batch = AcquisitionBatch.create(
            sensor_id=forecast.sensor_id,
            requested_interval_start=forecast.target_interval_start,
            requested_interval_end=forecast.target_interval_end,
            retrieved_at=forecast.target_interval_end + timedelta(hours=2),
            source_endpoint="/sensors/9/hours",
            http_status=200,
            raw_payload_sha256="d" * 64,
            normalized_payload_sha256="e" * 64,
            created_at=forecast.target_interval_end + timedelta(hours=2),
            records=[{
                "sensor_id": forecast.sensor_id,
                "event_time": forecast.target_interval_start,
                "period_end_utc": forecast.target_interval_end,
                "value_decimal": Decimal("12.5"),
                "unit": "µg/m³",
                "record_id": 1,
            }],
        )
        reconciler = GroundTruthReconciler(store, now=lambda: forecast.target_interval_end + timedelta(hours=2))
        self.assertEqual(reconciler.reconcile(forecast, batch).status, "reconciled")
        self.assertEqual(reconciler.reconcile(forecast, batch).status, "already_reconciled")
        evaluation = store.materialize_evaluation(forecast.forecast_id).record
        snapshot = store.create_evaluation_run_snapshot(
            "v1", "a" * 64, "b" * 64, 9,
            evaluation.target_interval_end, evaluation.target_interval_end + timedelta(hours=1),
            forecast.target_interval_end + timedelta(hours=2)).snapshot
        self.assertEqual(snapshot.evaluation_ids, (evaluation.evaluation_id,))
        publication = store.publish_consumer_performance(
            "v1", "a" * 64, "b" * 64, 9,
            datetime(2026, 1, 2, 17, 0, tzinfo=UTC), minimum_verified_count=1)
        self.assertTrue(publication.available)
        self.assertEqual(store.current_consumer_performance("v1", "a" * 64, "b" * 64, 9), publication)

        with closing(store._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute("""INSERT INTO evaluation_rows VALUES
                        (?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                        "orphan-evaluation", "missing-forecast", "missing-reconciliation", 9,
                        "2026-01-01T06:00:00Z", "2026-01-01T07:00:00Z", 1.0, 1.0, 1.0, 0.0, 0.0,
                        "2026-01-01T09:00:00Z", 1))
                names = tuple(field.name for field in fields(ForecastRecord))
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                        f"INSERT INTO forecasts ({', '.join(names)}) VALUES ({', '.join('?' for _ in names)})",
                        store._values(forecast))
            finally:
                connection.rollback()

        issued = datetime(2026, 1, 3, tzinfo=UTC)
        tied = [
            _forecast("f" * 64, prediction_time=forecast.prediction_time + timedelta(hours=2), issued_at=issued),
            _forecast("1" * 64, prediction_time=forecast.prediction_time + timedelta(hours=2), issued_at=issued),
        ]
        expected_latest = max(tied, key=lambda record: record.forecast_id)
        store.insert(expected_latest)
        store.insert(min(tied, key=lambda record: record.forecast_id))
        self.assertEqual(store.latest(), expected_latest)

        reader = LibSQLForecastStore(TEST_TURSO_URL, TEST_TURSO_TOKEN, read_only=True)
        reader.validate_existing()
        self.assertEqual(reader.latest(), expected_latest)
        with self.assertRaisesRegex(LedgerDatabaseError, "read-only"):
            reader.initialize()
        with self.assertRaisesRegex(LedgerDatabaseError, "read-only"):
            reader.insert(_forecast("2" * 64, prediction_time=forecast.prediction_time + timedelta(hours=3)))

    @unittest.skipUnless(TEST_TURSO_READER_TOKEN,
        "real Turso reader token unavailable; server-side read-only authorization untested")
    def test_real_turso_reader_token_rejects_direct_database_write(self):
        store = LibSQLForecastStore(TEST_TURSO_URL, TEST_TURSO_TOKEN)
        store.validate_existing()
        with closing(store._connect()) as connection:
            row = connection.execute(
                "SELECT applied_at FROM schema_migrations WHERE version=?", (SCHEMA_VERSION,)).fetchone()
        self.assertIsNotNone(row, "reader authorization probe requires an initialized integration schema")
        original_applied_at = row[0]
        probe_applied_at = "read-only-authorization-probe"
        if probe_applied_at == original_applied_at:
            probe_applied_at += "-changed"

        server_reader = LibSQLForecastStore(TEST_TURSO_URL, TEST_TURSO_READER_TOKEN, read_only=True)
        server_reader.validate_existing()

        driver = LibSQLForecastStore._load_driver()
        write_rejected = False
        authorization_rejection = False
        with closing(driver.connect(
                database=TEST_TURSO_URL,
                auth_token=TEST_TURSO_READER_TOKEN,
                timeout=30,
                isolation_level=None,
        )) as raw_connection:
            reader_row = raw_connection.execute(
                "SELECT applied_at FROM schema_migrations WHERE version=?", (SCHEMA_VERSION,)).fetchone()
            self.assertIsNotNone(reader_row)
            self.assertEqual(reader_row[0], original_applied_at)
            try:
                raw_connection.execute(
                    "UPDATE schema_migrations SET applied_at=? WHERE version=?",
                    (probe_applied_at, SCHEMA_VERSION))
                raw_connection.commit()
            except (driver.Error, ValueError) as error:
                write_rejected = True
                message = str(error).lower()
                authorization_rejection = any(marker in message for marker in (
                    "write operations are forbidden",
                    "doesn't have write permission",
                    "blocked",
                ))

        with closing(store._connect()) as connection:
            stored_applied_at = connection.execute(
                "SELECT applied_at FROM schema_migrations WHERE version=?", (SCHEMA_VERSION,)).fetchone()[0]
            if stored_applied_at != original_applied_at:
                connection.execute(
                    "UPDATE schema_migrations SET applied_at=? WHERE version=?",
                    (original_applied_at, SCHEMA_VERSION))
                connection.commit()

        self.assertTrue(write_rejected, "reader token unexpectedly permitted a direct database UPDATE")
        self.assertTrue(authorization_rejection,
            "reader token write rejection did not indicate a write-permission authorization failure")
        self.assertEqual(stored_applied_at, original_applied_at,
            "reader token direct UPDATE changed the integration database")


if __name__ == "__main__":
    unittest.main()
