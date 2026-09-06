import hashlib
import json
import math
import os
import sqlite3
import tempfile
import threading
import unittest
import unicodedata
import uuid
from dataclasses import replace
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from app.forecast_ledger import (
    FORECAST_NAMESPACE,
    ForecastStoreConfigurationError,
    ForecastIntegrityError,
    ForecastRecord,
    LedgerDatabaseError,
    LedgerTransientError,
    SCHEMA_VERSION,
    SQLiteForecastStore,
    canonical_feature_schema_json,
    canonical_identity_json,
    create_forecast_store,
    feature_schema_sha256,
    issue_forecast,
    sha256_file,
)
from app.ground_truth_reconciler import AcquisitionBatch, GroundTruthReconciler
from scripts.modeling.features import V1_FEATURE_COLUMNS
from scripts.modeling.train import save_artifact, train_v1_model
from scripts.modeling.features import build_v1_features


class ForecastLedgerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.database = self.root / "nested" / "forecast.sqlite3"
        self.prediction_time = datetime(2025, 2, 2, tzinfo=timezone.utc)
        self.record = ForecastRecord.create(
            sensor_id=13502151,
            prediction_time=self.prediction_time,
            predicted_pm25=42.5,
            persistence_prediction=24.0,
            model_version="v1",
            model_artifact_sha256="a" * 64,
            feature_schema_sha256="b" * 64,
            artifact_version=1,
            feature_configuration="A2",
            source_retrieved_at=self.prediction_time - timedelta(minutes=5),
            input_data_cutoff=self.prediction_time,
            history_start=self.prediction_time - timedelta(hours=24),
            history_end=self.prediction_time - timedelta(hours=1),
            data_mode_at_issue="fresh_openaq",
            freshness_status_at_issue="fresh",
            source_age_minutes_at_issue=0,
            issued_at=self.prediction_time + timedelta(minutes=1),
        )

    def tearDown(self):
        self.directory.cleanup()

    def _forecast(self, prediction_time):
        return ForecastRecord.create(
            sensor_id=13502151,
            prediction_time=prediction_time,
            predicted_pm25=12.0,
            persistence_prediction=14.0,
            model_version="v1",
            model_artifact_sha256="a" * 64,
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
            issued_at=prediction_time + timedelta(minutes=1),
        )

    def _insert_forecast(self, store, target_interval_end):
        forecast = self._forecast(target_interval_end - timedelta(hours=7))
        store.insert(forecast)
        return forecast

    def _evaluate_at(self, store, target_interval_end):
        forecast = self._insert_forecast(store, target_interval_end)
        batch = AcquisitionBatch.create(sensor_id=forecast.sensor_id,
            requested_interval_start=forecast.target_interval_start,
            requested_interval_end=forecast.target_interval_end,
            retrieved_at=forecast.target_interval_end + timedelta(hours=2),
            source_endpoint="/sensors/13502151/hours", http_status=200,
            raw_payload_sha256=(str(forecast.sensor_id) * 64)[:64],
            normalized_payload_sha256="f" * 64, created_at=forecast.target_interval_end + timedelta(hours=2),
            records=[{"sensor_id": forecast.sensor_id, "event_time": forecast.target_interval_start,
                "period_end_utc": forecast.target_interval_end, "value_decimal": Decimal("10"),
                "unit": "µg/m³", "record_id": forecast.sensor_id}])
        GroundTruthReconciler(store, now=lambda: forecast.target_interval_end + timedelta(hours=2)).reconcile(forecast, batch)
        return store.materialize_evaluation(forecast.forecast_id).record

    def _publication_window(self, reference):
        ict = ZoneInfo("Asia/Ho_Chi_Minh")
        publication_date = reference.astimezone(ict).date()
        end = datetime.combine(publication_date, time.min, ict).astimezone(timezone.utc)
        return publication_date, end - timedelta(days=30), end

    def test_store_factory_defaults_to_sqlite_when_backend_is_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            store = create_forecast_store(self.database)
        self.assertIsInstance(store, SQLiteForecastStore)
        self.assertEqual(store.path, self.database)
        self.assertFalse(store.read_only)

    def test_store_factory_accepts_explicit_sqlite(self):
        with patch.dict(os.environ, {"AIRAWARE_LEDGER_BACKEND": "sqlite"}, clear=True):
            store = create_forecast_store(self.database)
        self.assertIsInstance(store, SQLiteForecastStore)

    def test_store_factory_unknown_backend_fails_closed(self):
        with patch.dict(os.environ, {"AIRAWARE_LEDGER_BACKEND": "unknown"}, clear=True), patch(
                "app.forecast_ledger.SQLiteForecastStore") as sqlite_store:
            with self.assertRaisesRegex(ForecastStoreConfigurationError, "unsupported AIRAWARE_LEDGER_BACKEND='unknown'"):
                create_forecast_store(self.database)
        sqlite_store.assert_not_called()

    def test_store_factory_passes_read_only_to_sqlite(self):
        with patch.dict(os.environ, {"AIRAWARE_LEDGER_BACKEND": "sqlite"}, clear=True):
            store = create_forecast_store(self.database, read_only=True)
        self.assertIsInstance(store, SQLiteForecastStore)
        self.assertTrue(store.read_only)
        self.assertEqual(store.path, self.database)

    def test_sqlite_store_translates_runtime_database_errors(self):
        store = SQLiteForecastStore(self.database)
        with patch.object(store, "_connect", side_effect=sqlite3.OperationalError("unable to open database file")):
            with self.assertRaises(LedgerDatabaseError):
                store.latest()
        with patch.object(store, "_connect", side_effect=sqlite3.OperationalError("database is locked")):
            with self.assertRaises(LedgerTransientError):
                store.latest()


    def test_canonical_identity_and_uuid_are_exact_and_unicode_normalized(self):
        record = ForecastRecord.create(**{**self.record.as_dict(), "forecast_id": None, "model_version": "v\u0069\u0301"})
        canonical = canonical_identity_json(record)
        expected = {
            "feature_schema_sha256": "b" * 64,
            "forecast_horizon_hours": 6,
            "identity_serialization_version": 1,
            "model_artifact_sha256": "a" * 64,
            "model_version": unicodedata.normalize("NFC", "v\u0069\u0301"),
            "prediction_time": "2025-02-02T00:00:00Z",
            "sensor_id": 13502151,
            "target_interval_end": "2025-02-02T07:00:00Z",
            "target_interval_start": "2025-02-02T06:00:00Z",
        }
        self.assertEqual(canonical, json.dumps(expected, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        self.assertNotIn("\n", canonical)
        self.assertNotIn("\x00", canonical)
        self.assertEqual(record.forecast_id, str(uuid.uuid5(FORECAST_NAMESPACE, canonical)))

    def test_feature_schema_json_and_hash_match_contract(self):
        expected = {
            "calendar_timezone": "Asia/Ho_Chi_Minh",
            "feature_columns": V1_FEATURE_COLUMNS,
            "feature_configuration": "A2",
            "feature_schema_serialization_version": 1,
            "forecast_horizon_hours": 6,
            "raw_pm25_is_feature": False,
            "target_column": "target_pm25_t_plus_6",
            "weather_is_feature": False,
        }
        canonical = json.dumps(expected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self.assertEqual(canonical_feature_schema_json(), canonical)
        self.assertEqual(feature_schema_sha256(), hashlib.sha256(canonical.encode("utf-8")).hexdigest())

    def test_artifact_hash_uses_exact_file_bytes(self):
        path = self.root / "artifact.bin"
        path.write_bytes(b"x\n\x00\xff")
        self.assertEqual(sha256_file(path), hashlib.sha256(b"x\n\x00\xff").hexdigest())

    def test_validation_rejects_invalid_values_and_normalizes_offsets(self):
        offset = timezone(timedelta(hours=7))
        normalized = ForecastRecord.create(**{**self.record.as_dict(), "forecast_id": None, "prediction_time": datetime(2025, 2, 2, 7, tzinfo=offset)})
        self.assertEqual(normalized.prediction_time, self.prediction_time)
        invalid = (
            {"prediction_time": datetime(2025, 2, 2)},
            {"prediction_time": self.prediction_time + timedelta(minutes=1)},
            {"model_artifact_sha256": "A" * 64},
            {"feature_schema_sha256": "x" * 64},
            {"predicted_pm25": math.inf},
            {"persistence_prediction": math.nan},
            {"source_age_minutes_at_issue": -1},
            {"target_interval_start": self.prediction_time + timedelta(hours=5)},
            {"target_interval_end": self.prediction_time + timedelta(hours=8)},
            {"issuance_mode": "manual"},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ForecastRecord.create(**{**self.record.as_dict(), "forecast_id": None, **changes})

    def test_store_initializes_parent_schema_and_survives_reopen(self):
        store = SQLiteForecastStore(self.database)
        store.initialize()
        result = store.insert(self.record)
        self.assertEqual(result.status, "inserted")
        self.assertEqual(SQLiteForecastStore(self.database).latest(), self.record)
        with sqlite3.connect(self.database) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)

    def test_identical_insert_is_idempotent_but_conflict_never_overwrites(self):
        store = SQLiteForecastStore(self.database)
        store.initialize()
        store.insert(self.record)
        self.assertEqual(store.insert(self.record).status, "already_exists")
        with self.assertRaises(ForecastIntegrityError):
            store.insert(replace(self.record, predicted_pm25=99.0))
        self.assertEqual(store.latest().predicted_pm25, 42.5)

    def test_natural_identity_is_authoritative_even_with_wrong_forecast_id(self):
        store = SQLiteForecastStore(self.database)
        store.initialize()
        store.insert(self.record)
        with self.assertRaises(ForecastIntegrityError):
            store.insert(replace(self.record, forecast_id=str(uuid.uuid4())))
        self.assertEqual(store.count(), 1)

    def test_transaction_failure_has_no_partial_row(self):
        store = SQLiteForecastStore(self.database)
        store.initialize()
        with patch.object(store, "_after_insert", side_effect=RuntimeError("failure")):
            with self.assertRaises(RuntimeError):
                store.insert(self.record)
        self.assertEqual(store.count(), 0)

    def test_concurrent_identical_inserts_create_one_row(self):
        store = SQLiteForecastStore(self.database)
        store.initialize()
        statuses = []
        errors = []
        def insert():
            try:
                statuses.append(store.insert(self.record).status)
            except Exception as error:
                errors.append(error)
        threads = [threading.Thread(target=insert) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertFalse(errors)
        self.assertEqual(store.count(), 1)
        self.assertEqual(statuses.count("inserted"), 1)
        self.assertEqual(statuses.count("already_exists"), 7)

    def test_monitoring_lease_acquire_reacquire_expiry_takeover_and_release(self):
        store = SQLiteForecastStore(self.database)
        store.initialize()
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        lease_name = "production-monitoring"

        self.assertTrue(store.acquire_monitoring_lease(lease_name, "owner-a", now, 60))
        self.assertFalse(store.acquire_monitoring_lease(lease_name, "owner-b", now, 60))
        self.assertTrue(store.acquire_monitoring_lease(
            lease_name, "owner-a", now + timedelta(seconds=10), 60))
        with store._connect() as connection:
            row = connection.execute(
                "SELECT owner_id, acquired_at, expires_at FROM monitoring_leases WHERE lease_name=?",
                (lease_name,)).fetchone()
        self.assertEqual(row["owner_id"], "owner-a")
        self.assertEqual(row["acquired_at"], "2026-01-01T00:00:00Z")
        self.assertEqual(row["expires_at"], "2026-01-01T00:01:10Z")

        takeover = now + timedelta(seconds=70)
        self.assertTrue(store.acquire_monitoring_lease(lease_name, "owner-b", takeover, 60))
        self.assertFalse(store.release_monitoring_lease(lease_name, "owner-a"))
        self.assertTrue(store.release_monitoring_lease(lease_name, "owner-b"))
        self.assertTrue(store.acquire_monitoring_lease(
            lease_name, "owner-a", takeover + timedelta(seconds=1), 60))

    def test_monitoring_lease_validates_identity_time_and_ttl(self):
        store = SQLiteForecastStore(self.database)
        store.initialize()
        aware = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for lease_name, owner_id in (("", "owner"), ("lease", ""), (None, "owner"), ("lease", None)):
            with self.subTest(lease_name=lease_name, owner_id=owner_id), self.assertRaises(ValueError):
                store.acquire_monitoring_lease(lease_name, owner_id, aware, 60)
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            store.acquire_monitoring_lease("lease", "owner", datetime(2026, 1, 1), 60)
        for ttl in (0, -1, 1.5, True):
            with self.subTest(ttl=ttl), self.assertRaises(ValueError):
                store.acquire_monitoring_lease("lease", "owner", aware, ttl)

    def test_concurrent_monitoring_lease_acquisition_has_one_owner(self):
        store = SQLiteForecastStore(self.database)
        store.initialize()
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        barrier = threading.Barrier(8)
        outcomes = []
        errors = []

        def acquire(index):
            try:
                barrier.wait()
                outcomes.append(store.acquire_monitoring_lease(
                    "production-monitoring", f"owner-{index}", now, 60))
            except Exception as error:
                errors.append(error)

        threads = [threading.Thread(target=acquire, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertFalse(errors)
        self.assertEqual(outcomes.count(True), 1)
        self.assertEqual(outcomes.count(False), 7)

    def test_changed_artifact_or_schema_creates_distinct_identity(self):
        one = self.record
        two = ForecastRecord.create(**{**one.as_dict(), "forecast_id": None, "model_artifact_sha256": "c" * 64})
        three = ForecastRecord.create(**{**one.as_dict(), "forecast_id": None, "feature_schema_sha256": "d" * 64})
        store = SQLiteForecastStore(self.database)
        store.initialize()
        for record in (one, two, three):
            store.insert(record)
        self.assertEqual(store.count(), 3)
        self.assertEqual(len({one.forecast_id, two.forecast_id, three.forecast_id}), 3)

    def test_hard_coded_uuid_fixture(self):
        self.assertEqual(self.record.forecast_id, "d4366a57-69f9-5e62-a6f9-1c549c624b63")

    def test_hard_coded_feature_schema_hash(self):
        self.assertEqual(feature_schema_sha256(), "a6b1307497011bd8f54dae9a6b198938de2175d5c2843445a3c474a2c0447049")

    def test_feature_order_mutation_changes_hash(self):
        schema = json.loads(canonical_feature_schema_json())
        schema["feature_columns"] = list(reversed(schema["feature_columns"]))
        changed = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self.assertNotEqual(hashlib.sha256(changed.encode()).hexdigest(), feature_schema_sha256())

    def test_identity_field_order_does_not_affect_canonical_json(self):
        expected = json.loads(canonical_identity_json(self.record))
        reversed_payload = dict(reversed(list(expected.items())))
        self.assertEqual(json.dumps(reversed_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")), canonical_identity_json(self.record))

    def test_canonical_timestamps_are_exact_utc_seconds(self):
        canonical = canonical_identity_json(self.record)
        self.assertIn('"prediction_time":"2025-02-02T00:00:00Z"', canonical)
        self.assertNotIn("+00:00", canonical)

    def test_schema_declares_all_immutable_columns_primary_key_and_natural_unique(self):
        store = SQLiteForecastStore(self.database)
        store.initialize()
        with sqlite3.connect(self.database) as connection:
            columns = {row[1]: row for row in connection.execute("PRAGMA table_info(forecasts)")}
            indexes = list(connection.execute("PRAGMA index_list(forecasts)"))
            unique_columns = [tuple(row[2] for row in connection.execute(f"PRAGMA index_info({index[1]})")) for index in indexes if index[2]]
        self.assertEqual(set(columns), set(self.record.as_dict()))
        self.assertEqual(columns["forecast_id"][5], 1)
        self.assertIn(("sensor_id", "prediction_time", "target_interval_start", "target_interval_end", "forecast_horizon_hours", "model_version", "model_artifact_sha256", "feature_schema_sha256"), unique_columns)

    def test_get_by_id_reads_stored_record(self):
        store = SQLiteForecastStore(self.database); store.initialize(); store.insert(self.record)
        self.assertEqual(store.get_by_id(self.record.forecast_id), self.record)

    def test_get_by_id_missing_returns_none(self):
        store = SQLiteForecastStore(self.database); store.initialize()
        self.assertIsNone(store.get_by_id(str(uuid.uuid4())))

    def test_get_by_identity_reads_stored_record(self):
        store = SQLiteForecastStore(self.database); store.initialize(); store.insert(self.record)
        self.assertEqual(store.get_by_identity(self.record), self.record)

    def test_get_by_identity_missing_returns_none(self):
        store = SQLiteForecastStore(self.database); store.initialize()
        self.assertIsNone(store.get_by_identity(self.record))

    def test_identical_insert_returns_stored_not_caller_record(self):
        store = SQLiteForecastStore(self.database); store.initialize(); stored = store.insert(self.record).record
        caller = replace(self.record)
        result = store.insert(caller)
        self.assertEqual(result.status, "already_exists")
        self.assertEqual(result.record, stored)
        self.assertIsNot(result.record, caller)

    def test_create_rejects_supplied_wrong_forecast_id(self):
        with self.assertRaises(ForecastIntegrityError):
            ForecastRecord.create(**{**self.record.as_dict(), "forecast_id": str(uuid.uuid4())})

    def test_create_rejects_nonpositive_or_noninteger_sensor(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ForecastRecord.create(**{**self.record.as_dict(), "forecast_id": None, "sensor_id": value})

    def test_create_rejects_empty_identity_strings(self):
        for field in ("model_version", "unit", "feature_configuration", "data_mode_at_issue", "freshness_status_at_issue"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                ForecastRecord.create(**{**self.record.as_dict(), "forecast_id": None, field: ""})

    def test_create_rejects_input_cutoff_not_prediction_time(self):
        with self.assertRaises(ValueError):
            ForecastRecord.create(**{**self.record.as_dict(), "forecast_id": None, "input_data_cutoff": self.prediction_time - timedelta(hours=1)})

    def test_create_rejects_history_end_relationship(self):
        with self.assertRaises(ValueError):
            ForecastRecord.create(**{**self.record.as_dict(), "forecast_id": None, "history_end": self.prediction_time})

    def test_create_rejects_history_start_relationship(self):
        with self.assertRaises(ValueError):
            ForecastRecord.create(**{**self.record.as_dict(), "forecast_id": None, "history_start": self.prediction_time - timedelta(hours=23)})

    def test_create_rejects_noncanonical_second_timestamp(self):
        with self.assertRaises(ValueError):
            ForecastRecord.create(**{**self.record.as_dict(), "forecast_id": None, "issued_at": self.record.issued_at + timedelta(microseconds=1)})

    @unittest.skipUnless(Path(".artifacts/models/airaware_v1.joblib").is_file(), "production artifact unavailable")
    def test_production_artifact_hash_uses_exact_bytes(self):
        path = Path(".artifacts/models/airaware_v1.joblib")
        self.assertEqual(sha256_file(path), hashlib.sha256(path.read_bytes()).hexdigest())

    def test_issuer_not_eligible_creates_no_row(self):
        current_path = self.root / "current.json"
        current_path.write_text(json.dumps({"artifact_version": 1, "sensor_id": 13502151, "retrieved_at": self.prediction_time.isoformat(), "normalized_records": []}), encoding="utf-8")
        with patch("scripts.modeling.predict.load_artifact", return_value=(object(), {"feature_configuration": "A2"})), patch("app.main._validate_metadata"), patch("app.main._load_current_artifact", return_value=({"sensor_id": 13502151}, self.prediction_time)), patch("app.main._current_prediction_request", side_effect=ValueError("no completed hourly intervals")):
            result = issue_forecast(self.database, self.root / "model", current_path, now=lambda: self.prediction_time)
        self.assertEqual(result.outcome, "not_eligible")
        store = SQLiteForecastStore(self.database); store.initialize()
        self.assertEqual(store.count(), 0)

    def test_issuer_issues_then_reports_already_exists_and_failure_writes_nothing(self):
        source = pd.DataFrame({"event_time": pd.date_range("2025-01-01T00:00:00Z", periods=60, freq="1h"), "pm25": [float(value) for value in range(60)]})
        model, metadata = train_v1_model(build_v1_features(source, include_target=True).dropna())
        model_path = self.root / "model.joblib"
        save_artifact(model_path, model, metadata)
        current_path = self.root / "current.json"
        history = [{"event_time": (self.prediction_time - timedelta(hours=24-index)).isoformat(), "period_end_utc": (self.prediction_time - timedelta(hours=23-index)).isoformat(), "record_id": index, "sensor_id": 13502151, "unit": "µg/m³", "value": float(index + 1)} for index in range(24)]
        current_path.write_text(json.dumps({"artifact_version": 1, "sensor_id": 13502151, "retrieved_at": self.prediction_time.isoformat(), "normalized_records": history}), encoding="utf-8")
        first = issue_forecast(self.database, model_path, current_path, now=lambda: self.prediction_time + timedelta(minutes=1))
        second = issue_forecast(self.database, model_path, current_path, now=lambda: self.prediction_time + timedelta(minutes=1))
        self.assertEqual((first.outcome, second.outcome), ("issued", "already_exists"))
        self.assertEqual(first.record.persistence_prediction, 24.0)
        bad_database = self.root / "bad.sqlite3"
        failed = issue_forecast(bad_database, self.root / "missing", current_path, now=lambda: self.prediction_time)
        self.assertEqual(failed.outcome, "failed")
        store = SQLiteForecastStore(bad_database)
        store.initialize()
        self.assertEqual(store.count(), 0)

    def test_consumer_publication_empty_history_is_idempotent(self):
        store = SQLiteForecastStore(self.database)
        store.initialize()
        reference = datetime(2025, 2, 2, 20, 15, tzinfo=timezone.utc)
        first = store.publish_consumer_performance(
            "v1", "a" * 64, "b" * 64, 13502151, reference, reference)
        second = store.publish_consumer_performance(
            "v1", "a" * 64, "b" * 64, 13502151, reference, reference + timedelta(minutes=1))
        self.assertEqual(first, second)
        self.assertEqual((first.available, first.reason, first.verified_count,
                          first.mature_issued_count), (False, "insufficient_history", 0, 0))
        self.assertIsNone(first.snapshot_id)

    def test_consumer_publication_selects_and_persists_in_one_connection(self):
        class ConnectionCountingStore(SQLiteForecastStore):
            connection_count = 0

            def _connect(inner_self):
                inner_self.connection_count += 1
                return super(ConnectionCountingStore, inner_self)._connect()

        store = ConnectionCountingStore(self.database)
        store.initialize()
        store.connection_count = 0
        reference = datetime(2025, 2, 2, 20, 15, tzinfo=timezone.utc)
        store.publish_consumer_performance(
            "v1", "a" * 64, "b" * 64, 13502151, reference, reference)
        self.assertEqual(store.connection_count, 1)

    def test_publication_window_includes_exact_start_and_excludes_exact_end(self):
        store = SQLiteForecastStore(self.database)
        store.initialize()
        reference = datetime(2026, 1, 5, 20, 0, tzinfo=timezone.utc)
        publication_date, start, end = self._publication_window(reference)
        self._evaluate_at(store, start)
        self._evaluate_at(store, end)
        publication = store.publish_consumer_performance(
            "v1", "a" * 64, "b" * 64, 13502151, reference, reference)
        self.assertEqual(publication.publication_date, publication_date.isoformat())
        self.assertEqual(publication.verified_count, 1)
        self.assertEqual(publication.mature_issued_count, 1)
        self.assertIsNotNone(publication.snapshot_id)

    def test_publication_window_excludes_boundaries_just_outside_start_and_end(self):
        store = SQLiteForecastStore(self.database)
        store.initialize()
        reference = datetime(2026, 1, 5, 20, 0, tzinfo=timezone.utc)
        _, start, end = self._publication_window(reference)
        self._evaluate_at(store, start - timedelta(hours=1))
        self._evaluate_at(store, end + timedelta(hours=1))
        publication = store.publish_consumer_performance(
            "v1", "a" * 64, "b" * 64, 13502151, reference, reference)
        self.assertEqual(publication.verified_count, 0)
        self.assertEqual(publication.mature_issued_count, 0)

    def test_publication_mature_denominator_cuts_off_at_reference_minus_120_minutes(self):
        store = SQLiteForecastStore(self.database)
        store.initialize()
        reference = datetime(2026, 1, 9, 17, 0, tzinfo=timezone.utc)
        cutoff = reference - timedelta(minutes=120)
        _, start, _ = self._publication_window(reference)
        self._insert_forecast(store, cutoff - timedelta(hours=1))
        self._insert_forecast(store, cutoff)
        self._insert_forecast(store, cutoff + timedelta(hours=1))
        self._insert_forecast(store, start - timedelta(hours=1))
        publication = store.publish_consumer_performance(
            "v1", "a" * 64, "b" * 64, 13502151, reference, reference)
        self.assertEqual(publication.mature_issued_count, 2)
        self.assertEqual(publication.verified_count, 0)


    def _populate_verified(self, store, count, reference):
        _, start, _ = self._publication_window(reference)
        for index in range(count):
            self._evaluate_at(store, start + timedelta(hours=index))
        return start

    def test_consumer_publication_47_vs_48_verified_threshold(self):
        store = SQLiteForecastStore(self.database)
        store.initialize()
        reference = datetime(2026, 1, 5, 20, 0, tzinfo=timezone.utc)
        start = self._populate_verified(store, 47, reference)
        below = store.publish_consumer_performance(
            "v1", "a" * 64, "b" * 64, 13502151, reference, reference, minimum_verified_count=48)
        self.assertEqual(below.verified_count, 47)
        self.assertFalse(below.available)
        self.assertEqual(below.reason, "insufficient_history")
        self.assertIsNone(below.model_mae)
        self.assertIsNone(below.persistence_mae)
        self.assertIsNone(below.mae_difference)
        self.assertIsNotNone(below.snapshot_id)
        self._evaluate_at(store, start + timedelta(hours=47))
        above = store.publish_consumer_performance(
            "v1", "a" * 64, "b" * 64, 13502151, reference, reference, minimum_verified_count=48)
        self.assertEqual(above.verified_count, 48)
        self.assertTrue(above.available)
        self.assertIsNone(above.reason)
        self.assertAlmostEqual(above.model_mae, 2.0)
        self.assertAlmostEqual(above.persistence_mae, 4.0)
        self.assertAlmostEqual(above.mae_difference, 2.0)
        self.assertIsNotNone(above.snapshot_id)

    def test_consumer_publication_rejects_verified_exceeding_mature(self):
        store = SQLiteForecastStore(self.database)
        store.initialize()
        reference = datetime(2026, 1, 9, 17, 30, tzinfo=timezone.utc)
        _, start, end = self._publication_window(reference)
        cutoff = reference - timedelta(minutes=120)
        self.assertGreater(cutoff, start)
        self.assertLess(cutoff, end)
        self._evaluate_at(store, cutoff - timedelta(minutes=30))
        self._evaluate_at(store, cutoff + timedelta(minutes=30))
        with self.assertRaises(ForecastIntegrityError):
            store.publish_consumer_performance(
                "v1", "a" * 64, "b" * 64, 13502151, reference, reference)

    def test_consumer_publication_identical_repeat_same_id_no_duplicate(self):
        store = SQLiteForecastStore(self.database)
        store.initialize()
        reference = datetime(2026, 1, 5, 20, 0, tzinfo=timezone.utc)
        self._populate_verified(store, 48, reference)
        first = store.publish_consumer_performance(
            "v1", "a" * 64, "b" * 64, 13502151, reference, reference, minimum_verified_count=48)
        second = store.publish_consumer_performance(
            "v1", "a" * 64, "b" * 64, 13502151, reference, reference + timedelta(minutes=1), minimum_verified_count=48)
        self.assertEqual(first, second)
        self.assertEqual(first.publication_id, second.publication_id)
        with sqlite3.connect(self.database) as connection:
            count = connection.execute("SELECT COUNT(*) FROM consumer_performance_publications WHERE publication_id=?",
                (first.publication_id,)).fetchone()[0]
        self.assertEqual(count, 1)

    def test_consumer_publication_changed_membership_new_id_old_row_immutable(self):
        store = SQLiteForecastStore(self.database)
        store.initialize()
        reference = datetime(2026, 1, 5, 20, 0, tzinfo=timezone.utc)
        start = self._populate_verified(store, 48, reference)
        earlier = store.publish_consumer_performance(
            "v1", "a" * 64, "b" * 64, 13502151, reference, reference, minimum_verified_count=48)
        self._evaluate_at(store, start + timedelta(hours=48))
        later = store.publish_consumer_performance(
            "v1", "a" * 64, "b" * 64, 13502151, reference, reference, minimum_verified_count=48)
        self.assertEqual(later.verified_count, 49)
        self.assertNotEqual(earlier.membership_sha256, later.membership_sha256)
        self.assertNotEqual(earlier.publication_id, later.publication_id)
        with sqlite3.connect(self.database) as connection:
            rows = {row[0] for row in connection.execute("SELECT publication_id FROM consumer_performance_publications")}
            stored = connection.execute("SELECT verified_count, membership_sha256 FROM consumer_performance_publications WHERE publication_id=?",
                (earlier.publication_id,)).fetchone()
        self.assertEqual(len(rows), 2)
        self.assertIn(earlier.publication_id, rows)
        self.assertIn(later.publication_id, rows)
        self.assertEqual(tuple(stored), (48, earlier.membership_sha256))

    def test_current_consumer_performance_ordering_deterministic(self):
        store = SQLiteForecastStore(self.database)
        store.initialize()
        reference = datetime(2026, 1, 5, 20, 0, tzinfo=timezone.utc)
        start = self._populate_verified(store, 48, reference)
        store.publish_consumer_performance(
            "v1", "a" * 64, "b" * 64, 13502151, reference, reference, minimum_verified_count=48)
        self._evaluate_at(store, start + timedelta(hours=48))
        latest = store.publish_consumer_performance(
            "v1", "a" * 64, "b" * 64, 13502151, reference, reference, minimum_verified_count=48)
        current = store.current_consumer_performance("v1", "a" * 64, "b" * 64, 13502151)
        self.assertEqual(current.publication_id, latest.publication_id)
        self.assertEqual(current.verified_count, 49)
        self.assertEqual(current, store.current_consumer_performance("v1", "a" * 64, "b" * 64, 13502151))

    def test_consumer_publication_metrics_match_selected_cohort(self):
        store = SQLiteForecastStore(self.database)
        store.initialize()
        reference = datetime(2026, 1, 5, 20, 0, tzinfo=timezone.utc)
        _, start, end = self._publication_window(reference)
        self._populate_verified(store, 48, reference)
        publication = store.publish_consumer_performance(
            "v1", "a" * 64, "b" * 64, 13502151, reference, reference, minimum_verified_count=48)
        metrics = store.evaluation_metrics("v1", "a" * 64, "b" * 64, 13502151, start, end)
        self.assertEqual(metrics.count, publication.verified_count)
        self.assertEqual(metrics.model_mae, publication.model_mae)
        self.assertEqual(metrics.persistence_mae, publication.persistence_mae)
        self.assertEqual(metrics.mae_improvement, publication.mae_difference)


if __name__ == "__main__":
    unittest.main()
