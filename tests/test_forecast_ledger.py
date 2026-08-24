import hashlib
import json
import math
import sqlite3
import tempfile
import threading
import unittest
import unicodedata
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from app.forecast_ledger import (
    FORECAST_NAMESPACE,
    ForecastIntegrityError,
    ForecastRecord,
    SQLiteForecastStore,
    canonical_feature_schema_json,
    canonical_identity_json,
    feature_schema_sha256,
    issue_forecast,
    sha256_file,
)
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
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)

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


if __name__ == "__main__":
    unittest.main()
