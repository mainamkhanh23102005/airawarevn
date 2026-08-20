import json
import os
import subprocess
import sys
import tempfile
import unittest
import io
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import httpx

from scripts import openaq, openmeteo


UTC = timezone.utc
ROOT = Path(__file__).resolve().parents[1]


def response(results, found=None):
    meta = {} if found is None else {"found": found}
    return httpx.Response(200, content=json.dumps({"results": results, "meta": meta}, separators=(",", ":")).encode())


def hour_row(hour="2024-01-01T00:00:00Z", value=12.5, row_id=1):
    return {"id": row_id, "value": value, "parameter": {"id": 2, "name": "pm25", "units": "µg/m³"}, "period": {"datetimeFrom": {"utc": hour}, "datetimeTo": {"utc": hour.replace("00:00:00Z", "01:00:00Z")}}}


class DiscoveryTests(unittest.TestCase):
    def test_discovery_request_and_all_metadata_fields(self):
        requests = []
        location = {"id": 7, "name": "Hanoi", "timezone": "Asia/Ho_Chi_Minh", "provider": {"id": 4, "name": "Provider"}, "owner": {"id": 5, "name": "Owner"}, "isMobile": False, "isMonitor": True, "coordinates": {"latitude": 21.03, "longitude": 105.85}, "sensors": [{"id": 9, "parameter": {"id": 2, "name": "pm25", "units": "µg/m³"}, "datetimeFirst": {"utc": "2023-01-01T00:00:00Z"}, "datetimeLast": {"utc": "2024-01-01T00:00:00Z"}}]}

        def handler(request):
            requests.append(request)
            return response([location], 1)

        with tempfile.TemporaryDirectory() as directory, httpx.Client(transport=httpx.MockTransport(handler)) as client:
            result = openaq.discover(client, "secret", Path(directory), now=lambda: datetime(2025, 1, 1, tzinfo=UTC))

        item = result[0]
        self.assertEqual(item, {"location_id": 7, "location_name": "Hanoi", "timezone": "Asia/Ho_Chi_Minh", "provider": {"id": 4, "name": "Provider"}, "owner": {"id": 5, "name": "Owner"}, "isMobile": False, "isMonitor": True, "coordinates": {"latitude": 21.03, "longitude": 105.85}, "sensor_id": 9, "parameter_id": 2, "parameter_name": "pm25", "units": "µg/m³", "datetimeFirst": {"utc": "2023-01-01T00:00:00Z"}, "datetimeLast": {"utc": "2024-01-01T00:00:00Z"}})
        self.assertEqual(requests[0].url.params["coordinates"], "21.0285,105.8542")
        self.assertEqual(requests[0].url.params["radius"], "25000")
        self.assertEqual(requests[0].url.params["parameters_id"], "2")
        self.assertNotIn("secret", json.dumps(item))

    def test_cli_exact_subcommands_and_environment_key(self):
        completed = subprocess.run([sys.executable, "scripts/run_data_spike.py", "discovery", "--openaq-api-key", "secret"], cwd=ROOT, text=True, capture_output=True, env={**os.environ, "OPENAQ_API_KEY": "secret"})
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("unrecognized arguments", completed.stderr)
        help_result = subprocess.run([sys.executable, "scripts/run_data_spike.py", "coverage", "--help"], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("--sensor-id", help_result.stdout)
        self.assertNotIn("--datetime-from", help_result.stdout)


class FetchTests(unittest.TestCase):
    def test_numeric_found_is_ignored_and_full_page_requests_next_empty_page(self):
        pages = []
        def handler(request):
            page = int(request.url.params["page"])
            pages.append(page)
            return response([hour_row(row_id=page)] if page == 1 else [], 1)
        with tempfile.TemporaryDirectory() as directory, httpx.Client(transport=httpx.MockTransport(handler)) as client:
            records, _ = openaq.fetch_hours(client, "key", 9, datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 2, 1, tzinfo=UTC), Path(directory), limit=1)
        self.assertEqual(pages, [1, 2])
        self.assertEqual(len(records), 1)

    def test_nonnumeric_found_uses_full_then_empty_page_fallback(self):
        pages = []
        def handler(request):
            page = int(request.url.params["page"])
            pages.append(page)
            return response([hour_row(row_id=page)] if page == 1 else [], "unknown")
        with tempfile.TemporaryDirectory() as directory, httpx.Client(transport=httpx.MockTransport(handler)) as client:
            records, _ = openaq.fetch_hours(client, "key", 9, datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 2, 1, tzinfo=UTC), Path(directory), limit=1)
        self.assertEqual(pages, [1, 2])
        self.assertEqual(len(records), 1)

    def test_normalization_uses_period_utc_and_includes_sensor_id(self):
        record = openaq.normalize_measurement(hour_row(), 9)
        self.assertEqual(record["sensor_id"], 9)
        self.assertEqual(record["event_time"], datetime(2024, 1, 1, tzinfo=UTC))
        self.assertEqual(record["period_end_utc"], datetime(2024, 1, 1, 1, tzinfo=UTC))

    def test_provenance_has_exact_fields_and_raw_exact_bytes(self):
        payload = b'{"results":[],"meta":{"found":0}}'
        with tempfile.TemporaryDirectory() as directory, httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=payload))) as client:
            _, provenance = openaq.fetch_hours(client, "secret", 9, datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 2, 1, tzinfo=UTC), Path(directory), now=lambda: datetime(2025, 1, 1, tzinfo=UTC))
            item = provenance[0]
            self.assertEqual(set(item), {"endpoint", "sensor_id", "page", "limit", "datetime_from", "datetime_to", "retrieved_at", "http_status", "raw_payload_path", "raw_payload_sha256"})
            self.assertEqual(Path(item["raw_payload_path"]).read_bytes(), payload)
            self.assertNotIn("secret", json.dumps(item))

    def test_valid_exact_cache_reused_without_network(self):
        calls = Mock(return_value=response([], 0))
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            with httpx.Client(transport=httpx.MockTransport(calls)) as client:
                args = (client, "key", 9, datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 2, 1, tzinfo=UTC), raw)
                openaq.fetch_hours(*args)
                openaq.fetch_hours(*args)
        self.assertEqual(calls.call_count, 1)

    def test_corrupt_mismatched_and_legacy_cache_are_not_reused(self):
        cases = ("corrupt", "mismatch", "legacy")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                raw = Path(directory)
                calls = Mock(return_value=response([], 0))
                with httpx.Client(transport=httpx.MockTransport(calls)) as client:
                    args = (client, "key", 9, datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 2, 1, tzinfo=UTC), raw)
                    openaq.fetch_hours(*args)
                    manifest = next(raw.glob("*.provenance.json"))
                    data = json.loads(manifest.read_text())
                    if case == "corrupt":
                        Path(data["raw_payload_path"]).write_bytes(b"bad")
                    elif case == "mismatch":
                        data["page"] = 99
                        manifest.write_text(json.dumps(data))
                    else:
                        data.pop("http_status")
                        manifest.write_text(json.dumps(data))
                    openaq.fetch_hours(*args)
                self.assertEqual(calls.call_count, 2)

    def test_transient_failures_sleep_one_two_four_then_explicit_failure(self):
        sleep = Mock()
        client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(503)))
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(openaq.OpenAQError, "after retries"):
            openaq.discover(client, "secret", Path(directory), sleep=sleep)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2, 4])
        client.close()

    def test_nontransient_failure_is_explicit_not_empty(self):
        client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(401)))
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(openaq.OpenAQError, "HTTP 401"):
            openaq.discover(client, "secret", Path(directory))
        client.close()


class CoverageTests(unittest.TestCase):
    def test_sensor_metadata_enriches_compact_discovery_sensor_for_arbitrary_id(self):
        compact = {"sensor_id": 24680, "parameter_name": "pm25", "datetimeFirst": None, "datetimeLast": None}
        authoritative = {"id": 24680, "datetimeFirst": {"utc": "2023-01-15T00:00:00Z"}, "datetimeLast": {"utc": "2023-03-02T00:00:00Z"}}
        requests = []

        def handler(request):
            requests.append(request)
            return response([authoritative], 1)

        with tempfile.TemporaryDirectory() as directory, httpx.Client(transport=httpx.MockTransport(handler)) as client:
            enriched = openaq.enrich_sensor_metadata(client, "key", compact, Path(directory))

        self.assertEqual(requests[0].url.path, "/v3/sensors/24680")
        self.assertEqual(enriched["datetimeFirst"], authoritative["datetimeFirst"])
        self.assertEqual(enriched["datetimeLast"], authoritative["datetimeLast"])
        self.assertEqual(openaq.sensor_history_chunks(enriched)[0][0], datetime(2023, 1, 15, tzinfo=UTC))

    def test_coverage_chunks_use_enriched_sensor_metadata(self):
        compact = {"sensor_id": 86420, "datetimeFirst": None, "datetimeLast": None}
        authoritative = {"id": 86420, "datetimeFirst": {"utc": "2024-04-10T00:00:00Z"}, "datetimeLast": {"utc": "2024-05-03T00:00:00Z"}}
        with tempfile.TemporaryDirectory() as directory, httpx.Client(transport=httpx.MockTransport(lambda request: response([authoritative], 1))) as client:
            chunks = openaq.sensor_history_chunks_for_coverage(client, "key", compact, Path(directory))
        self.assertEqual(chunks[0][0], datetime(2024, 4, 10, tzinfo=UTC))
        self.assertEqual(chunks[-1][1], datetime(2024, 5, 3, tzinfo=UTC))

    def test_sensor_metadata_without_authoritative_history_bounds_is_explicit(self):
        compact = {"sensor_id": 24680, "datetimeFirst": None, "datetimeLast": None}
        with tempfile.TemporaryDirectory() as directory, httpx.Client(transport=httpx.MockTransport(lambda request: response([{"id": 24680}], 1))) as client:
            enriched = openaq.enrich_sensor_metadata(client, "key", compact, Path(directory))
        with self.assertRaisesRegex(openaq.OpenAQError, "lacks datetimeFirst or datetimeLast"):
            openaq.sensor_history_chunks(enriched)

    def test_history_bounds_derive_from_sensor_metadata_and_chunks_are_hanoi_months(self):
        metadata = {"datetimeFirst": {"utc": "2023-01-15T00:00:00Z"}, "datetimeLast": {"utc": "2023-03-02T00:00:00Z"}}
        chunks = openaq.sensor_history_chunks(metadata)
        self.assertEqual(chunks[0][0], datetime(2023, 1, 15, tzinfo=UTC))
        self.assertEqual(chunks[-1][1], datetime(2023, 3, 2, tzinfo=UTC))
        self.assertEqual(chunks[0][1], datetime(2023, 1, 31, 17, tzinfo=UTC))

    def test_report_derives_source_range_and_all_duplicate_gap_winter_diagnostics(self):
        rows = [openaq.normalize_measurement(hour_row(row_id=1), 9), openaq.normalize_measurement(hour_row(value=13, row_id=2), 9)]
        rows[1]["period_end_utc"] = datetime(2024, 1, 1, 2, tzinfo=UTC)
        with tempfile.TemporaryDirectory() as directory:
            report = json.loads(openaq.write_coverage_report(rows, Path(directory)).read_text())
        self.assertEqual(report["source_date_range"]["start_utc"], "2024-01-01T00:00:00+00:00")
        self.assertEqual(report["raw_row_count"], 2)
        self.assertEqual(report["unique_canonical_key_count"], 1)
        self.assertEqual(report["duplicate_row_count"], 1)
        self.assertEqual(report["duplicate_hour_count"], 1)
        self.assertEqual(report["ambiguous_conflict_count"], 1)
        self.assertIn("hourly_timestamp_completeness_pct", report["coverage"])
        self.assertIn("usable_pm25_coverage_pct", report["coverage"])
        self.assertIn("longest_hours", report["gaps"])
        self.assertIn("overall_status", report)

    def test_coverage_persists_weather_compatible_exact_normalized_records_and_formats_paths(self):
        start = datetime(2022, 11, 1, tzinfo=openaq.HANOI).astimezone(UTC)
        end = datetime(2023, 11, 1, tzinfo=openaq.HANOI).astimezone(UTC)
        records = [{"sensor_id": 13502151, "event_time": start, "period_end_utc": start.replace(hour=18), "value": 10, "unit": "µg/m³", "record_id": 7}]
        metadata = {"sensor_id": 13502151, "coordinates": {"latitude": 21.0031, "longitude": 105.7947}}
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            paths = openaq.write_coverage_artifacts(records, metadata, Path(directory), source_start=start, source_end=end, output=output)
            normalized = json.loads(paths["normalized_path"].read_text())
            loaded = openmeteo.load_pm25_artifact(paths["normalized_path"], 13502151)
        self.assertEqual(normalized["sensor_metadata"], metadata)
        self.assertEqual(normalized["normalized_records"], [{**records[0], "event_time": start.isoformat(), "period_end_utc": start.replace(hour=18).isoformat()}])
        self.assertEqual(normalized["normalized_records"][0]["event_time"], "2022-10-31T17:00:00+00:00")
        self.assertEqual(normalized["frozen_candidate"], {"start_utc": start.isoformat(), "end_utc": start.replace(hour=18).isoformat()})
        self.assertEqual(loaded.records[0]["event_time"], start)
        for label in ("Sensor:", "Source date range:", "Frozen candidate:", "Expected hours:", "Observed hours:", "Valid PM2.5 hours:", "Timestamp completeness:", "Usable PM2.5 coverage:", "Winter completeness:", "Winter usable coverage:", "Longest gap:", "PM2.5 gate:", "Normalized artifact path:", "Coverage artifact path:"):
            self.assertIn(label, output.getvalue())
        self.assertIn(str(paths["normalized_path"]), output.getvalue())

    def test_winter_reports_percentages_and_explicit_status(self):
        start = datetime(2022, 11, 1, tzinfo=openaq.HANOI).astimezone(UTC)
        end = datetime(2023, 11, 1, tzinfo=openaq.HANOI).astimezone(UTC)
        rows = [{"sensor_id": 9, "event_time": start, "period_end_utc": end, "value": 10, "unit": "µg/m³", "record_id": 1}]
        with tempfile.TemporaryDirectory() as directory:
            report = json.loads(openaq.write_coverage_report(rows, Path(directory), source_start=start, source_end=end).read_text())
        winter = report["winter"][0]
        self.assertIn("hourly_timestamp_completeness_pct", winter)
        self.assertIn("usable_pm25_coverage_pct", winter)
        self.assertEqual(winter["status"], "FAIL")
        self.assertEqual(report["overall_status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
