import io
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import httpx

from scripts import openmeteo

UTC = timezone.utc
ROOT = Path(__file__).resolve().parents[1]
VARIABLES = openmeteo.CORE_WEATHER_VARIABLES


def artifact(sensor_id=13502151, coordinates=None):
    return {
        "sensor_metadata": {"sensor_id": sensor_id, "coordinates": coordinates or {"latitude": 21.0031, "longitude": 105.7947}},
        "frozen_candidate": {"start_utc": "2024-01-01T00:00:00Z", "end_utc": "2024-01-01T03:00:00Z"},
        "normalized_records": [
            {"sensor_id": sensor_id, "event_time": "2024-01-01T00:00:00Z", "value": 10, "unit": "µg/m³", "record_id": "a"},
            {"sensor_id": sensor_id, "event_time": "2024-01-01T01:00:00Z", "value": 11, "unit": "µg/m³", "record_id": "b"},
        ],
    }


def payload(times=None, timezone_name="UTC", offset=0, values=None):
    times = times or ["2024-01-01T00:00", "2024-01-01T01:00", "2024-01-01T02:00"]
    hourly = {"time": times}
    for index, variable in enumerate(VARIABLES):
        hourly[variable] = list((values or {}).get(variable, [index + 1] * len(times)))
    return {"timezone": timezone_name, "utc_offset_seconds": offset, "hourly_units": {variable: "unit" for variable in VARIABLES}, "hourly": hourly}


class ArtifactTests(unittest.TestCase):
    def test_loads_explicit_frozen_candidate_and_aware_utc_records(self):
        loaded = openmeteo.load_pm25_artifact(artifact(), 13502151)
        self.assertEqual(loaded.start, datetime(2024, 1, 1, tzinfo=UTC))
        self.assertEqual(loaded.end, datetime(2024, 1, 1, 3, tzinfo=UTC))
        self.assertEqual(loaded.records[0]["event_time"].tzinfo, UTC)

    def test_accepts_explicit_frozen_interval_shape(self):
        data = artifact()
        data["frozen_interval"] = data.pop("frozen_candidate")
        self.assertEqual(openmeteo.load_pm25_artifact(data, 13502151).end.hour, 3)

    def test_missing_or_mismatched_interval_fails_explicitly(self):
        missing = artifact()
        missing.pop("frozen_candidate")
        mismatch = artifact()
        mismatch["frozen_interval"] = {"start_utc": "2024-01-01T00:00:00Z", "end_utc": "2024-01-02T00:00:00Z"}
        with self.assertRaisesRegex(openmeteo.OpenMeteoError, "frozen interval"):
            openmeteo.load_pm25_artifact(missing, 13502151)
        with self.assertRaisesRegex(openmeteo.OpenMeteoError, "mismatch"):
            openmeteo.load_pm25_artifact(mismatch, 13502151)

    def test_rejects_naive_pm25_timestamp(self):
        data = artifact()
        data["normalized_records"][0]["event_time"] = "2024-01-01T00:00:00"
        with self.assertRaisesRegex(openmeteo.OpenMeteoError, "aware UTC"):
            openmeteo.load_pm25_artifact(data, 13502151)

    def test_sensor_id_and_coordinates_must_match_artifact(self):
        with self.assertRaisesRegex(openmeteo.OpenMeteoError, "sensor ID"):
            openmeteo.load_pm25_artifact(artifact(), 10)
        with self.assertRaisesRegex(openmeteo.OpenMeteoError, "coordinates"):
            openmeteo.load_pm25_artifact(artifact(coordinates={"latitude": 1, "longitude": 2}), 13502151)

    def test_real_sensor_metadata_coordinate_shape_accepts_float_representation_noise(self):
        data = artifact(coordinates={"latitude": 21.0031, "longitude": 105.79470000000002})
        loaded = openmeteo.load_pm25_artifact(data, 13502151)
        self.assertEqual(loaded.sensor_id, 13502151)

    def test_genuinely_different_longitude_fails(self):
        data = artifact(coordinates={"latitude": 21.0031, "longitude": 105.7957})
        with self.assertRaisesRegex(openmeteo.OpenMeteoError, "coordinates"):
            openmeteo.load_pm25_artifact(data, 13502151)

    def test_genuinely_different_latitude_fails(self):
        data = artifact(coordinates={"latitude": 21.0041, "longitude": 105.7947})
        with self.assertRaisesRegex(openmeteo.OpenMeteoError, "coordinates"):
            openmeteo.load_pm25_artifact(data, 13502151)

    def test_duplicate_resolution_yields_unique_valid_hours_inside_half_open_interval(self):
        data = artifact()
        data["normalized_records"] += [
            {"sensor_id": 13502151, "event_time": "2024-01-01T00:00:00Z", "value": 10, "unit": "µg/m³", "record_id": "same"},
            {"sensor_id": 13502151, "event_time": "2024-01-01T01:00:00Z", "value": 12, "unit": "µg/m³", "record_id": "conflict"},
            {"sensor_id": 13502151, "event_time": "2024-01-01T03:00:00Z", "value": 9, "unit": "µg/m³", "record_id": "end"},
        ]
        self.assertEqual(openmeteo.valid_pm25_hours(openmeteo.load_pm25_artifact(data, 13502151)), [datetime(2024, 1, 1, tzinfo=UTC)])


class RequestTests(unittest.TestCase):
    def test_request_uses_exact_endpoint_parameters_and_whole_dates(self):
        seen = []
        raw = json.dumps(payload(), separators=(",", ":")).encode()
        with tempfile.TemporaryDirectory() as directory, httpx.Client(transport=httpx.MockTransport(lambda request: seen.append(request) or httpx.Response(200, content=raw))) as client:
            openmeteo.fetch(client, datetime(2024, 1, 1, 23, tzinfo=UTC), datetime(2024, 1, 3, 1, tzinfo=UTC), Path(directory), now=lambda: datetime(2025, 1, 1, tzinfo=UTC))
        request = seen[0]
        self.assertEqual(str(request.url).split("?")[0], openmeteo.HISTORICAL_FORECAST_ENDPOINT)
        self.assertEqual(request.url.params["hourly"], ",".join(VARIABLES))
        self.assertEqual(request.url.params["timezone"], "UTC")
        self.assertEqual(request.url.params["start_date"], "2024-01-01")
        self.assertEqual(request.url.params["end_date"], "2024-01-03")
        self.assertNotIn("wind_direction_10m", str(request.url))

    def test_raw_bytes_and_manifest_required_fields_hash_are_preserved(self):
        raw = b'{"exact": true, "spaces": 2}'
        with tempfile.TemporaryDirectory() as directory, httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=raw))) as client:
            _, manifest = openmeteo.fetch(client, datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC), Path(directory), now=lambda: datetime(2025, 1, 1, tzinfo=UTC))
            self.assertEqual(set(manifest), openmeteo.PROVENANCE_FIELDS)
            self.assertEqual(manifest["requested_hourly_variables"], list(VARIABLES))
            self.assertNotIn("hourly", manifest)
            self.assertEqual(Path(manifest["raw_payload_path"]).read_bytes(), raw)
            self.assertEqual(manifest["raw_payload_sha256"], sha256(raw).hexdigest())


class NormalizationTests(unittest.TestCase):
    def test_utc_and_gmt_labels_with_zero_offset_normalize_naive_hours_to_aware_utc(self):
        for label in ("UTC", "GMT"):
            with self.subTest(label=label):
                result = openmeteo.normalize_payload(payload(timezone_name=label))
                self.assertEqual(result["records"][0]["event_time"], datetime(2024, 1, 1, tzinfo=UTC))
                self.assertEqual(set(result["records"][0]) - {"event_time"}, set(VARIABLES))

    def test_nonzero_offset_or_other_timezone_is_rejected(self):
        for item in (payload(offset=3600), payload(timezone_name="Europe/London")):
            with self.assertRaisesRegex(openmeteo.OpenMeteoError, "UTC/GMT"):
                openmeteo.normalize_payload(item)

    def test_schema_and_array_length_fail_explicitly(self):
        malformed = payload()
        malformed["hourly"][VARIABLES[0]].pop()
        with self.assertRaisesRegex(openmeteo.OpenMeteoError, "array length"):
            openmeteo.normalize_payload(malformed)
        with self.assertRaisesRegex(openmeteo.OpenMeteoError, "hourly_units"):
            openmeteo.normalize_payload({"timezone": "UTC", "utc_offset_seconds": 0, "hourly": {}})

    def test_hourly_units_evidence_is_preserved(self):
        normalized = openmeteo.normalize_payload(payload())
        self.assertEqual(normalized["hourly_units"], payload()["hourly_units"])


class ReportTests(unittest.TestCase):
    def test_exact_join_report_counts_gate_missing_hours_and_per_variable_missing(self):
        pm25 = [datetime(2024, 1, 1, hour, tzinfo=UTC) for hour in range(3)]
        values = {VARIABLES[0]: [1, None, 1], VARIABLES[1]: [1, 1, None]}
        normalized = openmeteo.normalize_payload(payload(values=values))
        report = openmeteo.build_report(pm25, normalized["records"])
        self.assertEqual(report["valid_pm25_hours"], 3)
        self.assertEqual(report["complete_core_weather_hours"], 1)
        self.assertEqual(report["weather_join_coverage_pct"], 100 / 3)
        self.assertEqual(report["weather_gate"], "FAIL")
        self.assertEqual(len(report["missing_weather_timestamps_utc"]), 2)
        self.assertEqual(report["per_variable_missing_counts"][VARIABLES[0]], 1)
        self.assertEqual(report["per_variable_missing_counts"][VARIABLES[1]], 1)
        self.assertNotIn("optional_coverage", report)

    def test_exact_timestamp_only_never_uses_nearest_weather(self):
        hour = datetime(2024, 1, 1, 1, tzinfo=UTC)
        rows = openmeteo.normalize_payload(payload(times=["2024-01-01T00:00", "2024-01-01T02:00"]))["records"]
        report = openmeteo.build_report([hour], rows)
        self.assertEqual(report["exact_joined_hours"], 0)

    def test_denominator_is_valid_pm25_hours_only(self):
        hours = [datetime(2024, 1, 1, hour, tzinfo=UTC) for hour in range(2)]
        rows = openmeteo.normalize_payload(payload())["records"]
        self.assertEqual(openmeteo.build_report(hours, rows)["weather_join_coverage_pct"], 100.0)

    def test_weather_gate_exactly_95_passes_and_below_fails(self):
        start = datetime(2024, 1, 1, tzinfo=UTC)
        hours = [start.replace(hour=index) for index in range(20)]
        rows = [{"event_time": hour, **dict.fromkeys(VARIABLES, 1)} for hour in hours[:19]]
        self.assertEqual(openmeteo.build_report(hours, rows)["weather_numeric_gate"], "PASS")
        self.assertEqual(openmeteo.build_report(hours + [datetime(2024, 1, 2, tzinfo=UTC)], rows)["weather_numeric_gate"], "FAIL")

    def test_boundary_day_rows_outside_frozen_interval_are_ignored(self):
        start = datetime(2024, 1, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 1, 3, tzinfo=UTC)
        rows = openmeteo.normalize_payload(payload(times=["2024-01-01T00:00", "2024-01-01T01:00", "2024-01-01T02:00", "2024-01-01T03:00"]))["records"]
        report = openmeteo.build_report([start], rows, start, end)
        self.assertEqual(report["weather_hourly_rows"], 2)
        self.assertEqual(report["weather_valid_core_hours"], 2)

    def test_run_reuses_frozen_interval_without_reselection_and_prints_all_concepts(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "pm25.json"
            source.write_text(json.dumps(artifact()))
            output = io.StringIO()
            transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload()))
            with httpx.Client(transport=transport) as client:
                paths = openmeteo.run(client, 13502151, source, base / ".artifacts/data_spike", output=output, now=lambda: datetime(2025, 1, 1, tzinfo=UTC))
            report = json.loads(paths["report_path"].read_text())
            normalized = json.loads(paths["normalized_weather_path"].read_text())
            self.assertEqual(report["frozen_interval"], artifact()["frozen_candidate"] | {"start_utc": "2024-01-01T00:00:00+00:00", "end_utc": "2024-01-01T03:00:00+00:00"})
            self.assertIn("event_time", normalized["records"][0])
            self.assertEqual(report["weather_product"], openmeteo.HISTORICAL_FORECAST_ENDPOINT)
            for key in ("weather_hourly_rows", "weather_valid_core_hours", "exact_joined_hours", "coordinates"):
                self.assertIn(key, report)
            labels = ("Weather product:", "Coordinates:", "Frozen interval:", "Valid PM2.5 hours:", "Weather hourly rows:", "Weather valid core hours:", "Exact joined hours:", "Weather join coverage %:", "Weather numeric gate PASS/FAIL:", "Missing weather-at-PM2.5 hours:")
            for label in labels:
                self.assertIn(label, output.getvalue())
            for variable in VARIABLES:
                self.assertIn(f"{variable} missing count:", output.getvalue())

    def test_cli_weather_requires_sensor_id_and_pm25_path(self):
        result = subprocess.run([sys.executable, "scripts/run_data_spike.py", "weather"], cwd=ROOT, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--sensor-id", result.stderr)
        self.assertIn("--pm25-normalized-path", result.stderr)


if __name__ == "__main__":
    unittest.main()
