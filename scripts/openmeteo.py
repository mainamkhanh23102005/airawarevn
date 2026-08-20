from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

UTC = timezone.utc
HISTORICAL_FORECAST_ENDPOINT = "https://historical-forecast-api.open-meteo.com/v1/forecast"
CORE_WEATHER_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "surface_pressure",
)
TARGET_SENSOR_ID = 13502151
TARGET_LATITUDE = 21.0031
TARGET_LONGITUDE = 105.7947
PROVENANCE_FIELDS = {
    "endpoint",
    "latitude",
    "longitude",
    "requested_hourly_variables",
    "timezone",
    "start_date",
    "end_date",
    "retrieved_at",
    "http_status",
    "raw_payload_path",
    "raw_payload_sha256",
}


class OpenMeteoError(RuntimeError):
    pass


@dataclass(frozen=True)
class Pm25Artifact:
    sensor_id: int
    start: datetime
    end: datetime
    records: tuple


def _aware_utc(value, label):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as error:
        raise OpenMeteoError(f"{label} must be an aware UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise OpenMeteoError(f"{label} must be an aware UTC timestamp")
    return parsed.astimezone(UTC)


def load_pm25_artifact(source, sensor_id):
    if isinstance(source, (str, Path)):
        try:
            data = json.loads(Path(source).read_text())
        except (OSError, ValueError) as error:
            raise OpenMeteoError(f"Unable to load normalized PM2.5 artifact: {error}") from error
    else:
        data = source
    if not isinstance(data, dict):
        raise OpenMeteoError("Normalized PM2.5 artifact must be a JSON object")
    candidate = data.get("frozen_candidate")
    interval = data.get("frozen_interval")
    if candidate is None and interval is None:
        raise OpenMeteoError("Normalized PM2.5 artifact lacks frozen interval")
    if candidate is not None and interval is not None and candidate != interval:
        raise OpenMeteoError("frozen_candidate and frozen_interval mismatch")
    frozen = candidate if candidate is not None else interval
    if not isinstance(frozen, dict) or set(("start_utc", "end_utc")) - set(frozen):
        raise OpenMeteoError("Invalid frozen interval")
    start = _aware_utc(frozen["start_utc"], "Frozen interval start")
    end = _aware_utc(frozen["end_utc"], "Frozen interval end")
    if start >= end:
        raise OpenMeteoError("Frozen interval start must precede end")
    metadata = data.get("sensor_metadata")
    if sensor_id != TARGET_SENSOR_ID or not isinstance(metadata, dict) or metadata.get("sensor_id") != sensor_id:
        raise OpenMeteoError(f"Unsupported or mismatched sensor ID {sensor_id}")
    coordinates = metadata.get("coordinates")
    latitude = coordinates.get("latitude") if isinstance(coordinates, dict) else None
    longitude = coordinates.get("longitude") if isinstance(coordinates, dict) else None
    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)) or not math.isclose(latitude, TARGET_LATITUDE, rel_tol=0, abs_tol=1e-6) or not math.isclose(longitude, TARGET_LONGITUDE, rel_tol=0, abs_tol=1e-6):
        raise OpenMeteoError("Sensor coordinates do not match fixed target")
    rows = data.get("normalized_records")
    if not isinstance(rows, list):
        raise OpenMeteoError("Normalized PM2.5 artifact lacks normalized records")
    records = []
    for row in rows:
        if not isinstance(row, dict) or row.get("sensor_id") != sensor_id or "event_time" not in row:
            raise OpenMeteoError("Normalized PM2.5 record schema mismatch")
        normalized = dict(row)
        normalized["event_time"] = _aware_utc(row["event_time"], "PM2.5 event_time")
        records.append(normalized)
    return Pm25Artifact(sensor_id, start, end, tuple(records))


def valid_pm25_hours(artifact):
    from scripts import run_data_spike as spike

    grouped = {}
    for record in artifact.records:
        hour = record["event_time"].replace(minute=0, second=0, microsecond=0)
        if artifact.start <= hour < artifact.end:
            grouped.setdefault(hour, []).append(record)
    return sorted(hour for hour, rows in grouped.items() if spike.resolve_hour_duplicates(rows).has_valid_pm25)


def fetch(client, start, end, raw_directory, now=lambda: datetime.now(UTC)):
    params = {
        "latitude": TARGET_LATITUDE,
        "longitude": TARGET_LONGITUDE,
        "hourly": ",".join(CORE_WEATHER_VARIABLES),
        "timezone": "UTC",
        "start_date": start.astimezone(UTC).date().isoformat(),
        "end_date": end.astimezone(UTC).date().isoformat(),
    }
    import httpx

    try:
        response = client.get(HISTORICAL_FORECAST_ENDPOINT, params=params)
        response.raise_for_status()
    except (httpx.RequestError, httpx.HTTPStatusError) as error:
        raise OpenMeteoError(f"Open-Meteo request failed: {error}") from error
    raw_directory.mkdir(parents=True, exist_ok=True)
    raw = response.content
    digest = sha256(raw).hexdigest()
    raw_path = raw_directory / f"openmeteo-{digest}.json"
    raw_path.write_bytes(raw)
    manifest = {
        "endpoint": HISTORICAL_FORECAST_ENDPOINT,
        "latitude": params["latitude"],
        "longitude": params["longitude"],
        "requested_hourly_variables": list(CORE_WEATHER_VARIABLES),
        "timezone": params["timezone"],
        "start_date": params["start_date"],
        "end_date": params["end_date"],
        "retrieved_at": now().astimezone(UTC).isoformat(),
        "http_status": response.status_code,
        "raw_payload_path": str(raw_path),
        "raw_payload_sha256": digest,
    }
    (raw_directory / f"openmeteo-{digest}.provenance.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        return json.loads(raw), manifest
    except ValueError as error:
        raise OpenMeteoError("Open-Meteo returned invalid JSON") from error


def normalize_payload(payload):
    if not isinstance(payload, dict) or payload.get("timezone") not in ("UTC", "GMT") or payload.get("utc_offset_seconds") != 0:
        raise OpenMeteoError("Open-Meteo timezone must be UTC/GMT with utc_offset_seconds==0")
    hourly = payload.get("hourly")
    units = payload.get("hourly_units")
    if not isinstance(hourly, dict) or not isinstance(units, dict) or any(variable not in units for variable in CORE_WEATHER_VARIABLES):
        raise OpenMeteoError("Open-Meteo hourly_units schema mismatch")
    times = hourly.get("time")
    if not isinstance(times, list) or any(not isinstance(hourly.get(variable), list) for variable in CORE_WEATHER_VARIABLES):
        raise OpenMeteoError("Open-Meteo hourly schema mismatch")
    if any(len(hourly[variable]) != len(times) for variable in CORE_WEATHER_VARIABLES):
        raise OpenMeteoError("Open-Meteo hourly array length mismatch")
    records = []
    seen = set()
    for index, value in enumerate(times):
        timestamp = _aware_utc(value + "+00:00" if isinstance(value, str) and datetime.fromisoformat(value).tzinfo is None else value, "Weather timestamp")
        if timestamp in seen:
            raise OpenMeteoError("Duplicate Open-Meteo hourly timestamp")
        seen.add(timestamp)
        records.append({"event_time": timestamp, **{variable: hourly[variable][index] for variable in CORE_WEATHER_VARIABLES}})
    return {"timezone": payload["timezone"], "utc_offset_seconds": 0, "hourly_units": dict(units), "records": records}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def build_report(pm25_hours, weather_records, start=None, end=None):
    bounded = [row for row in weather_records if (start is None or row["event_time"] >= start) and (end is None or row["event_time"] < end)]
    weather = {row["event_time"]: row for row in bounded}
    valid_weather = sum(all(_finite(row.get(variable)) for variable in CORE_WEATHER_VARIABLES) for row in bounded)
    joined = 0
    missing_timestamps = []
    missing_counts = dict.fromkeys(CORE_WEATHER_VARIABLES, 0)
    for hour in pm25_hours:
        row = weather.get(hour)
        missing = [variable for variable in CORE_WEATHER_VARIABLES if row is None or not _finite(row.get(variable))]
        for variable in missing:
            missing_counts[variable] += 1
        if missing:
            missing_timestamps.append(hour.isoformat())
        else:
            joined += 1
    coverage = 100 * joined / len(pm25_hours) if pm25_hours else None
    gate = "PASS" if coverage is not None and coverage >= 95 else "FAIL"
    return {
        "valid_pm25_hours": len(pm25_hours),
        "weather_hourly_rows": len(bounded),
        "weather_valid_core_hours": valid_weather,
        "exact_joined_hours": joined,
        "complete_core_weather_hours": joined,
        "weather_join_coverage_pct": coverage,
        "weather_numeric_gate": gate,
        "weather_gate": gate,
        "missing_weather_timestamps_utc": missing_timestamps,
        "per_variable_missing_counts": missing_counts,
    }


def run(client, sensor_id, pm25_path, artifact_directory=Path(".artifacts/data_spike"), output=None, now=lambda: datetime.now(UTC)):
    output = output or sys.stdout
    artifact = load_pm25_artifact(pm25_path, sensor_id)
    pm25_hours = valid_pm25_hours(artifact)
    payload, provenance = fetch(client, artifact.start, artifact.end, artifact_directory / "raw/openmeteo", now=now)
    normalized = normalize_payload(payload)
    weather_records = normalized["records"]
    normalized["records"] = [{**row, "event_time": row["event_time"].isoformat()} for row in weather_records]
    weather_directory = artifact_directory / "weather"
    weather_directory.mkdir(parents=True, exist_ok=True)
    normalized_path = weather_directory / "openmeteo_normalized.json"
    normalized_path.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n")
    report = build_report(pm25_hours, weather_records, artifact.start, artifact.end)
    report["weather_product"] = HISTORICAL_FORECAST_ENDPOINT
    report["frozen_interval"] = {"start_utc": artifact.start.isoformat(), "end_utc": artifact.end.isoformat()}
    report["sensor_id"] = sensor_id
    report["coordinates"] = {"latitude": TARGET_LATITUDE, "longitude": TARGET_LONGITUDE}
    report["provenance_manifest"] = provenance
    report_path = weather_directory / "openmeteo_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    coverage = "undefined" if report["weather_join_coverage_pct"] is None else f'{report["weather_join_coverage_pct"]:.6f}%'
    print(f"Weather product: {HISTORICAL_FORECAST_ENDPOINT}", file=output)
    print(f"Coordinates: {TARGET_LATITUDE},{TARGET_LONGITUDE}", file=output)
    print(f"Frozen interval: {artifact.start.isoformat()} to {artifact.end.isoformat()}", file=output)
    print(f'Valid PM2.5 hours: {report["valid_pm25_hours"]}', file=output)
    print(f'Weather hourly rows: {report["weather_hourly_rows"]}', file=output)
    print(f'Weather valid core hours: {report["weather_valid_core_hours"]}', file=output)
    print(f'Exact joined hours: {report["exact_joined_hours"]}', file=output)
    print(f"Weather join coverage %: {coverage}", file=output)
    print(f'Weather numeric gate PASS/FAIL: {report["weather_numeric_gate"]}', file=output)
    print(f'Missing weather-at-PM2.5 hours: {json.dumps(report["missing_weather_timestamps_utc"])}', file=output)
    for variable in CORE_WEATHER_VARIABLES:
        print(f'{variable} missing count: {report["per_variable_missing_counts"][variable]}', file=output)
    return {"normalized_weather_path": normalized_path, "report_path": report_path}
