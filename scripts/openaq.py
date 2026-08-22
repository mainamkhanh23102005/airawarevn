from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import httpx


BASE_URL = "https://api.openaq.org/v3"
HANOI = ZoneInfo("Asia/Ho_Chi_Minh")
UTC = timezone.utc
TRANSIENT_STATUSES = {408, 429, 500, 502, 503, 504}
PROVENANCE_FIELDS = {"endpoint", "sensor_id", "page", "limit", "datetime_from", "datetime_to", "retrieved_at", "http_status", "raw_payload_path", "raw_payload_sha256"}


class OpenAQError(RuntimeError):
    pass


def _parse_utc(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _cache_key(request):
    exact = {key: request.get(key) for key in ("endpoint", "sensor_id", "page", "limit", "datetime_from", "datetime_to")}
    return sha256(json.dumps(exact, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _cached(raw_directory, request):
    manifest = raw_directory / f"{_cache_key(request)}.provenance.json"
    if not manifest.exists():
        return None
    try:
        provenance = json.loads(manifest.read_text())
        if set(provenance) != PROVENANCE_FIELDS or provenance["http_status"] < 200 or provenance["http_status"] >= 300:
            return None
        if any(provenance.get(key) != value for key, value in request.items()):
            return None
        payload = Path(provenance["raw_payload_path"]).read_bytes()
        if sha256(payload).hexdigest() != provenance["raw_payload_sha256"]:
            return None
        return json.loads(payload), provenance
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _request(client, api_key, path, params, raw_directory, sensor_id=None, sleep=time.sleep, now=lambda: datetime.now(UTC)):
    raw_directory.mkdir(parents=True, exist_ok=True)
    request = {"endpoint": path, "sensor_id": sensor_id, "page": params.get("page"), "limit": params.get("limit"), "datetime_from": params.get("datetime_from"), "datetime_to": params.get("datetime_to")}
    cached = _cached(raw_directory, request)
    if cached:
        return cached
    for attempt in range(4):
        try:
            response = client.get(f"{BASE_URL}{path}", params=params, headers={"X-API-Key": api_key})
        except httpx.TransportError as error:
            if attempt == 3:
                raise OpenAQError(f"OpenAQ request failed after retries: {error}") from error
            sleep(2 ** attempt)
            continue
        if response.status_code in TRANSIENT_STATUSES:
            if attempt < 3:
                sleep(2 ** attempt)
                continue
            raise OpenAQError(f"OpenAQ request failed after retries with HTTP {response.status_code}")
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise OpenAQError(f"OpenAQ request failed with HTTP {response.status_code}") from error
        payload = response.content
        digest = sha256(payload).hexdigest()
        payload_path = raw_directory / f"{_cache_key(request)}-{digest}.json"
        payload_path.write_bytes(payload)
        provenance = {**request, "retrieved_at": now().astimezone(UTC).isoformat(), "http_status": response.status_code, "raw_payload_path": str(payload_path), "raw_payload_sha256": digest}
        (raw_directory / f"{_cache_key(request)}.provenance.json").write_text(json.dumps(provenance, sort_keys=True))
        try:
            return json.loads(payload), provenance
        except ValueError as error:
            raise OpenAQError("OpenAQ returned invalid JSON") from error
    raise OpenAQError("OpenAQ request failed")


def discover(client, api_key, raw_directory, sleep=time.sleep, now=lambda: datetime.now(UTC)):
    params = {"coordinates": "21.0285,105.8542", "radius": 25000, "parameters_id": 2, "limit": 1000, "page": 1}
    payload, _ = _request(client, api_key, "/locations", params, raw_directory, sleep=sleep, now=now)
    discovered = []
    for location in payload.get("results", []):
        for sensor in location.get("sensors", []):
            parameter = sensor.get("parameter", {})
            if parameter.get("id") == 2:
                discovered.append({"location_id": location.get("id"), "location_name": location.get("name"), "timezone": location.get("timezone"), "provider": location.get("provider"), "owner": location.get("owner"), "isMobile": location.get("isMobile"), "isMonitor": location.get("isMonitor"), "coordinates": location.get("coordinates"), "sensor_id": sensor.get("id"), "parameter_id": parameter.get("id"), "parameter_name": parameter.get("name"), "units": parameter.get("units"), "datetimeFirst": sensor.get("datetimeFirst"), "datetimeLast": sensor.get("datetimeLast")})
    return discovered


def enrich_sensor_metadata(client, api_key, metadata, raw_directory, sleep=time.sleep, now=lambda: datetime.now(UTC)):
    sensor_id = metadata.get("sensor_id")
    if sensor_id is None:
        raise OpenAQError("Sensor metadata lacks sensor_id")
    payload, _ = _request(client, api_key, f"/sensors/{sensor_id}", {}, raw_directory, sensor_id, sleep, now)
    results = payload.get("results", [])
    if not isinstance(results, list) or not results:
        raise OpenAQError(f"Sensor {sensor_id} metadata not found")
    authoritative = results[0]
    return {**metadata, "datetimeFirst": authoritative.get("datetimeFirst"), "datetimeLast": authoritative.get("datetimeLast")}


def sensor_history_chunks_for_coverage(client, api_key, metadata, raw_directory, sleep=time.sleep, now=lambda: datetime.now(UTC)):
    return sensor_history_chunks(enrich_sensor_metadata(client, api_key, metadata, raw_directory, sleep, now))


def normalize_measurement(row, sensor_id):
    period = row["period"]
    return {"sensor_id": sensor_id, "event_time": _parse_utc(period["datetimeFrom"]["utc"]), "period_end_utc": _parse_utc(period["datetimeTo"]["utc"]), "value": row.get("value"), "unit": row.get("parameter", {}).get("units"), "record_id": row.get("id")}


def fetch_hours(client, api_key, sensor_id, start, end, raw_directory, limit=1000, sleep=time.sleep, now=lambda: datetime.now(UTC)):
    records, provenance, page = [], [], 1
    while True:
        params = {"datetime_from": start.isoformat(), "datetime_to": end.isoformat(), "limit": limit, "page": page}
        payload, item = _request(client, api_key, f"/sensors/{sensor_id}/hours", params, raw_directory, sensor_id, sleep, now)
        rows = payload.get("results", [])
        if not isinstance(rows, list):
            raise OpenAQError("OpenAQ results is not a list")
        records.extend(normalize_measurement(row, sensor_id) for row in rows)
        provenance.append(item)
        if len(rows) < limit:
            break
        page += 1
    return records, provenance


def hanoi_month_chunks(start, end):
    current = datetime(start.astimezone(HANOI).year, start.astimezone(HANOI).month, 1, tzinfo=HANOI)
    chunks = []
    while current.astimezone(UTC) < end:
        following = datetime(current.year + (current.month == 12), 1 if current.month == 12 else current.month + 1, 1, tzinfo=HANOI)
        chunks.append((max(start, current.astimezone(UTC)), min(end, following.astimezone(UTC))))
        current = following
    return chunks


def sensor_history_chunks(metadata):
    first = metadata.get("datetimeFirst")
    last = metadata.get("datetimeLast")
    if isinstance(first, dict):
        first = first.get("utc")
    if isinstance(last, dict):
        last = last.get("utc")
    if not first or not last:
        raise OpenAQError("Sensor metadata lacks datetimeFirst or datetimeLast")
    return hanoi_month_chunks(_parse_utc(first), _parse_utc(last))


def write_coverage_artifacts(records, sensor_metadata, coverage_directory, source_start=None, source_end=None, output=None):
    output = output or __import__("sys").stdout
    coverage_path = write_coverage_report(records, coverage_directory, source_start, source_end)
    report = json.loads(coverage_path.read_text())
    frozen = report["frozen_candidate"] or {"start_utc": report["source_date_range"]["start_utc"], "end_utc": report["source_date_range"]["end_utc"]}
    timestamp = frozen["start_utc"].replace(":", "").replace("-", "")
    normalized_path = coverage_directory / f'openaq_normalized_sensor_{sensor_metadata["sensor_id"]}_{timestamp}.json'
    normalized = {
        "sensor_metadata": sensor_metadata,
        "frozen_candidate": frozen,
        "normalized_records": [
            {**record, "event_time": record["event_time"].isoformat(), "period_end_utc": record["period_end_utc"].isoformat()}
            for record in records
            if _parse_utc(frozen["start_utc"]) <= record["event_time"] < _parse_utc(frozen["end_utc"])
        ],
    }
    normalized_path.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n")
    coverage = report["coverage"]
    winter = report["winter"][0] if report["winter"] else None
    print(f'Sensor: {sensor_metadata["sensor_id"]}', file=output)
    print(f'Source date range: {report["source_date_range"]["start_utc"]} to {report["source_date_range"]["end_utc"]}', file=output)
    print(f'Frozen candidate: {frozen["start_utc"]} to {frozen["end_utc"]}', file=output)
    print(f'Expected hours: {coverage["expected_hours"]}', file=output)
    print(f'Observed hours: {coverage["observed_unique_hours"]}', file=output)
    print(f'Valid PM2.5 hours: {coverage["valid_pm25_hours"]}', file=output)
    print(f'Timestamp completeness: {coverage["hourly_timestamp_completeness_pct"]}', file=output)
    print(f'Usable PM2.5 coverage: {coverage["usable_pm25_coverage_pct"]}', file=output)
    print(f'Winter completeness: {winter["hourly_timestamp_completeness_pct"] if winter else "undefined"}', file=output)
    print(f'Winter usable coverage: {winter["usable_pm25_coverage_pct"] if winter else "undefined"}', file=output)
    print(f'Longest gap: {report["gaps"]["longest_hours"]}', file=output)
    print(f'PM2.5 gate: {report["numeric_pm25_gate"]["status"]}', file=output)
    print(f'Normalized artifact path: {normalized_path}', file=output)
    print(f'Coverage artifact path: {coverage_path}', file=output)
    return {"normalized_path": normalized_path, "coverage_path": coverage_path}


def write_coverage_report(records, coverage_directory, source_start=None, source_end=None):
    from scripts import run_data_spike as spike

    if not records:
        raise OpenAQError("Coverage requires at least one normalized record")
    coverage_directory.mkdir(parents=True, exist_ok=True)
    derived_start = min(record["event_time"] for record in records)
    derived_end = max(record["period_end_utc"] for record in records)
    structural_start = source_start or derived_start
    structural_end = source_end or derived_end
    months = spike.complete_calendar_months(records, HANOI, structural_start, structural_end)
    candidate = spike.select_primary_candidate_interval(months, HANOI)
    interval_start, interval_end = (candidate.start_utc, candidate.end_utc) if candidate else (derived_start, derived_end)
    metrics = spike.pm25_metrics(records, interval_start, interval_end)
    grouped = {}
    for record in records:
        key = (record["sensor_id"], record["event_time"])
        grouped.setdefault(key, []).append(record)
    hour_groups = {}
    for record in records:
        hour_groups.setdefault(record["event_time"].replace(minute=0, second=0, microsecond=0), []).append(record)
    observed = set(hour_groups)
    gaps = spike.detect_gaps(spike.expected_hourly_grid(interval_start, interval_end), observed)
    winters = []
    for year in range(interval_start.astimezone(HANOI).year - 1, interval_end.astimezone(HANOI).year + 1):
        window = spike.winter_window_utc(year, HANOI)
        if interval_start <= window.start_utc and window.end_utc <= interval_end:
            winter = spike.pm25_metrics(records, window.start_utc, window.end_utc)
            passes = winter.hourly_timestamp_completeness_pct >= 85 and winter.usable_pm25_coverage_pct >= 85
            winters.append({"start_utc": window.start_utc.isoformat(), "end_utc": window.end_utc.isoformat(), "expected_hours": winter.expected_hours, "observed_hours": winter.observed_unique_hours, "valid_pm25_hours": winter.valid_pm25_hours, "hourly_timestamp_completeness_pct": winter.hourly_timestamp_completeness_pct, "usable_pm25_coverage_pct": winter.usable_pm25_coverage_pct, "status": "PASS" if passes else "FAIL"})
    numeric_pass = metrics.hourly_timestamp_completeness_pct >= 85 and metrics.usable_pm25_coverage_pct >= 85
    winter_pass = bool(winters) and any(item["status"] == "PASS" for item in winters)
    report = {"source_date_range": {"start_utc": derived_start.isoformat(), "end_utc": derived_end.isoformat()}, "raw_row_count": len(records), "unique_canonical_key_count": len(grouped), "duplicate_row_count": len(records) - len(grouped), "duplicate_hour_count": sum(len(rows) > 1 for rows in hour_groups.values()), "ambiguous_conflict_count": sum(spike.resolve_hour_duplicates(rows).is_ambiguous for rows in hour_groups.values()), "structural_months": [f"{year:04d}-{month:02d}" for year, month in months], "frozen_candidate": {"start_utc": candidate.start_utc.isoformat(), "end_utc": candidate.end_utc.isoformat()} if candidate else None, "coverage": {"expected_hours": metrics.expected_hours, "observed_unique_hours": metrics.observed_unique_hours, "valid_pm25_hours": metrics.valid_pm25_hours, "hourly_timestamp_completeness_pct": metrics.hourly_timestamp_completeness_pct, "usable_pm25_coverage_pct": metrics.usable_pm25_coverage_pct, "observed_row_non_null_rate_pct": metrics.observed_row_non_null_rate_pct}, "winter": winters, "gaps": {"longest_hours": spike.longest_gap_hours(gaps), "over_twelve_hours": [{"start_utc": gap.start_utc.isoformat(), "end_utc": gap.end_utc.isoformat(), "hours": gap.hours} for gap in spike.long_gaps(gaps)]}, "numeric_pm25_gate": {"status": "PASS" if numeric_pass else "FAIL"}, "overall_status": "PASS" if candidate and winter_pass and numeric_pass else "FAIL"}
    path = coverage_directory / "openaq_coverage.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return path
