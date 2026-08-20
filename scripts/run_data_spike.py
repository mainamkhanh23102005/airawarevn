from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math
from zoneinfo import ZoneInfo

UTC = timezone.utc
CORE_WEATHER_VARIABLES = (
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "surface_pressure"
)


def normalize_to_utc(value, source_timezone=None):
    if value.tzinfo is None:
        if source_timezone is None:
            raise ValueError("naive timestamp needs source timezone")
        value = value.replace(tzinfo=source_timezone)
    return value.astimezone(UTC)


def expected_hourly_grid(start, end):
    start, end = normalize_to_utc(start), normalize_to_utc(end)
    result, current = [], start
    while current < end:
        result.append(current)
        current += timedelta(hours=1)
    return result


def calendar_month_bounds_utc(year, month, zone):
    start = datetime(year, month, 1, tzinfo=zone)
    end = datetime(year + (month == 12), 1 if month == 12 else month + 1, 1, tzinfo=zone)
    return start.astimezone(UTC), end.astimezone(UTC)


def convert_pm25_to_ug_m3(value, unit):
    aliases = {"µg/m³", "ug/m3", "μg/m³", "µg/m3"}
    if unit in aliases:
        return value
    if unit == "mg/m³":
        return value * 1000
    raise ValueError("unsupported PM2.5 unit")


def _valid_value(record):
    value = record.get("value")
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    try:
        return convert_pm25_to_ug_m3(value, record.get("unit"))
    except ValueError:
        return None


@dataclass(frozen=True)
class HourResolution:
    has_observation: bool
    has_valid_pm25: bool
    is_ambiguous: bool
    selected_value_ug_m3: float | None
    source_record_ids: tuple
    revision_metadata: tuple


def resolve_hour_duplicates(records):
    ordered = sorted(records, key=lambda r: str(r.get("record_id", "")))
    values = [_valid_value(r) for r in ordered]
    valid = [v for v in values if v is not None]
    ids = tuple(r.get("record_id") for r in ordered)
    revisions = tuple(r.get("provider_updated_at") for r in ordered)
    if not records:
        return HourResolution(False, False, False, None, (), ())
    if len(records) == 1:
        return HourResolution(True, bool(valid), False, valid[0] if valid else None, ids, revisions)
    if len(valid) == len(records) and len(set(valid)) == 1:
        return HourResolution(True, True, False, valid[0], ids, revisions)
    return HourResolution(True, False, True, None, ids, revisions)


@dataclass(frozen=True)
class Pm25Metrics:
    expected_hours: int
    observed_unique_hours: int
    valid_pm25_hours: int
    hourly_timestamp_completeness_pct: float | None
    usable_pm25_coverage_pct: float | None
    observed_row_non_null_rate_pct: float | None


def pm25_metrics(records, start, end):
    expected = expected_hourly_grid(start, end)
    grouped = {}
    for record in records:
        hour = normalize_to_utc(record["event_time"]).replace(minute=0, second=0, microsecond=0)
        if hour in expected:
            grouped.setdefault(hour, []).append(record)
    observed = len(grouped)
    valid = sum(resolve_hour_duplicates(rows).has_valid_pm25 for rows in grouped.values())
    total = len(expected)
    return Pm25Metrics(total, observed, valid, 100 * observed / total if total else None,
                       100 * valid / total if total else None, 100 * valid / observed if observed else None)


@dataclass(frozen=True)
class Gap:
    start_utc: datetime
    end_utc: datetime
    hours: int


def detect_gaps(expected, observed):
    gaps, start = [], None
    for hour in expected:
        if hour not in observed and start is None:
            start = hour
        if hour in observed and start is not None:
            gaps.append(Gap(start, hour, int((hour - start).total_seconds() // 3600)))
            start = None
    if start is not None:
        end = expected[-1] + timedelta(hours=1)
        gaps.append(Gap(start, end, int((end - start).total_seconds() // 3600)))
    return gaps


def longest_gap_hours(gaps): return max((gap.hours for gap in gaps), default=0)
def long_gaps(gaps): return [gap for gap in gaps if gap.hours > 12]


@dataclass(frozen=True)
class WinterWindow:
    start_utc: datetime
    end_utc: datetime
    expected_hours: int


def winter_window_utc(year, zone):
    start = datetime(year, 11, 1, tzinfo=zone).astimezone(UTC)
    end = datetime(year + 1, 3, 1, tzinfo=zone).astimezone(UTC)
    return WinterWindow(start, end, len(expected_hourly_grid(start, end)))


@dataclass(frozen=True)
class WinterMetrics:
    calendar_window_exists: bool
    expected_hours: int
    observed_hours: int
    valid_pm25_hours: int
    start_utc: datetime | None = None
    end_utc: datetime | None = None

    @property
    def completeness_passes(self): return self.observed_hours / self.expected_hours >= .85
    @property
    def usable_pm25_coverage_passes(self): return self.valid_pm25_hours / self.expected_hours >= .85


def winter_metrics(window, observed_hours, valid_pm25_hours):
    return WinterMetrics(True, window.expected_hours, observed_hours, valid_pm25_hours, window.start_utc, window.end_utc)

def winter_gate(windows, interval_start, interval_end):
    return any(
        w.calendar_window_exists
        and (w.start_utc is None or interval_start <= w.start_utc)
        and (w.end_utc is None or interval_end >= w.end_utc)
        and w.completeness_passes
        and w.usable_pm25_coverage_passes
        for w in windows
    )


@dataclass(frozen=True)
class CandidateInterval:
    start_utc: datetime
    end_utc: datetime
    months: tuple

    @property
    def start_local(self): return self.start_utc.astimezone(ZoneInfo("Asia/Ho_Chi_Minh"))
    @property
    def end_local(self): return self.end_utc.astimezone(ZoneInfo("Asia/Ho_Chi_Minh"))


def _next_month(item):
    year, month = item
    return (year + 1, 1) if month == 12 else (year, month + 1)

def consecutive_calendar_periods(months):
    months = sorted(set(months))
    periods = []
    for month in months:
        if not periods or month != _next_month(periods[-1][-1]): periods.append([month])
        else: periods[-1].append(month)
    return periods

def source_present_months(records, zone):
    return sorted({(normalize_to_utc(r["event_time"]).astimezone(zone).year, normalize_to_utc(r["event_time"]).astimezone(zone).month) for r in records})

def complete_calendar_months(records, zone, source_start=None, source_end=None):
    if not records:
        return []
    event_times = [normalize_to_utc(record["event_time"]) for record in records]
    source_start = normalize_to_utc(source_start) if source_start else min(event_times)
    source_end = normalize_to_utc(source_end) if source_end else max(event_times) + timedelta(hours=1)
    result = []
    for year, month in source_present_months(records, zone):
        start, end = calendar_month_bounds_utc(year, month, zone)
        if source_start <= start and end <= source_end:
            result.append((year, month))
    return result

def _interval(months, zone):
    start = datetime(months[0][0], months[0][1], 1, tzinfo=zone).astimezone(UTC)
    y, m = _next_month(months[-1])
    end = datetime(y, m, 1, tzinfo=zone).astimezone(UTC)
    return CandidateInterval(start, end, tuple(months))

def select_primary_candidate_interval(months, zone, quality=None):
    candidates = []
    for period in consecutive_calendar_periods(months):
        for index in range(len(period) - 11):
            candidate = _interval(period[index:index + 12], zone)
            years = range(candidate.start_local.year - 1, candidate.end_local.year + 1)
            if any(candidate.start_utc <= winter_window_utc(year, zone).start_utc and winter_window_utc(year, zone).end_utc <= candidate.end_utc for year in years):
                candidates.append(candidate)
    return max(candidates, key=lambda c: c.start_utc) if candidates else None

def legitimate_candidate_intervals(records, zone):
    months = complete_calendar_months(records, zone)
    candidate = select_primary_candidate_interval(months, zone)
    return [candidate] if candidate else []
def metrics_for_frozen_interval(records, interval):
    return {"start_utc": interval.start_utc, "end_utc": interval.end_utc, "metrics": pm25_metrics(records, interval.start_utc, interval.end_utc)}


@dataclass(frozen=True)
class WeatherJoinMetrics:
    complete_core_weather_hours: int
    weather_join_coverage_pct: float | None
    optional_coverage: dict
    invalid_duplicate_weather_hours: int

def _finite(value): return isinstance(value, (int, float)) and math.isfinite(value)
def weather_join_metrics(pm25_hours, weather, core_variables=CORE_WEATHER_VARIABLES):
    complete, invalid = 0, 0
    optional = 0
    for hour in pm25_hours:
        row = weather.get(hour)
        if isinstance(row, list): invalid += 1; continue
        if not isinstance(row, dict): continue
        if all(_finite(row.get(variable)) for variable in core_variables):
            complete += 1
            optional += _finite(row.get("wind_direction_10m"))
    denominator = len(pm25_hours)
    return WeatherJoinMetrics(complete, 100 * complete / denominator if denominator else None,
                              {"wind_direction_10m": 100 * optional / denominator if optional else None}, invalid)
def weather_gate_passes(value): return value is not None and value >= 95


def raw_payload_hash(payload): return sha256(payload).hexdigest()
def canonical_payload_hash(records):
    normalized = sorted((json.dumps(record, sort_keys=True, separators=(",", ":"), default=str) for record in records))
    return sha256(json.dumps(normalized, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class VintageComparison:
    unchanged_count: int
    revised_count: int
    backfilled_count: int
    missing_count: int
    metadata_change_count: int
    point_in_time_confidence: str = "UNKNOWN"
    claims_immutable: bool = False

def compare_vintages(baseline, current):
    old, new = {x["id"]: x for x in baseline}, {x["id"]: x for x in current}
    unchanged = revised = metadata = 0
    for key in old.keys() & new.keys():
        if old[key].get("value") != new[key].get("value"): revised += 1
        elif old[key].get("metadata") != new[key].get("metadata"): metadata += 1
        else: unchanged += 1
    return VintageComparison(unchanged, revised, len(new.keys() - old.keys()), len(old.keys() - new.keys()), metadata)


def point_in_time_eligible(record, prediction_time, require_ingested=False):
    event, available = record.get("event_time"), record.get("available_time")
    if event is None or available is None or event > prediction_time or available > prediction_time: return False
    return not require_ingested or record.get("ingested_at") is not None and record["ingested_at"] <= prediction_time

def retrospective_target_eligible(record): return bool(record.get("canonical"))


@dataclass(frozen=True)
class StructuralGateResult:
    has_legitimate_twelve_month_interval: bool
    has_complete_contained_winter: bool
    winter_quality_passes: bool
    units_convertible: bool
    duplicate_treatment_valid: bool
    long_gaps_reported: bool
    licensing_attribution_satisfied: bool
    weather_core_supported: bool

    @classmethod
    def all_passing(cls): return cls(*([True] * 8))
    @property
    def passes(self): return all(self.__dict__.values())

@dataclass(frozen=True)
class NumericGateResult:
    hourly_timestamp_completeness: float | None
    usable_pm25_coverage: float | None
    weather_join_coverage: float | None

@dataclass(frozen=True)
class Stage0GateEvaluation:
    pass_eligible: bool
    near_miss_eligible: bool
    numeric_gate_count: int = 3

def evaluate_stage0_gates(structural, numeric):
    values, thresholds = (numeric.hourly_timestamp_completeness, numeric.usable_pm25_coverage, numeric.weather_join_coverage), (85, 85, 95)
    passed = [value is not None and value >= threshold for value, threshold in zip(values, thresholds)]
    if not structural.passes: return Stage0GateEvaluation(False, False)
    if all(passed): return Stage0GateEvaluation(True, False)
    failed = [index for index, ok in enumerate(passed) if not ok]
    if len(failed) != 1: return Stage0GateEvaluation(False, False)
    value, threshold = values[failed[0]], thresholds[failed[0]]
    return Stage0GateEvaluation(False, value is not None and value >= threshold - 5)


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("discovery")
    coverage_parser = subparsers.add_parser("coverage")
    coverage_parser.add_argument("--sensor-id", required=True, type=int)
    weather_parser = subparsers.add_parser("weather")
    weather_parser.add_argument("--sensor-id", required=True, type=int)
    weather_parser.add_argument("--pm25-normalized-path", required=True)
    args = parser.parse_args(argv)
    import os
    from pathlib import Path

    import httpx
    from scripts import openaq, openmeteo

    if args.command == "weather":
        with httpx.Client(timeout=30) as client:
            openmeteo.run(client, args.sensor_id, Path(args.pm25_normalized_path), output=None)
        return
    api_key = os.environ.get("OPENAQ_API_KEY")
    if not api_key:
        parser.error("OPENAQ_API_KEY is required")
    artifact_directory = Path(".artifacts/data_spike")
    raw_directory = artifact_directory / "raw/openaq"
    discovery_path = artifact_directory / "openaq_discovery.json"
    with httpx.Client(timeout=30) as client:
        if args.command == "discovery":
            discovered = openaq.discover(client, api_key, raw_directory)
            artifact_directory.mkdir(parents=True, exist_ok=True)
            discovery_path.write_text(json.dumps(discovered, indent=2, sort_keys=True) + "\n")
            return
        discovered = openaq.discover(client, api_key, raw_directory)
        sensor = next((item for item in discovered if item["sensor_id"] == args.sensor_id), None)
        if sensor is None:
            raise openaq.OpenAQError(f"Sensor {args.sensor_id} not found in Hanoi discovery")
        sensor = openaq.enrich_sensor_metadata(client, api_key, sensor, raw_directory)
        records, provenance = [], []
        for chunk_start, chunk_end in openaq.sensor_history_chunks(sensor):
            rows, pages = openaq.fetch_hours(client, api_key, args.sensor_id, chunk_start, chunk_end, raw_directory)
            records.extend(rows)
            provenance.extend(pages)
    artifact_directory.mkdir(parents=True, exist_ok=True)
    (artifact_directory / "openaq_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    openaq.write_coverage_artifacts(records, sensor, artifact_directory / "coverage")


if __name__ == "__main__":
    main()
