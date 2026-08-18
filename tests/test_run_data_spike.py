import math
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from zoneinfo import ZoneInfo

from scripts import run_data_spike as spike

UTC = timezone.utc
HANOI = ZoneInfo("Asia/Ho_Chi_Minh")
CORE_WEATHER_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "surface_pressure",
)


def utc_hour(year, month, day, hour=0):
    return datetime(year, month, day, hour, tzinfo=UTC)


def measurement(hour, value=25.0, unit="µg/m³", **extra):
    return {
        "event_time": hour,
        "value": value,
        "unit": unit,
        **extra,
    }


def measurements_for_half_open_range(start, end):
    records = []
    current = start
    while current < end:
        records.append(measurement(current))
        current += timedelta(hours=1)
    return records


class CandidatePeriodSelectionTests(unittest.TestCase):
    def test_source_present_months_include_null_unsupported_and_ambiguous_records(self):
        records = [
            measurement(utc_hour(2023, 1, 15), None),
            measurement(utc_hour(2023, 2, 15), 1.0, "mystery"),
            measurement(utc_hour(2023, 3, 15), 10.0, record_id="a"),
            measurement(utc_hour(2023, 3, 15), 20.0, record_id="b"),
        ]

        months = spike.source_present_months(records, HANOI)

        self.assertEqual(months, [(2023, 1), (2023, 2), (2023, 3)])

    def test_zero_record_month_breaks_consecutive_period(self):
        months = [(2022, 10), (2022, 11), (2023, 1), (2023, 2)]

        periods = spike.consecutive_calendar_periods(months)

        self.assertEqual(periods, [[(2022, 10), (2022, 11)], [(2023, 1), (2023, 2)]])

    def test_sparse_internal_observed_months_are_structural_when_source_range_spans_boundaries(self):
        records = [measurement(datetime(2022, month, 15, tzinfo=HANOI).astimezone(UTC), None) for month in range(1, 13)]

        months = spike.complete_calendar_months(
            records,
            HANOI,
            datetime(2022, 1, 1, tzinfo=HANOI).astimezone(UTC),
            datetime(2023, 1, 1, tzinfo=HANOI).astimezone(UTC),
        )

        self.assertEqual(months, [(2022, month) for month in range(1, 13)])

    def test_partial_boundary_months_cannot_form_candidate_boundaries(self):
        start = datetime(2022, 10, 1, tzinfo=HANOI).astimezone(UTC)
        end = datetime(2023, 12, 1, tzinfo=HANOI).astimezone(UTC)
        records = measurements_for_half_open_range(start, end)
        records = [
            record
            for record in records
            if record["event_time"] >= datetime(2022, 10, 15, tzinfo=HANOI).astimezone(UTC)
            and record["event_time"] < datetime(2023, 11, 15, tzinfo=HANOI).astimezone(UTC)
        ]

        candidates = spike.legitimate_candidate_intervals(records, HANOI)

        self.assertEqual(
            spike.complete_calendar_months(records, HANOI),
            [(2022, 11), (2022, 12)] + [(2023, month) for month in range(1, 11)],
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].start_local, datetime(2022, 11, 1, tzinfo=HANOI))
        self.assertEqual(candidates[0].end_local, datetime(2023, 11, 1, tzinfo=HANOI))

    def test_source_span_excludes_partial_boundaries_retains_sparse_internal_and_breaks_zero_month(self):
        source_start = datetime(2022, 1, 15, tzinfo=HANOI).astimezone(UTC)
        source_end = datetime(2023, 3, 15, tzinfo=HANOI).astimezone(UTC)
        records = [measurement(datetime(2022, month, 15, tzinfo=HANOI).astimezone(UTC)) for month in range(1, 13) if month != 6]
        records += [measurement(datetime(2023, month, 15, tzinfo=HANOI).astimezone(UTC)) for month in range(1, 4)]

        months = spike.complete_calendar_months(records, HANOI, source_start, source_end)

        self.assertNotIn((2022, 1), months)
        self.assertNotIn((2023, 3), months)
        self.assertNotIn((2022, 6), months)
        self.assertEqual(spike.consecutive_calendar_periods(months)[0][-1], (2022, 5))
        self.assertEqual(spike.consecutive_calendar_periods(months)[1][0], (2022, 7))

    def test_candidate_requires_actual_contained_november_to_march_winter(self):
        months = [(2022, month) for month in range(12, 13)] + [(2023, month) for month in range(1, 13)]

        selected = spike.select_primary_candidate_interval(months, HANOI)

        self.assertIsNone(selected)

    def test_selects_most_recent_structurally_qualifying_twelve_month_interval(self):
        months = [(2021, month) for month in range(11, 13)] + [
            (2022, month) for month in range(1, 13)
        ] + [(2023, month) for month in range(1, 11)]

        selected = spike.select_primary_candidate_interval(months, HANOI)

        self.assertEqual(selected.start_local, datetime(2022, 11, 1, tzinfo=HANOI))
        self.assertEqual(selected.end_local, datetime(2023, 11, 1, tzinfo=HANOI))
        self.assertEqual(len(selected.months), 12)

    def test_selection_is_independent_of_radically_different_quality_metrics(self):
        months = [(2021, month) for month in range(11, 13)] + [
            (2022, month) for month in range(1, 13)
        ] + [(2023, month) for month in range(1, 11)]
        terrible_newest_quality = {(2023, month): {"observed_hours": 1, "valid_hours": 0} for month in range(1, 11)}
        excellent_newest_quality = {(2023, month): {"observed_hours": 744, "valid_hours": 744} for month in range(1, 11)}

        terrible_selected = spike.select_primary_candidate_interval(months, HANOI, quality=terrible_newest_quality)
        excellent_selected = spike.select_primary_candidate_interval(months, HANOI, quality=excellent_newest_quality)

        self.assertEqual(terrible_selected, excellent_selected)
        self.assertEqual(terrible_selected.start_local, datetime(2022, 11, 1, tzinfo=HANOI))

    def test_candidate_selection_is_invariant_to_input_order(self):
        months = [(2022, month) for month in range(11, 13)] + [
            (2023, month) for month in range(1, 11)
        ]

        selected = spike.select_primary_candidate_interval(months, HANOI)
        reversed_selected = spike.select_primary_candidate_interval(list(reversed(months)), HANOI)

        self.assertEqual(selected, reversed_selected)

    def test_poor_internal_month_remains_inside_selected_interval(self):
        months = [(2022, month) for month in range(11, 13)] + [
            (2023, month) for month in range(1, 11)
        ]
        quality = {(2023, 5): {"observed_hours": 1, "valid_hours": 0}}

        selected = spike.select_primary_candidate_interval(months, HANOI, quality=quality)

        self.assertIn((2023, 5), selected.months)

    def test_ten_or_eleven_month_periods_do_not_qualify(self):
        ten_months = [(2022, month) for month in range(3, 13)]
        eleven_months = [(2022, month) for month in range(2, 13)]

        self.assertIsNone(spike.select_primary_candidate_interval(ten_months, HANOI))
        self.assertIsNone(spike.select_primary_candidate_interval(eleven_months, HANOI))

    def test_frozen_boundaries_reused_without_reselection(self):
        frozen = spike.CandidateInterval(
            start_utc=utc_hour(2022, 10, 31, 17),
            end_utc=utc_hour(2023, 10, 31, 17),
            months=tuple((2022, month) for month in range(11, 13))
            + tuple((2023, month) for month in range(1, 11)),
        )

        metrics = spike.metrics_for_frozen_interval([], frozen)

        self.assertEqual(metrics["start_utc"], frozen.start_utc)
        self.assertEqual(metrics["end_utc"], frozen.end_utc)


class ExpectedHourlyGridTests(unittest.TestCase):
    def test_half_open_expected_hour_counts(self):
        self.assertEqual(spike.expected_hourly_grid(utc_hour(2024, 1, 1), utc_hour(2024, 1, 1, 1)), [utc_hour(2024, 1, 1)])
        self.assertEqual(len(spike.expected_hourly_grid(utc_hour(2024, 1, 1), utc_hour(2024, 1, 2))), 24)
        self.assertEqual(len(spike.expected_hourly_grid(utc_hour(2024, 2, 1), utc_hour(2024, 3, 1))), 696)
        self.assertEqual(len(spike.expected_hourly_grid(utc_hour(2023, 2, 1), utc_hour(2023, 3, 1))), 672)
        self.assertEqual(len(spike.expected_hourly_grid(utc_hour(2023, 1, 1), utc_hour(2024, 1, 1))), 8760)

    def test_hanoi_calendar_boundaries_normalize_to_utc(self):
        start, end = spike.calendar_month_bounds_utc(2023, 11, HANOI)

        self.assertEqual(start, utc_hour(2023, 10, 31, 17))
        self.assertEqual(end, utc_hour(2023, 11, 30, 17))


class Pm25MetricTests(unittest.TestCase):
    def test_timestamp_completeness_counts_unique_observed_hours_only(self):
        start = utc_hour(2024, 1, 1)
        end = start + timedelta(hours=4)
        records = [
            measurement(start),
            measurement(start),
            measurement(start + timedelta(hours=1), None),
            measurement(start + timedelta(hours=3)),
        ]

        metrics = spike.pm25_metrics(records, start, end)

        self.assertEqual(metrics.observed_unique_hours, 3)
        self.assertEqual(metrics.expected_hours, 4)
        self.assertEqual(metrics.hourly_timestamp_completeness_pct, 75.0)

    def test_generated_hours_do_not_count_as_observed(self):
        start = utc_hour(2024, 1, 1)
        metrics = spike.pm25_metrics([measurement(start)], start, start + timedelta(hours=2))

        self.assertEqual(metrics.observed_unique_hours, 1)
        self.assertEqual(metrics.hourly_timestamp_completeness_pct, 50.0)

    def test_usable_coverage_rejects_invalid_values_and_ambiguous_duplicates(self):
        start = utc_hour(2024, 1, 1)
        records = [
            measurement(start, 1.0),
            measurement(start + timedelta(hours=1), None),
            measurement(start + timedelta(hours=2), math.nan),
            measurement(start + timedelta(hours=3), math.inf),
            measurement(start + timedelta(hours=4), -math.inf),
            measurement(start + timedelta(hours=5), 2.0, "unknown"),
            measurement(start + timedelta(hours=6), 3.0, record_id="left"),
            measurement(start + timedelta(hours=6), 4.0, record_id="right"),
        ]

        metrics = spike.pm25_metrics(records, start, start + timedelta(hours=8))

        self.assertEqual(metrics.valid_pm25_hours, 1)
        self.assertEqual(metrics.usable_pm25_coverage_pct, 12.5)

    def test_completeness_can_pass_when_usable_coverage_fails(self):
        start = utc_hour(2024, 1, 1)
        records = [measurement(start + timedelta(hours=index), None) for index in range(20)]
        records[0] = measurement(start)

        metrics = spike.pm25_metrics(records, start, start + timedelta(hours=20))

        self.assertEqual(metrics.hourly_timestamp_completeness_pct, 100.0)
        self.assertEqual(metrics.usable_pm25_coverage_pct, 5.0)

    def test_observed_row_non_null_rate_is_diagnostic_and_handles_zero_denominator(self):
        start = utc_hour(2024, 1, 1)
        sparse = spike.pm25_metrics([measurement(start)], start, start + timedelta(hours=10))
        empty = spike.pm25_metrics([], start, start + timedelta(hours=1))

        self.assertEqual(sparse.observed_row_non_null_rate_pct, 100.0)
        self.assertEqual(sparse.usable_pm25_coverage_pct, 10.0)
        self.assertIsNone(empty.observed_row_non_null_rate_pct)


class WinterQualityTests(unittest.TestCase):
    def test_winter_boundaries_and_leap_year_hours(self):
        window = spike.winter_window_utc(2023, HANOI)

        self.assertEqual(window.start_utc, utc_hour(2023, 10, 31, 17))
        self.assertEqual(window.end_utc, utc_hour(2024, 2, 29, 17))
        self.assertEqual(window.expected_hours, 121 * 24)

    def test_winter_exact_completeness_threshold_passes_and_just_below_fails(self):
        window = spike.winter_window_utc(2022, HANOI)
        exact = spike.winter_metrics(window, observed_hours=2448, valid_pm25_hours=2448)
        below = spike.winter_metrics(window, observed_hours=2447, valid_pm25_hours=2447)

        self.assertEqual(window.expected_hours, 2880)
        self.assertTrue(exact.completeness_passes)
        self.assertFalse(below.completeness_passes)

    def test_winter_exact_usable_coverage_threshold_passes_and_just_below_fails(self):
        window = spike.winter_window_utc(2022, HANOI)
        exact = spike.winter_metrics(window, observed_hours=2880, valid_pm25_hours=2448)
        below = spike.winter_metrics(window, observed_hours=2880, valid_pm25_hours=2447)

        self.assertTrue(exact.usable_pm25_coverage_passes)
        self.assertFalse(below.usable_pm25_coverage_passes)

    def test_winter_requires_complete_containment_and_both_quality_thresholds(self):
        window = spike.winter_window_utc(2022, HANOI)
        passing = spike.winter_metrics(window, observed_hours=2448, valid_pm25_hours=2448)
        failing_completeness = spike.winter_metrics(window, observed_hours=2447, valid_pm25_hours=2448)

        self.assertTrue(spike.winter_gate([passing], window.start_utc, window.end_utc))
        self.assertFalse(spike.winter_gate([failing_completeness], window.start_utc, window.end_utc))
        self.assertFalse(spike.winter_gate([passing], window.start_utc + timedelta(hours=1), window.end_utc))

    def test_any_passing_contained_winter_satisfies_composite_requirement(self):
        failed = spike.WinterMetrics(False, 100, 84, 84)
        passed = spike.WinterMetrics(True, 100, 85, 85)

        self.assertTrue(spike.winter_gate([failed, passed], utc_hour(2022, 1, 1), utc_hour(2025, 1, 1)))


class DuplicateAndUnitTests(unittest.TestCase):
    def test_identical_duplicates_collapse_without_order_dependence_and_preserve_metadata(self):
        hour = utc_hour(2024, 1, 1)
        left = measurement(hour, 10.0, record_id="a", provider_updated_at=utc_hour(2024, 1, 2))
        right = measurement(hour, 10.0, record_id="b", provider_updated_at=utc_hour(2024, 1, 3))

        first = spike.resolve_hour_duplicates([left, right])
        second = spike.resolve_hour_duplicates([right, left])

        self.assertEqual(first, second)
        self.assertFalse(first.is_ambiguous)
        self.assertTrue(first.has_valid_pm25)
        self.assertEqual(set(first.source_record_ids), {"a", "b"})
        self.assertEqual(len(first.revision_metadata), 2)

    def test_conflicting_duplicates_are_ambiguous_and_later_update_does_not_win(self):
        hour = utc_hour(2024, 1, 1)
        earlier = measurement(hour, 10.0, record_id="old", provider_updated_at=utc_hour(2024, 1, 2))
        later = measurement(hour, 20.0, record_id="new", provider_updated_at=utc_hour(2024, 1, 3))

        resolved = spike.resolve_hour_duplicates([earlier, later])

        self.assertTrue(resolved.is_ambiguous)
        self.assertTrue(resolved.has_observation)
        self.assertFalse(resolved.has_valid_pm25)
        self.assertIsNone(resolved.selected_value_ug_m3)
        self.assertEqual(len(resolved.revision_metadata), 2)

    def test_units_support_mass_concentration_only(self):
        self.assertEqual(spike.convert_pm25_to_ug_m3(5, "µg/m³"), 5)
        self.assertEqual(spike.convert_pm25_to_ug_m3(5, "ug/m3"), 5)
        self.assertEqual(spike.convert_pm25_to_ug_m3(0.5, "mg/m³"), 500)
        with self.assertRaises(ValueError):
            spike.convert_pm25_to_ug_m3(5, "ppm")


class GapAndTimezoneTests(unittest.TestCase):
    def test_gap_detection_reports_maximal_runs_and_only_long_gaps_over_twelve_hours(self):
        start = utc_hour(2024, 1, 1)
        expected = spike.expected_hourly_grid(start, start + timedelta(hours=30))
        observed = set(expected)
        for hour in expected[3:4] + expected[10:22] + expected[25:]:
            observed.remove(hour)

        gaps = spike.detect_gaps(expected, observed)

        self.assertEqual([gap.hours for gap in gaps], [1, 12, 5])
        self.assertEqual(spike.longest_gap_hours(gaps), 12)
        self.assertEqual(spike.long_gaps(gaps), [])

    def test_thirteen_hour_gap_is_reported_individually(self):
        start = utc_hour(2024, 1, 1)
        expected = spike.expected_hourly_grid(start, start + timedelta(hours=30))
        observed = set(expected) - set(expected[5:18])

        gaps = spike.detect_gaps(expected, observed)

        self.assertEqual(len(spike.long_gaps(gaps)), 1)
        self.assertEqual(spike.long_gaps(gaps)[0].hours, 13)

    def test_timezone_normalization_rejects_naive_without_source_timezone(self):
        hanoi_time = datetime(2024, 1, 1, tzinfo=HANOI)

        self.assertEqual(spike.normalize_to_utc(hanoi_time), utc_hour(2023, 12, 31, 17))
        with self.assertRaises(ValueError):
            spike.normalize_to_utc(datetime(2024, 1, 1))
        self.assertEqual(
            spike.normalize_to_utc(datetime(2024, 1, 1), source_timezone=HANOI),
            utc_hour(2023, 12, 31, 17),
        )


class WeatherJoinTests(unittest.TestCase):
    def test_exact_join_requires_all_core_variables_but_not_optional_wind_direction(self):
        hour = utc_hour(2024, 1, 1)
        pm25_hours = [hour]
        weather = {
            hour: {
                "temperature_2m": 20,
                "relative_humidity_2m": 80,
                "wind_speed_10m": 4,
                "precipitation": 0,
                "surface_pressure": 1010,
            }
        }

        result = spike.weather_join_metrics(pm25_hours, weather, CORE_WEATHER_VARIABLES)

        self.assertEqual(result.complete_core_weather_hours, 1)
        self.assertEqual(result.weather_join_coverage_pct, 100.0)
        self.assertIsNone(result.optional_coverage["wind_direction_10m"])

    def test_nearest_or_forward_filled_weather_values_do_not_join(self):
        hour = utc_hour(2024, 1, 1)
        complete = dict.fromkeys(CORE_WEATHER_VARIABLES, 1)
        weather = {hour + timedelta(hours=1): complete, hour - timedelta(hours=1): complete}

        result = spike.weather_join_metrics([hour], weather, CORE_WEATHER_VARIABLES)

        self.assertEqual(result.complete_core_weather_hours, 0)
        self.assertEqual(result.weather_join_coverage_pct, 0.0)

    def test_each_missing_core_variable_makes_hour_incomplete(self):
        hour = utc_hour(2024, 1, 1)

        for variable in CORE_WEATHER_VARIABLES:
            with self.subTest(variable=variable):
                incomplete = dict.fromkeys(CORE_WEATHER_VARIABLES, 1)
                incomplete[variable] = None

                result = spike.weather_join_metrics([hour], {hour: incomplete}, CORE_WEATHER_VARIABLES)

                self.assertEqual(result.complete_core_weather_hours, 0)
                self.assertEqual(result.weather_join_coverage_pct, 0.0)

    def test_duplicate_weather_timestamp_fails_and_zero_denominator_is_undefined(self):
        hour = utc_hour(2024, 1, 1)
        duplicate = spike.weather_join_metrics([hour], {hour: [dict.fromkeys(CORE_WEATHER_VARIABLES, 1)] * 2}, CORE_WEATHER_VARIABLES)
        empty = spike.weather_join_metrics([], {}, CORE_WEATHER_VARIABLES)

        self.assertEqual(duplicate.invalid_duplicate_weather_hours, 1)
        self.assertIsNone(empty.weather_join_coverage_pct)

    def test_weather_gate_threshold(self):
        self.assertTrue(spike.weather_gate_passes(95.0))
        self.assertFalse(spike.weather_gate_passes(94.999))


class HashingAndVintageTests(unittest.TestCase):
    def test_raw_and_canonical_hashing_contracts(self):
        self.assertEqual(spike.raw_payload_hash(b"same"), spike.raw_payload_hash(b"same"))
        self.assertNotEqual(spike.raw_payload_hash(b"same"), spike.raw_payload_hash(b"changed"))
        self.assertEqual(spike.raw_payload_hash(b"same"), sha256(b"same").hexdigest())

        record_a = {"event_time": "2024-01-01T00:00:00Z", "value": 1, "unit": "ug/m3"}
        record_b = {"event_time": "2024-01-01T01:00:00Z", "value": 2, "unit": "ug/m3"}
        reordered_keys = {"unit": "ug/m3", "value": 1, "event_time": "2024-01-01T00:00:00Z"}
        changed = {"event_time": "2024-01-01T01:00:00Z", "value": 3, "unit": "ug/m3"}

        self.assertEqual(spike.canonical_payload_hash([record_a]), spike.canonical_payload_hash([reordered_keys]))
        self.assertEqual(spike.canonical_payload_hash([record_a, record_b]), spike.canonical_payload_hash([record_b, record_a]))
        self.assertNotEqual(spike.canonical_payload_hash([record_a, record_b]), spike.canonical_payload_hash([record_a, changed]))

    def test_vintage_comparison_keeps_classes_separate_and_zero_changes_is_not_proven(self):
        baseline = [
            {"id": "same", "value": 1, "metadata": {"owner": "A"}},
            {"id": "revised", "value": 1, "metadata": {"owner": "A"}},
            {"id": "missing", "value": 1, "metadata": {"owner": "A"}},
            {"id": "metadata", "value": 1, "metadata": {"owner": "A"}},
        ]
        current = [
            {"id": "same", "value": 1, "metadata": {"owner": "A"}},
            {"id": "revised", "value": 2, "metadata": {"owner": "A"}},
            {"id": "backfill", "value": 1, "metadata": {"owner": "A"}},
            {"id": "metadata", "value": 1, "metadata": {"owner": "B"}},
        ]

        result = spike.compare_vintages(baseline, current)
        unchanged = spike.compare_vintages(baseline[:1], baseline[:1])

        self.assertEqual(result.unchanged_count, 1)
        self.assertEqual(result.revised_count, 1)
        self.assertEqual(result.backfilled_count, 1)
        self.assertEqual(result.missing_count, 1)
        self.assertEqual(result.metadata_change_count, 1)
        self.assertNotEqual(unchanged.point_in_time_confidence, "PROVEN")
        self.assertFalse(unchanged.claims_immutable)


class PointInTimeAndGateTests(unittest.TestCase):
    def test_point_in_time_eligibility_requires_event_availability_and_ingestion(self):
        prediction = utc_hour(2024, 1, 2)
        valid = {"event_time": utc_hour(2024, 1, 1), "available_time": utc_hour(2024, 1, 1, 1), "ingested_at": utc_hour(2024, 1, 1, 2)}
        late_available = {"event_time": utc_hour(2024, 1, 1), "available_time": utc_hour(2024, 1, 3), "ingested_at": utc_hour(2024, 1, 3)}
        late_ingested = {"event_time": utc_hour(2024, 1, 1), "available_time": utc_hour(2024, 1, 1, 1), "ingested_at": utc_hour(2024, 1, 3)}
        unknown_available = {"event_time": utc_hour(2024, 1, 1), "available_time": None, "provider_updated_at": utc_hour(2024, 1, 1, 1)}

        self.assertTrue(spike.point_in_time_eligible(valid, prediction, require_ingested=True))
        self.assertFalse(spike.point_in_time_eligible(late_available, prediction, require_ingested=True))
        future_event = {"event_time": utc_hour(2024, 1, 3), "available_time": utc_hour(2024, 1, 1, 1), "ingested_at": utc_hour(2024, 1, 1, 2)}

        self.assertFalse(spike.point_in_time_eligible(late_ingested, prediction, require_ingested=True))
        self.assertFalse(spike.point_in_time_eligible(unknown_available, prediction, require_ingested=False))
        self.assertFalse(spike.point_in_time_eligible(future_event, prediction, require_ingested=True))

    def test_later_qa_revision_can_be_target_but_not_earlier_feature(self):
        prediction = utc_hour(2024, 1, 2)
        revision = {
            "event_time": utc_hour(2024, 1, 1),
            "available_time": utc_hour(2024, 1, 3),
            "provider_updated_at": utc_hour(2024, 1, 3),
            "canonical": True,
        }

        self.assertTrue(spike.retrospective_target_eligible(revision))
        self.assertFalse(spike.point_in_time_eligible(revision, prediction, require_ingested=False))

    def test_each_structural_failure_independently_blocks_pass_and_near_miss(self):
        numeric = spike.NumericGateResult(82.0, 85.0, 95.0)
        failures = (
            "has_legitimate_twelve_month_interval",
            "has_complete_contained_winter",
            "winter_quality_passes",
            "units_convertible",
            "duplicate_treatment_valid",
            "long_gaps_reported",
            "licensing_attribution_satisfied",
            "weather_core_supported",
        )

        for field in failures:
            with self.subTest(structural_requirement=field):
                structural = replace(spike.StructuralGateResult.all_passing(), **{field: False})
                result = spike.evaluate_stage0_gates(structural, numeric)

                self.assertFalse(result.pass_eligible)
                self.assertFalse(result.near_miss_eligible)
                self.assertEqual(result.numeric_gate_count, 3)

    def test_numeric_gates_and_near_miss_boundaries(self):
        structural = spike.StructuralGateResult.all_passing()

        self.assertTrue(spike.evaluate_stage0_gates(structural, spike.NumericGateResult(85.0, 85.0, 95.0)).pass_eligible)
        self.assertTrue(spike.evaluate_stage0_gates(structural, spike.NumericGateResult(80.0, 85.0, 95.0)).near_miss_eligible)
        self.assertFalse(spike.evaluate_stage0_gates(structural, spike.NumericGateResult(79.999, 85.0, 95.0)).near_miss_eligible)
        self.assertTrue(spike.evaluate_stage0_gates(structural, spike.NumericGateResult(85.0, 80.0, 95.0)).near_miss_eligible)
        self.assertFalse(spike.evaluate_stage0_gates(structural, spike.NumericGateResult(85.0, 79.999, 95.0)).near_miss_eligible)
        self.assertTrue(spike.evaluate_stage0_gates(structural, spike.NumericGateResult(85.0, 85.0, 90.0)).near_miss_eligible)
        self.assertFalse(spike.evaluate_stage0_gates(structural, spike.NumericGateResult(85.0, 85.0, 89.999)).near_miss_eligible)
        self.assertFalse(spike.evaluate_stage0_gates(structural, spike.NumericGateResult(80.0, 80.0, 95.0)).near_miss_eligible)
        self.assertFalse(spike.evaluate_stage0_gates(structural, spike.NumericGateResult(None, 85.0, 95.0)).near_miss_eligible)


if __name__ == "__main__":
    unittest.main()
