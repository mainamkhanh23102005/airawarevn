import ast
import math
import subprocess
import sys
import textwrap
import unittest
from dataclasses import FrozenInstanceError, fields
from decimal import Decimal
from enum import Enum, IntEnum
from fractions import Fraction
from pathlib import Path
from typing import get_type_hints

from app.pm25_trend import (
    POLICY_ID,
    STABLE_CHANGE_THRESHOLD_UG_M3,
    ForecastTrend,
    PM25Trend,
    calculate_pm25_trend,
)


class PM25TrendTests(unittest.TestCase):
    def test_contract_and_policy_values(self):
        self.assertEqual([member.value for member in ForecastTrend],
                         ["improving", "stable", "worsening"])
        self.assertEqual(POLICY_ID, "airaware_pm25_forecast_trend_v1")
        self.assertEqual(STABLE_CHANGE_THRESHOLD_UG_M3, 0.05)
        self.assertEqual([field.name for field in fields(PM25Trend)],
                         ["trend", "change_ug_m3", "policy_id"])
        self.assertEqual(get_type_hints(PM25Trend), {
            "trend": ForecastTrend,
            "change_ug_m3": int | float,
            "policy_id": str,
        })
        self.assertEqual(get_type_hints(calculate_pm25_trend), {
            "current_pm25": int | float,
            "forecast_pm25": int | float,
            "return": PM25Trend,
        })

    def test_clear_directions_and_both_zero(self):
        cases = (
            (10, 9, ForecastTrend.IMPROVING, -1),
            (9, 10, ForecastTrend.WORSENING, 1),
            (0, 0, ForecastTrend.STABLE, 0.0),
            (0.0, -0.0, ForecastTrend.STABLE, 0.0),
        )
        for current, forecast, trend, change in cases:
            with self.subTest(current=current, forecast=forecast):
                result = calculate_pm25_trend(current, forecast)
                self.assertIs(result.trend, trend)
                self.assertEqual(result.change_ug_m3, change)
                self.assertEqual(result.policy_id, POLICY_ID)

    def test_inside_threshold_on_both_signs(self):
        for current, forecast in ((10.0, 10.04), (10.0, 9.96)):
            with self.subTest(current=current, forecast=forecast):
                result = calculate_pm25_trend(current, forecast)
                self.assertIs(result.trend, ForecastTrend.STABLE)
                self.assertEqual(result.change_ug_m3, forecast - current)

    def test_exact_thresholds(self):
        for current, forecast, trend in ((1.0, 0.95, ForecastTrend.IMPROVING),
                                         (1.0, 1.05, ForecastTrend.WORSENING)):
            with self.subTest(current=current, forecast=forecast):
                self.assertIs(calculate_pm25_trend(current, forecast).trend, trend)

    def test_nextafter_neighbors_of_thresholds(self):
        threshold = STABLE_CHANGE_THRESHOLD_UG_M3
        cases = (
            (math.nextafter(threshold, 0.0), ForecastTrend.STABLE),
            (threshold, ForecastTrend.WORSENING),
            (math.nextafter(threshold, math.inf), ForecastTrend.WORSENING),
            (-math.nextafter(threshold, 0.0), ForecastTrend.STABLE),
            (-threshold, ForecastTrend.IMPROVING),
            (-math.nextafter(threshold, math.inf), ForecastTrend.IMPROVING),
        )
        for change, trend in cases:
            with self.subTest(change=change):
                current, forecast = (0.0, change) if change >= 0 else (-change, 0.0)
                result = calculate_pm25_trend(current, forecast)
                self.assertIs(result.trend, trend)

    def test_integer_float_and_mixed_inputs(self):
        for current, forecast, trend, change in ((4, 4.0, ForecastTrend.STABLE, 0.0),
                                                  (4.0, 5, ForecastTrend.WORSENING, 1.0),
                                                  (5, 4.0, ForecastTrend.IMPROVING, -1.0)):
            with self.subTest(current=current, forecast=forecast):
                result = calculate_pm25_trend(current, forecast)
                self.assertIs(result.trend, trend)
                self.assertEqual(result.change_ug_m3, change)

    def test_invalid_values_rejected(self):
        for current, forecast in ((-1, 0), (0, -1), (math.nan, 0), (0, math.nan),
                                  (math.inf, 0), (0, math.inf), (-math.inf, 0),
                                  (0, -math.inf)):
            with self.subTest(current=current, forecast=forecast):
                with self.assertRaises(ValueError):
                    calculate_pm25_trend(current, forecast)

    def test_invalid_types_rejected_without_coercion(self):
        class NumericEnum(Enum):
            VALUE = 1

        class IntegerEnum(IntEnum):
            VALUE = 1

        class CustomInt(int):
            pass

        class CustomFloat(float):
            pass

        class Coercible:
            def __float__(self):
                raise AssertionError("must not coerce")

            def __int__(self):
                raise AssertionError("must not coerce")

        invalid = (None, True, False, "1", b"1", 1j, Decimal("1"), Fraction(1),
                   [], {}, (), set(), object(), NumericEnum.VALUE, IntegerEnum.VALUE,
                   CustomInt(1), CustomFloat(1.0), Coercible())
        for value in invalid:
            for current, forecast in ((value, 0), (0, value)):
                with self.subTest(value=value, current=current, forecast=forecast):
                    with self.assertRaises(TypeError):
                        calculate_pm25_trend(current, forecast)

    def test_results_are_frozen_deterministic_and_canonical_zero(self):
        result = calculate_pm25_trend(10, 10)
        self.assertEqual(result.change_ug_m3, 0.0)
        self.assertEqual(math.copysign(1.0, result.change_ug_m3), 1.0)
        for field in fields(result):
            with self.subTest(field=field.name):
                with self.assertRaises(FrozenInstanceError):
                    setattr(result, field.name, None)
        expected = calculate_pm25_trend(4.0, 5.0)
        for _ in range(10):
            self.assertEqual(calculate_pm25_trend(4.0, 5.0), expected)

    def test_extreme_mixed_subtraction_becomes_value_error(self):
        huge = 10 ** 10000
        for current, forecast in ((huge, 0.0), (0.0, huge)):
            with self.subTest(current_type=type(current), forecast_type=type(forecast)):
                with self.assertRaisesRegex(ValueError, "change must be finite"):
                    calculate_pm25_trend(current, forecast)
        result = calculate_pm25_trend(huge, huge)
        self.assertIs(result.trend, ForecastTrend.STABLE)
        self.assertEqual(result.change_ug_m3, 0.0)

    def test_source_contract_and_isolation(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "app" / "pm25_trend.py").read_text(encoding="utf-8"))
        imports = []
        numeric_thresholds = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                imports.append(node.module)
            if isinstance(node, ast.Constant) and node.value == 0.05:
                numeric_thresholds.append(node)
            if isinstance(node, ast.Compare):
                for comparator in node.comparators:
                    if isinstance(comparator, ast.Constant) and comparator.value == 0.05:
                        self.fail("comparisons must use named threshold")
        self.assertEqual(imports, ["math", "dataclasses", "enum"])
        self.assertEqual(len(numeric_thresholds), 1)
        code = textwrap.dedent("""
            import builtins
            import importlib.abc
            import sys

            sys.path.insert(0, sys.argv[1])
            allowed_app = {"app", "app.pm25_trend"}
            forbidden = {"sqlite3", "socket", "_socket", "ssl", "http", "urllib",
                         "dbm", "shelve", "pickle", "requests", "fastapi", "pandas",
                         "numpy", "sklearn", "joblib"}

            def check(name):
                root = name.split(".", 1)[0]
                if name in allowed_app:
                    return
                if name.startswith("app.") or root == "scripts":
                    raise AssertionError("forbidden project import: " + name)
                if root in forbidden:
                    raise AssertionError("forbidden dependency: " + name)

            class Guard(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    check(fullname)

            original_import = builtins.__import__
            def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                if level == 0:
                    check(name)
                return original_import(name, globals, locals, fromlist, level)

            def audit(event, args):
                if event.startswith(("socket.", "sqlite3.", "subprocess.", "os.system")):
                    raise AssertionError("forbidden side effect: " + event)
                if event == "open":
                    path = args[0]
                    if not isinstance(path, str) or not path.endswith((".py", ".pyc")):
                        raise AssertionError("non-source file access")
                    if args[1] not in ("r", "rb"):
                        raise AssertionError("file write")

            sys.meta_path.insert(0, Guard())
            builtins.__import__ = guarded_import
            sys.addaudithook(audit)
            from app.pm25_trend import calculate_pm25_trend
            assert calculate_pm25_trend(2, 1).trend.value == "improving"
            assert set(name for name in sys.modules if name == "app" or name.startswith("app.")) == allowed_app
            print("isolated import ok")
        """)
        completed = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c", code, str(root)],
            capture_output=True, text=True, timeout=30, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(completed.stdout.strip(), "isolated import ok")
        self.assertEqual(completed.stderr, "")


if __name__ == "__main__":
    unittest.main()
