import math
import subprocess
import sys
import textwrap
import unittest
from dataclasses import FrozenInstanceError, fields
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from app.pm25_categories import (
    PM25Classification,
    PM25ForecastCategory,
    classify_pm25_forecast,
)


class PM25CategoryTests(unittest.TestCase):
    def test_representative_values_and_stable_contract(self):
        cases = (
            (12, PM25ForecastCategory.GOOD, "good", "Good"),
            (40, PM25ForecastCategory.MODERATE, "moderate", "Moderate"),
            (65, PM25ForecastCategory.POOR, "poor", "Poor"),
            (100, PM25ForecastCategory.BAD, "bad", "Bad"),
            (200, PM25ForecastCategory.VERY_BAD, "very_bad", "Very bad"),
            (300, PM25ForecastCategory.HAZARDOUS, "hazardous", "Hazardous"),
        )
        self.assertEqual(len(PM25ForecastCategory), 6)
        self.assertEqual([field.name for field in fields(PM25Classification)],
                         ["category", "label", "standard_id"])
        for value, category, identifier, label in cases:
            for concentration in (value, float(value)):
                with self.subTest(concentration=concentration):
                    result = classify_pm25_forecast(concentration)
                    self.assertIsInstance(result, PM25Classification)
                    self.assertIs(result.category, category)
                    self.assertIsInstance(result.category, str)
                    self.assertEqual(result.category.value, identifier)
                    self.assertEqual(result.label, label)
                    self.assertEqual(result.standard_id,
                                     "vn_1459_2019_pm25_forecast_aligned_v1")

    def test_boundaries_and_immediate_neighbors(self):
        cases = (
            (25, PM25ForecastCategory.GOOD, PM25ForecastCategory.MODERATE),
            (50, PM25ForecastCategory.MODERATE, PM25ForecastCategory.POOR),
            (80, PM25ForecastCategory.POOR, PM25ForecastCategory.BAD),
            (150, PM25ForecastCategory.BAD, PM25ForecastCategory.VERY_BAD),
            (250, PM25ForecastCategory.VERY_BAD, PM25ForecastCategory.HAZARDOUS),
        )
        for boundary, lower, upper in cases:
            for value, expected in (
                (math.nextafter(boundary, -math.inf), lower),
                (boundary, lower),
                (float(boundary), lower),
                (math.nextafter(boundary, math.inf), upper),
            ):
                with self.subTest(value=value):
                    self.assertIs(classify_pm25_forecast(value).category, expected)

    def test_zero_and_immediate_neighbors(self):
        for value in (0, 0.0, -0.0, math.nextafter(0.0, math.inf)):
            with self.subTest(value=value):
                self.assertIs(classify_pm25_forecast(value).category,
                              PM25ForecastCategory.GOOD)
        with self.assertRaises(ValueError):
            classify_pm25_forecast(math.nextafter(0.0, -math.inf))

    def test_high_anchors_do_not_add_categories(self):
        for anchor in (350, 500):
            for value in (anchor, float(anchor), math.nextafter(anchor, -math.inf),
                          math.nextafter(anchor, math.inf)):
                with self.subTest(value=value):
                    self.assertIs(classify_pm25_forecast(value).category,
                                  PM25ForecastCategory.HAZARDOUS)

    def test_above_500_and_arbitrarily_large_integers(self):
        for value in (501, 1000.0, sys.float_info.max, 10 ** 10000):
            self.assertIs(classify_pm25_forecast(value).category,
                          PM25ForecastCategory.HAZARDOUS)
        with self.assertRaises(ValueError):
            classify_pm25_forecast(-(10 ** 10000))

    def test_negative_and_nonfinite_values_rejected(self):
        for value in (-1, -0.1, -500, math.inf, -math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    classify_pm25_forecast(value)

    def test_unsupported_types_rejected_without_coercion(self):
        class Coercible:
            def __float__(self):
                raise AssertionError("must not coerce")

            def __int__(self):
                raise AssertionError("must not coerce")

        for value in (None, True, False, "25", "", b"25", 25 + 0j,
                      Decimal("25"), Fraction(25), [], {}, object(), Coercible()):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    classify_pm25_forecast(value)

    def test_result_is_frozen(self):
        result = classify_pm25_forecast(25)
        for name, value in (("category", PM25ForecastCategory.BAD),
                            ("label", "Bad"), ("standard_id", "changed")):
            with self.subTest(field=name):
                with self.assertRaises(FrozenInstanceError):
                    setattr(result, name, value)

    def test_determinism_and_original_concentration_unchanged(self):
        concentration = math.nextafter(25.0, math.inf)
        original = concentration.hex()
        expected = classify_pm25_forecast(concentration)
        for _ in range(10):
            self.assertEqual(classify_pm25_forecast(concentration), expected)
        self.assertEqual(concentration.hex(), original)
        self.assertIs(expected.category, PM25ForecastCategory.MODERATE)
        negative_zero = -0.0
        classify_pm25_forecast(negative_zero)
        self.assertEqual(math.copysign(1.0, negative_zero), -1.0)

    def test_isolated_import_has_no_external_or_initialization_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        code = textwrap.dedent("""
            import builtins
            import importlib.abc
            import sys

            sys.path.insert(0, sys.argv[1])
            allowed_app = {"app", "app.pm25_categories"}
            forbidden = {"sqlite3", "socket", "_socket", "ssl", "http", "urllib",
                         "dbm", "shelve", "pickle"}

            def check(name):
                root = name.split(".", 1)[0]
                if name in allowed_app:
                    return
                if name.startswith("app."):
                    raise AssertionError("forbidden application import: " + name)
                if root == "scripts":
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
            from app.pm25_categories import classify_pm25_forecast
            result = classify_pm25_forecast(501)
            assert result.category.value == "hazardous"
            assert set(name for name in sys.modules if name == "app" or
                       name.startswith("app.")) == allowed_app
            assert not any(name == "scripts" or name.startswith("scripts.")
                           for name in sys.modules)
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
