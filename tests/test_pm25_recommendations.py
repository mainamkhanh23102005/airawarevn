import ast
import subprocess
import sys
import textwrap
import unittest
from dataclasses import FrozenInstanceError, fields
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import get_type_hints

from app.pm25_categories import PM25ForecastCategory, classify_pm25_forecast
from app.pm25_recommendations import (
    GUIDANCE_ID,
    AirQualityRecommendation,
    _RECOMMENDATIONS,
    recommendation_for_category,
)


CASES = (
    (PM25ForecastCategory.GOOD,
     "For the forecast period: outdoor activities can proceed normally.", None),
    (PM25ForecastCategory.MODERATE,
     "For the forecast period: outdoor activities can proceed normally.",
     "For the forecast period: reduce time outdoors and strenuous activity."),
    (PM25ForecastCategory.POOR,
     "For the forecast period: reduce time spent outdoors.",
     "For the forecast period: limit outdoor and strenuous activity; favor lighter activity and more rest."),
    (PM25ForecastCategory.BAD,
     "For the forecast period: limit outdoor and strenuous activity.",
     "For the forecast period: avoid outdoor activity; choose indoor activity."),
    (PM25ForecastCategory.VERY_BAD,
     "For the forecast period: avoid prolonged outdoor and strenuous activity; choose indoor activity.",
     "For the forecast period: avoid all outdoor activity; choose indoor activity."),
    (PM25ForecastCategory.HAZARDOUS,
     "For the forecast period: avoid outdoor activity; choose indoor activity.", None),
)


class PM25RecommendationTests(unittest.TestCase):
    def test_exact_policy_for_all_six_categories(self):
        self.assertEqual({case[0] for case in CASES}, set(PM25ForecastCategory))
        self.assertEqual(GUIDANCE_ID, "airaware_pm25_forecast_guidance_v1")
        for category, message, sensitive in CASES:
            with self.subTest(category=category):
                result = recommendation_for_category(category)
                self.assertIsInstance(result, AirQualityRecommendation)
                self.assertIs(result.category, category)
                self.assertEqual(result.message, message)
                self.assertEqual(result.sensitive_group_message, sensitive)
                self.assertEqual(result.guidance_id, "airaware_pm25_forecast_guidance_v1")

    def test_contract_fields_and_annotations(self):
        expected = {"category": PM25ForecastCategory, "message": str,
                    "sensitive_group_message": str | None, "guidance_id": str}
        self.assertEqual([field.name for field in fields(AirQualityRecommendation)],
                         list(expected))
        self.assertEqual(get_type_hints(AirQualityRecommendation), expected)
        self.assertEqual(get_type_hints(recommendation_for_category),
                         {"category": PM25ForecastCategory,
                          "return": AirQualityRecommendation})

    def test_none_means_general_guidance(self):
        for category, message, sensitive in CASES:
            result = recommendation_for_category(category)
            if category in (PM25ForecastCategory.GOOD, PM25ForecastCategory.HAZARDOUS):
                self.assertIsNone(result.sensitive_group_message)
                self.assertEqual(result.sensitive_group_message or result.message, message)
            else:
                self.assertIsNotNone(result.sensitive_group_message)
                self.assertNotEqual(sensitive, message)

    def test_wrong_types_rejected_without_coercion(self):
        class OtherCategory(str, Enum):
            GOOD = "good"

        class NumberCategory(Enum):
            GOOD = 1

        class Coercible:
            def __str__(self):
                raise AssertionError("must not coerce")

        values = (None, True, False, 0, 1, 1.0, float("nan"), float("inf"),
                  1j, Decimal("1"), Fraction(1), b"good", "", "GOOD",
                  OtherCategory.GOOD, NumberCategory.GOOD,
                  classify_pm25_forecast(12), [], {}, (), set(),
                  [PM25ForecastCategory.GOOD], {"category": "good"},
                  object(), Coercible()) + tuple(category.value for category in PM25ForecastCategory)
        for value in values:
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    recommendation_for_category(value)

    def test_results_are_frozen(self):
        for category in PM25ForecastCategory:
            result = recommendation_for_category(category)
            for field in fields(result):
                with self.subTest(category=category, field=field.name):
                    with self.assertRaises(FrozenInstanceError):
                        setattr(result, field.name, None)
                    with self.assertRaises(FrozenInstanceError):
                        delattr(result, field.name)

    def test_lookup_is_immutable_and_complete(self):
        self.assertIsInstance(_RECOMMENDATIONS, MappingProxyType)
        self.assertEqual(set(_RECOMMENDATIONS), set(PM25ForecastCategory))
        for category in PM25ForecastCategory:
            self.assertIs(_RECOMMENDATIONS[category], recommendation_for_category(category))
            with self.assertRaises(TypeError):
                _RECOMMENDATIONS[category] = None
            with self.assertRaises(TypeError):
                del _RECOMMENDATIONS[category]

    def test_determinism(self):
        expected = {category: recommendation_for_category(category)
                    for category in PM25ForecastCategory}
        for _ in range(10):
            for category in reversed(tuple(PM25ForecastCategory)):
                self.assertEqual(recommendation_for_category(category), expected[category])

    def test_composition_with_m5_a(self):
        for concentration, (category, message, sensitive) in zip(
                (12, 40, 65, 100, 200, 300), CASES):
            classification = classify_pm25_forecast(concentration)
            result = recommendation_for_category(classification.category)
            self.assertIs(result.category, category)
            self.assertEqual((result.message, result.sensitive_group_message),
                             (message, sensitive))

    def test_source_has_only_category_import_and_no_thresholds_or_classifier_calls(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "app" / "pm25_recommendations.py").read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.fail("unexpected module import")
            if isinstance(node, ast.ImportFrom):
                imports.append((node.module, tuple(alias.name for alias in node.names)))
            if isinstance(node, ast.Constant):
                self.assertNotIn(type(node.value), (int, float, complex))
            if isinstance(node, ast.Compare):
                self.assertFalse(any(isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE))
                                     for op in node.ops))
            if isinstance(node, ast.Call):
                self.assertIsInstance(node.func, ast.Name)
                self.assertIn(node.func.id, {"dataclass", "MappingProxyType",
                                            "AirQualityRecommendation", "isinstance", "TypeError"})
        self.assertEqual(imports, [("dataclasses", ("dataclass",)),
                                   ("types", ("MappingProxyType",)),
                                   ("app.pm25_categories", ("PM25ForecastCategory",))])

    def test_isolated_import_has_no_external_or_initialization_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        code = textwrap.dedent("""
            import builtins
            import importlib.abc
            import sys

            sys.path.insert(0, sys.argv[1])
            allowed_app = {"app", "app.pm25_categories", "app.pm25_recommendations"}
            forbidden = {"sqlite3", "socket", "_socket", "ssl", "http", "urllib",
                         "dbm", "shelve", "pickle", "requests", "fastapi", "pandas",
                         "numpy", "sklearn", "joblib"}

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
            from app.pm25_categories import PM25ForecastCategory
            from app.pm25_recommendations import recommendation_for_category
            for category in PM25ForecastCategory:
                result = recommendation_for_category(category)
                assert result.category is category
                assert result.guidance_id == "airaware_pm25_forecast_guidance_v1"
                assert recommendation_for_category(category) == result
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
