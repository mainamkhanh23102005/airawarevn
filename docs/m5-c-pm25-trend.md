# M5-C: PM2.5 forecast trend

## Scope

`app.pm25_trend` compares current PM2.5 with one forecast PM2.5 value. Formula:

```text
change_ug_m3 = forecast_pm25 - current_pm25
```

Lower forecast pollution is `improving`; higher forecast pollution is
`worsening`. Stable is strictly `-0.05 < change_ug_m3 < 0.05`. At `-0.05` trend
is Improving; at `0.05` trend is Worsening. Difference is raw, with no rounding.
Exact zero returns canonical `0.0`, never `-0.0`.

`0.05` follows existing one-decimal frontend presentation behavior in
`app/index.html:146-153`. It is product display policy only. It is not an
official VN AQI threshold, measurement-uncertainty claim, health standard, or
percentage.

## Contract and versioning

```python
POLICY_ID = "airaware_pm25_forecast_trend_v1"
STABLE_CHANGE_THRESHOLD_UG_M3 = 0.05

class ForecastTrend(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    WORSENING = "worsening"

@dataclass(frozen=True)
class PM25Trend:
    trend: ForecastTrend
    change_ug_m3: int | float
    policy_id: str


def calculate_pm25_trend(current_pm25: int | float,
                          forecast_pm25: int | float) -> PM25Trend:
    ...
```

Only actual `int` and `float` instances are accepted. `bool` is rejected before
numeric checks. `None`, strings, complex numbers, `Decimal`, `Fraction`,
containers, unrelated objects, and enums raise `TypeError`; no coercion occurs.
Negative values and nonfinite floats raise `ValueError`. Extreme mixed
int/float subtraction that overflows raises `ValueError("change must be finite")`.

`POLICY_ID` versions this policy. Any semantic, threshold, or boundary change
requires new policy version.

## Exclusions

No category, recommendation, model, API, frontend, database, network,
persistence, or integration changes. No official AQI calculation, health
recommendation, percentage calculation, or rounding.

## Verification

`tests/test_pm25_trend.py` covers directions, zeros, strict boundaries,
`math.nextafter` neighbors, integer/float inputs, invalid values/types, frozen
results, deterministic canonical zero, huge integers, mixed arithmetic overflow,
source AST restrictions, and isolated `-I -S -B` import audit.

```text
.venv-wsl-main/bin/python -m unittest tests.test_pm25_categories tests.test_pm25_recommendations tests.test_pm25_trend
.venv-wsl-main/bin/python -m unittest discover -s tests
.venv-wsl-main/bin/python -m compileall -q app scripts tests
git diff --check
git diff --no-index --check /dev/null app/pm25_trend.py
git diff --no-index --check /dev/null tests/test_pm25_trend.py
git diff --no-index --check /dev/null docs/m5-c-pm25-trend.md
git status --short
```
