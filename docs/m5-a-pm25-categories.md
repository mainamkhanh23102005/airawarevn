# M5-A: PM2.5 forecast categories

## Framework source and interpretation

Framework source: Decision **1459/QĐ-TCMT**, dated **12 November 2019**,
ban hành **Hướng dẫn kỹ thuật tính toán và công bố chỉ số chất lượng không khí
Việt Nam (VN_AQI)**, Tổng cục Môi trường. The decision, concentration/index
anchors, and official category bands below were verified by the user.
No unverified source URL is supplied.

Official hourly PM2.5 AQI uses **Nowcast from up to 12 recent hours** of
observations, subject to the framework's data requirements. A single forecast
concentration is not that observed Nowcast series.

AirAware's **+6h forecast point concentration** estimates PM2.5 for **one
one-hour target interval six hours ahead**, not a six-hour average. This module
assigns an **aligned forecast category, NOT official VN_AQI**. It computes and
returns **no numeric AQI**, no official hourly AQI, and no daily AQI. The official
index numbers below document the source framework only, not module output.

## User-verified framework tables

PM2.5 concentration is in µg/m³.

| Concentration anchor | Index anchor |
|---:|---:|
| 0 | 0 |
| 25 | 50 |
| 50 | 100 |
| 80 | 150 |
| 150 | 200 |
| 250 | 300 |
| 350 | 400 |
| ≥500 | 500 |

| Official index band | English label | Vietnamese label |
|---|---|---|
| 0–50 | Good | Tốt |
| 51–100 | Moderate | Trung bình |
| 101–150 | Poor | Kém |
| 151–200 | Bad | Xấu |
| 201–300 | Very bad | Rất xấu |
| 301–500 | Hazardous | Nguy hại |

## Approved direct forecast mapping

| Forecast concentration (µg/m³) | Stable category value | Returned English label |
|---|---|---|
| [0, 25] | `good` | Good |
| (25, 50] | `moderate` | Moderate |
| (50, 80] | `poor` | Poor |
| (80, 150] | `bad` | Bad |
| (150, 250] | `very_bad` | Very bad |
| (250, infinity) | `hazardous` | Hazardous |

All finite concentrations above 250 are Hazardous, including 350, 500, and
values above 500. The 350 and 500 anchors do not introduce additional
categories. Infinity itself is invalid; the last interval has no finite upper
bound. Boundaries use the original concentration directly, with **no rounding,
interpolation, clipping, or numeric AQI conversion**. The original concentration
is untouched, including the sign of negative zero.

## Python contract

`app.pm25_categories` provides:

- `PM25ForecastCategory(str, Enum)`: `GOOD`, `MODERATE`, `POOR`, `BAD`,
  `VERY_BAD`, `HAZARDOUS`, with the stable snake_case values above.
- Frozen dataclass `PM25Classification(category, label, standard_id)`, containing
  a `PM25ForecastCategory` and two strings; no concentration or numeric AQI field.
- `classify_pm25_forecast(concentration: int | float) -> PM25Classification`.
- `STANDARD_ID = "vn_1459_2019_pm25_forecast_aligned_v1"`, returned for every
  valid classification. This explicitly versioned identifier names the project's
  forecast-aligned policy, not an official AQI product. Mapping or semantic
  changes require a new policy version rather than silently redefining v1.

Input policy:

- Accept only `int` and `float` instances, with `bool` rejected before the
  numeric type check. No string parsing or other numeric coercion occurs.
- `None`, booleans, strings, bytes, complex numbers, `Decimal`, `Fraction`,
  containers, and other unsupported types raise `TypeError`.
- Negative numbers and nonfinite floats (`nan`, `inf`, `-inf`) raise `ValueError`.
- `0`, `0.0`, and `-0.0` are valid and Good.
- Arbitrarily large positive integers are valid and Hazardous. Integers are
  compared directly, never converted to float or passed to `math.isfinite`,
  avoiding integer-to-float overflow. Negative integers remain invalid regardless
  of size.

The function is deterministic and uses only the Python standard library. It
imports no model, database, network, monitoring, or API modules and performs no
initialization of those systems. M5-A adds this standalone utility only; it does
not change monitoring, API responses, forecasting, persistence, or model behavior.

## Verification

`tests/test_pm25_categories.py` covers representative integer/float values,
all category boundaries and their immediate `math.nextafter` neighbors, zero
and its invalid negative neighbor, the 350/500 anchors and neighbors, values
above 500, huge integers, invalid inputs, exact identifiers/labels, deterministic
results, unchanged input, and frozen fields.

An isolated subprocess (`-I -S -B`) imports and exercises the module with
standard-library-only import guards, blocks other application/model/API imports,
blocks database/network imports and audit events, and rejects non-source file
access or file writes. This avoids relying on modules already initialized by
other tests or on installed third-party dependencies.

```text
python -m unittest tests.test_pm25_categories
python -m unittest discover -s tests
python -m compileall -q app scripts tests
git diff --check
```
