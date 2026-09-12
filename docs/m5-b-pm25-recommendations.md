# M5-B: PM2.5 forecast recommendations

## Scope and interpretation

This standalone utility maps an existing M5-A `PM25ForecastCategory` to fixed
activity-planning text. Official hourly PM2.5 AQI uses **Nowcast from up to 12
recent hours of observations**, subject to official data requirements.
AirAware's **+6h one-hour target point forecast** is not that observed series and
is not a six-hour average. These are **aligned forecast planning recommendations,
not official calculated VN_AQI and not a government advisory**. No numeric AQI
is calculated. English messages are **product paraphrases, not official
translations**, and are not exhaustive reproductions of source health guidance.

Sensitive groups mean children, pregnant people, older adults, and people with
respiratory or cardiovascular conditions. These messages provide no
individualized medical advice.

## Sources and selected Vietnamese excerpts

### [1] Good only: Decision 1459/QĐ-TCMT

Decision **1459/QĐ-TCMT**, **12 November 2019**, Table 5, Tổng cục Môi trường.
The user directly verified the official source: Good allows both general and
sensitive groups outdoor activity freely:

> Tự do thực hiện các hoạt động ngoài trời

- [Official landing page](https://vea.mae.gov.vn/chat-luong-moi-truong-khong-khi/5872/chi-so-chat-luong-khong-khi-aqi)
- [Official decision PDF](https://vea.mae.gov.vn/Data/files/QD%201459%20TCMT%20ngay%2012_11_2019%20AQI(1).pdf)

Verification provenance for this quotation is the user's direct verification,
not an independent PDF extraction in this implementation session.

### [2] Moderate through Hazardous only: CDC Hà Nội

**Trung tâm Kiểm soát bệnh tật thành phố Hà Nội (CDC Hà Nội)**,
**18 December 2025**, [Khuyến cáo phòng, chống ảnh hưởng của ô nhiễm không khí tới sức khỏe cộng đồng](https://hanoicdc.gov.vn/2618n/khuyen-cao-phong-chong-anh-huong-cua-o-nhiem-khong-khi-toi-suc-khoe-cong-dong.html).
The article was successfully fetched during implementation. The following are
exact selected Vietnamese excerpts, not replacement translations. All
Moderate–Hazardous product messages rely exclusively on this source.

**Moderate / Trung bình**, general:

> Đối với người bình thường, có thể tham gia các hoạt động ngoài trời không hạn chế.

Sensitive, selected clause:

> Đối với những người nhạy cảm, cần giảm thời gian hoạt động ngoài trời và các hoạt động vận động cần gắng sức;

**Poor / Kém**, general, selected clause:

> Người bình thường cần giảm thời gian tham gia các hoạt động ngoài trời, đặc biệt là những người có triệu chứng đau mắt, ho, đau họng;

Sensitive, selected clause:

> Với những người nhạy cảm, cần hạn chế hoạt động ngoài trời và các hoạt động vận động cần gắng sức, tăng thời gian nghỉ ngơi và hoạt động nhẹ nhàng;

**Bad / Xấu**, general, selected clause:

> Người bình thường cần hạn chế hoạt động ngoài trời hoặc các hoạt động vận động cần gắng sức;

Sensitive:

> Đối với những người nhạy cảm, cần tránh các hoạt động ngoài trời, chuyển sang các hoạt động vận động trong nhà, hạn chế mở cửa sổ và theo dõi sức khỏe để kịp thời đi khám khi có triệu chứng bất thường.

**Very bad / Rất xấu**, general:

> Người bình thường cần tránh các hoạt động ngoài trời trong thời gian dài hoặc các hoạt động vận động cần gắng sức, khuyến khích thực hiện các hoạt động trong nhà.

Sensitive, selected clause:

> Đối với những người nhạy cảm, cần tránh tất cả các hoạt động ngoài trời, chuyển sang các hoạt động trong nhà hoặc lùi sang thời điểm khi chất lượng không khí tốt hơn;

**Hazardous / Nguy hại**, shared selected activity instruction:

> Cả người bình thường và những người nhạy cảm đều cần tránh các hoạt động ngoài trời, chuyển sang các hoạt động trong nhà hoặc sang thời điểm khác khi chất lượng không khí được cải thiện.

Sensitive-group definition, selected clause:

> Đối với những người nhạy cảm với các chất ô nhiễm trong không khí như trẻ em, phụ nữ mang thai, người mắc bệnh hô hấp, tim mạch và người cao tuổi, cần tránh tiếp xúc với các nguồn phát thải ô nhiễm;

## Exact v1 text policy

Every result has `guidance_id = "airaware_pm25_forecast_guidance_v1"`.
Text and punctuation in this table are the frozen product policy.

| Category | `message` | `sensitive_group_message` | Source |
|---|---|---|---|
| `GOOD` | For the forecast period: outdoor activities can proceed normally. | `None` | [1], Table 5, both groups |
| `MODERATE` | For the forecast period: outdoor activities can proceed normally. | For the forecast period: reduce time outdoors and strenuous activity. | [2], Trung bình |
| `POOR` | For the forecast period: reduce time spent outdoors. | For the forecast period: limit outdoor and strenuous activity; favor lighter activity and more rest. | [2], Kém |
| `BAD` | For the forecast period: limit outdoor and strenuous activity. | For the forecast period: avoid outdoor activity; choose indoor activity. | [2], Xấu |
| `VERY_BAD` | For the forecast period: avoid prolonged outdoor and strenuous activity; choose indoor activity. | For the forecast period: avoid all outdoor activity; choose indoor activity. | [2], Rất xấu |
| `HAZARDOUS` | For the forecast period: avoid outdoor activity; choose indoor activity. | `None` | [2], Nguy hại |

`None` means **use the general guidance**, not missing guidance and not absence
of risk. Good uses the same source instruction for both groups. Hazardous uses
the same core selected activity instruction; this does **not** assert that the
full guidance for both groups is identical.

The article's multi-day hazardous provisions (daily VN_AQI at least 301 for
three consecutive days) are excluded: a one-hour forecast point cannot establish
that condition. Trends, postponement based on predicted improvements, medical
procedures, API/frontend integration, model changes, and DB changes are out of
scope.

## Python contract and versioning

`app.pm25_recommendations` provides:

```python
GUIDANCE_ID = "airaware_pm25_forecast_guidance_v1"

@dataclass(frozen=True)
class AirQualityRecommendation:
    category: PM25ForecastCategory
    message: str
    sensitive_group_message: str | None
    guidance_id: str


def recommendation_for_category(category: PM25ForecastCategory) -> AirQualityRecommendation:
    ...
```

Only actual `PM25ForecastCategory` instances are accepted. Raw strings (including
`"good"`), `None`, booleans, numbers, `PM25Classification` objects, unrelated enums,
containers, and all other unsupported inputs raise `TypeError`; no coercion is
performed. Compose explicitly as
`recommendation_for_category(classify_pm25_forecast(value).category)`.
The recommendation module imports only the category from M5-A, not its classifier,
and defines no concentration thresholds.

A `MappingProxyType` lookup holds frozen dataclass results, with no retained
mutable backing-dictionary alias. Repeated calls are deterministic. The module
uses only standard-library dataclass/mapping support and the category import;
it performs no model/API/DB/network/environment/time or data-file access.
Normal Python source loading is not application data-file access.

This identifier versions AirAware's guidance policy, independently of M5-A's
classification `STANDARD_ID`. Changes to text, punctuation, source mapping,
meaning, or `None` semantics require a new guidance version; do not silently
redefine v1. Source revisions require review, not runtime fetching.

## Verification

`tests/test_pm25_recommendations.py` covers all six exact message pairs, category
identity, IDs, field names/types, `None` semantics, wrong types, frozen fields,
immutable complete lookup, determinism, and explicit M5-A composition. A focused
AST check permits only the intended imports and construction/type-check calls,
rejects numeric threshold constants and ordering comparisons, and prevents
classifier calls in the recommendation module.

The portable M5-A-style isolation test uses a fresh `-I -S -B` subprocess. It
allows only `app`, `app.pm25_categories`, and `app.pm25_recommendations` from the
application, rejects other application/scripts and forbidden dependency roots,
and audits network/DB/process side effects and non-source file access/writes.
Normal optional standard-library import probes remain allowed to fail normally;
there is no blanket restriction to modules already loaded in the parent process.

Run from the repository root in WSL Ubuntu-26.04 with the existing environment:

```text
.venv-wsl-main/bin/python -m unittest tests.test_pm25_categories tests.test_pm25_recommendations
.venv-wsl-main/bin/python -m unittest discover -s tests
.venv-wsl-main/bin/python -m compileall -q app scripts tests
git diff --check
git diff --no-index --check /dev/null app/pm25_recommendations.py
git diff --no-index --check /dev/null tests/test_pm25_recommendations.py
git diff --no-index --check /dev/null docs/m5-b-pm25-recommendations.md
git status --short
```
