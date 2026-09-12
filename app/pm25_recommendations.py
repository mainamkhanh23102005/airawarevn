from dataclasses import dataclass
from types import MappingProxyType

from app.pm25_categories import PM25ForecastCategory


GUIDANCE_ID = "airaware_pm25_forecast_guidance_v1"


@dataclass(frozen=True)
class AirQualityRecommendation:
    category: PM25ForecastCategory
    message: str
    sensitive_group_message: str | None
    guidance_id: str


_RECOMMENDATIONS = MappingProxyType({
    PM25ForecastCategory.GOOD: AirQualityRecommendation(
        PM25ForecastCategory.GOOD,
        "For the forecast period: outdoor activities can proceed normally.",
        None,
        GUIDANCE_ID,
    ),
    PM25ForecastCategory.MODERATE: AirQualityRecommendation(
        PM25ForecastCategory.MODERATE,
        "For the forecast period: outdoor activities can proceed normally.",
        "For the forecast period: reduce time outdoors and strenuous activity.",
        GUIDANCE_ID,
    ),
    PM25ForecastCategory.POOR: AirQualityRecommendation(
        PM25ForecastCategory.POOR,
        "For the forecast period: reduce time spent outdoors.",
        "For the forecast period: limit outdoor and strenuous activity; favor lighter activity and more rest.",
        GUIDANCE_ID,
    ),
    PM25ForecastCategory.BAD: AirQualityRecommendation(
        PM25ForecastCategory.BAD,
        "For the forecast period: limit outdoor and strenuous activity.",
        "For the forecast period: avoid outdoor activity; choose indoor activity.",
        GUIDANCE_ID,
    ),
    PM25ForecastCategory.VERY_BAD: AirQualityRecommendation(
        PM25ForecastCategory.VERY_BAD,
        "For the forecast period: avoid prolonged outdoor and strenuous activity; choose indoor activity.",
        "For the forecast period: avoid all outdoor activity; choose indoor activity.",
        GUIDANCE_ID,
    ),
    PM25ForecastCategory.HAZARDOUS: AirQualityRecommendation(
        PM25ForecastCategory.HAZARDOUS,
        "For the forecast period: avoid outdoor activity; choose indoor activity.",
        None,
        GUIDANCE_ID,
    ),
})


def recommendation_for_category(category: PM25ForecastCategory) -> AirQualityRecommendation:
    if not isinstance(category, PM25ForecastCategory):
        raise TypeError("category must be a PM25ForecastCategory instance")
    return _RECOMMENDATIONS[category]
