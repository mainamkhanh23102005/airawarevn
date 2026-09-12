import math
from dataclasses import dataclass
from enum import Enum


STANDARD_ID = "vn_1459_2019_pm25_forecast_aligned_v1"


class PM25ForecastCategory(str, Enum):
    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"
    BAD = "bad"
    VERY_BAD = "very_bad"
    HAZARDOUS = "hazardous"


@dataclass(frozen=True)
class PM25Classification:
    category: PM25ForecastCategory
    label: str
    standard_id: str


_BANDS = (
    (25, PM25ForecastCategory.GOOD, "Good"),
    (50, PM25ForecastCategory.MODERATE, "Moderate"),
    (80, PM25ForecastCategory.POOR, "Poor"),
    (150, PM25ForecastCategory.BAD, "Bad"),
    (250, PM25ForecastCategory.VERY_BAD, "Very bad"),
)


def classify_pm25_forecast(concentration: int | float) -> PM25Classification:
    if isinstance(concentration, bool):
        raise TypeError("concentration must be int or float, not bool")
    if not isinstance(concentration, (int, float)):
        raise TypeError("concentration must be int or float")
    if concentration < 0 or (isinstance(concentration, float) and not math.isfinite(concentration)):
        raise ValueError("concentration must be finite and nonnegative")
    for upper_bound, category, label in _BANDS:
        if concentration <= upper_bound:
            return PM25Classification(category, label, STANDARD_ID)
    return PM25Classification(PM25ForecastCategory.HAZARDOUS, "Hazardous", STANDARD_ID)
