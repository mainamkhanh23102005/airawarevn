import math
from dataclasses import dataclass
from enum import Enum


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
    for value in (current_pm25, forecast_pm25):
        if type(value) not in (int, float):
            raise TypeError("PM2.5 values must be int or float")
        if value < 0 or (isinstance(value, float) and not math.isfinite(value)):
            raise ValueError("PM2.5 values must be finite and nonnegative")
    try:
        change = forecast_pm25 - current_pm25
    except OverflowError as error:
        raise ValueError("change must be finite") from error
    if isinstance(change, float) and not math.isfinite(change):
        raise ValueError("change must be finite")
    if change == 0:
        change = 0.0
    if -STABLE_CHANGE_THRESHOLD_UG_M3 < change < STABLE_CHANGE_THRESHOLD_UG_M3:
        trend = ForecastTrend.STABLE
    elif change <= -STABLE_CHANGE_THRESHOLD_UG_M3:
        trend = ForecastTrend.IMPROVING
    else:
        trend = ForecastTrend.WORSENING
    return PM25Trend(trend, change, POLICY_ID)
