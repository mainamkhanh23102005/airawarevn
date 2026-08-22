from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from scripts.modeling.features import FORECAST_HORIZON_HOURS, TARGET_COLUMN, V1_FEATURE_COLUMNS


CALENDAR_TIMEZONE = "Asia/Ho_Chi_Minh"
INITIAL_TRAINING_MONTH = "2025-08"
VALIDATION_MONTHS = ("2026-02", "2026-03", "2026-04", "2026-05", "2026-06")
FINAL_TEST_MONTH = "2026-07"
HIGH_PM25_THRESHOLDS = (35, 75)


def _hist_gradient_boosting():
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )


DEFAULT_MODEL_FACTORIES = {
    "persistence": None,
    "linear_regression": LinearRegression,
    "hist_gradient_boosting": _hist_gradient_boosting,
}


@dataclass(frozen=True)
class HighPm25Metrics:
    threshold: float
    count: int
    mae: float | None
    underprediction_rate: float | None
    signed_bias: float | None


@dataclass(frozen=True)
class Metrics:
    mae: float
    rmse: float
    bias: float
    absolute_error_p50: float
    absolute_error_p90: float
    absolute_error_p95: float
    maximum_absolute_error: float
    high_pm25: dict[float, HighPm25Metrics]


@dataclass(frozen=True)
class WalkForwardFold:
    name: str
    training: pd.DataFrame
    validation: pd.DataFrame
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    purged_boundary_count: int


@dataclass(frozen=True)
class ReservedPeriod:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    data: pd.DataFrame


@dataclass(frozen=True)
class FoldResult:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    training_count: int
    validation_count: int
    purged_boundary_count: int
    metrics: Metrics


@dataclass(frozen=True)
class ModelResult:
    folds: tuple[FoldResult, ...]
    macro: Metrics
    pooled: Metrics
    total_oof_count: int


@dataclass(frozen=True)
class WalkForwardReport:
    models: dict[str, ModelResult]
    total_oof_count: int
    final_test: ReservedPeriod
    predictions: pd.DataFrame


def _month_bounds(month):
    period = pd.Period(month, freq="M")
    start = pd.Timestamp(period.start_time, tz=CALENDAR_TIMEZONE).tz_convert("UTC")
    end = pd.Timestamp((period + 1).start_time, tz=CALENDAR_TIMEZONE).tz_convert("UTC")
    return start, end


def _validate_dataframe(dataframe, feature_columns):
    required = ["event_time", *feature_columns, TARGET_COLUMN]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if dataframe[required].isna().any().any():
        raise ValueError("backtesting dataframe must contain complete modeling rows")
    if not dataframe["event_time"].is_monotonic_increasing:
        raise ValueError("backtesting dataframe must be chronologically ordered")


def create_walk_forward_folds(dataframe, feature_columns=None):
    feature_columns = list(feature_columns or V1_FEATURE_COLUMNS)
    _validate_dataframe(dataframe, feature_columns)
    data = dataframe.copy()
    data["event_time"] = pd.to_datetime(data["event_time"], utc=True)
    training_start, _ = _month_bounds(INITIAL_TRAINING_MONTH)
    final_start, final_end = _month_bounds(FINAL_TEST_MONTH)
    folds = []
    horizon = pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
    for month in VALIDATION_MONTHS:
        validation_start, validation_end = _month_bounds(month)
        training = data.loc[(data["event_time"] >= training_start) & ((data["event_time"] + horizon) < validation_start)].reset_index(drop=True)
        validation = data.loc[(data["event_time"] >= validation_start) & (data["event_time"] < validation_end)].reset_index(drop=True)
        purged = data.loc[(data["event_time"] >= validation_start - horizon) & (data["event_time"] < validation_start)]
        folds.append(WalkForwardFold(month, training, validation, validation_start, validation_end, len(purged)))
    final_data = data.loc[(data["event_time"] >= final_start) & (data["event_time"] < final_end)].reset_index(drop=True)
    return tuple(folds), ReservedPeriod(FINAL_TEST_MONTH, final_start, final_end, final_data)


def calculate_error_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    absolute = np.abs(error)
    high = {}
    for threshold in HIGH_PM25_THRESHOLDS:
        selected = actual > threshold
        count = int(selected.sum())
        high[threshold] = HighPm25Metrics(
            threshold,
            count,
            float(absolute[selected].mean()) if count else None,
            float((predicted[selected] < actual[selected]).mean()) if count else None,
            float(error[selected].mean()) if count else None,
        )
    return Metrics(
        float(absolute.mean()),
        float(np.sqrt(np.mean(error ** 2))),
        float(error.mean()),
        float(np.percentile(absolute, 50)),
        float(np.percentile(absolute, 90)),
        float(np.percentile(absolute, 95)),
        float(absolute.max()),
        high,
    )


def calculate_metrics(actual, predicted):
    return calculate_error_metrics(actual, predicted)


def _predict(model_name, factory, feature_columns, training, validation):
    if factory is None:
        if "pm25_lag_1h" not in validation:
            raise ValueError("persistence requires pm25_lag_1h")
        return validation["pm25_lag_1h"].to_numpy()
    model = factory()
    model.fit(training[feature_columns], training[TARGET_COLUMN])
    return model.predict(validation[feature_columns])


def _macro_metrics(results, pooled):
    return Metrics(
        *(sum(getattr(result.metrics, field) for result in results) / len(results) for field in (
            "mae", "rmse", "bias", "absolute_error_p50", "absolute_error_p90", "absolute_error_p95", "maximum_absolute_error"
        )),
        pooled.high_pm25,
    )


def evaluate_walk_forward(dataframe, feature_columns=None, model_factories=None):
    feature_columns = list(feature_columns or V1_FEATURE_COLUMNS)
    model_factories = model_factories or DEFAULT_MODEL_FACTORIES
    folds, final_test = create_walk_forward_folds(dataframe, feature_columns)
    model_results = {}
    prediction_frames = []
    for model_name, factory in model_factories.items():
        fold_results = []
        model_predictions = []
        for fold in folds:
            actual = fold.validation[TARGET_COLUMN].to_numpy()
            predicted = _predict(model_name, factory, feature_columns, fold.training, fold.validation)
            frame = pd.DataFrame({"timestamp": fold.validation["event_time"], "fold": fold.name, "model": model_name, "actual_pm25": actual, "predicted_pm25": predicted})
            model_predictions.append(frame)
            prediction_frames.append(frame)
            fold_results.append(FoldResult(fold.name, fold.training["event_time"].min(), fold.training["event_time"].max(), fold.validation_start, fold.validation_end, len(fold.training), len(fold.validation), fold.purged_boundary_count, calculate_error_metrics(actual, predicted)))
        combined = pd.concat(model_predictions, ignore_index=True)
        pooled = calculate_error_metrics(
            combined["actual_pm25"], combined["predicted_pm25"]
        )
        model_results[model_name] = ModelResult(
            tuple(fold_results),
            _macro_metrics(fold_results, pooled),
            pooled,
            len(combined),
        )
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame(columns=["timestamp", "fold", "model", "actual_pm25", "predicted_pm25"])
    return WalkForwardReport(model_results, sum(len(fold.validation) for fold in folds), final_test, predictions)
