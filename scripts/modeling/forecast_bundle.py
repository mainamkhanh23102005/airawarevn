from __future__ import annotations

import hashlib
import io
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression

from scripts.modeling.features import V1_FEATURE_COLUMNS

HORIZONS = (1, 2, 3, 4, 5, 6)
BUNDLE_FORMAT = "airaware.multi_horizon_bundle"
BUNDLE_VERSION = 1
MODEL_TYPE = "sklearn.linear_model.LinearRegression"
CALENDAR_TIMEZONE = "Asia/Ho_Chi_Minh"


class ForecastBundle:
    def __init__(self, models, metadata, sha256):
        self.models = models
        self.metadata = metadata
        self.sha256 = sha256


def target_column_for_horizon(horizon):
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    return f"target_pm25_t_plus_{horizon}"


def feature_schema_sha256(horizon):
    schema = {
        "calendar_timezone": CALENDAR_TIMEZONE,
        "feature_columns": V1_FEATURE_COLUMNS,
        "feature_configuration": "A2",
        "forecast_horizon_hours": horizon,
        "raw_pm25_is_feature": False,
        "target_column": target_column_for_horizon(horizon),
        "weather_is_feature": False,
    }
    encoded = __import__("json").dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_estimator(model):
    if type(model) is not LinearRegression:
        raise ValueError("bundle model must be LinearRegression")
    if model.get_params(deep=False) != LinearRegression().get_params(deep=False):
        raise ValueError("bundle model parameters do not match default LinearRegression")
    if getattr(model, "n_features_in_", None) != len(V1_FEATURE_COLUMNS):
        raise ValueError("bundle model feature count does not match A2")
    if list(getattr(model, "feature_names_in_", [])) != V1_FEATURE_COLUMNS:
        raise ValueError("bundle model feature order does not match A2")
    coefficient = np.asarray(getattr(model, "coef_", []))
    intercept = np.asarray(getattr(model, "intercept_", []))
    if coefficient.shape != (len(V1_FEATURE_COLUMNS),) or intercept.shape != ():
        raise ValueError("bundle model must be fitted single-output LinearRegression")
    if not np.isfinite(coefficient).all() or not np.isfinite(intercept).all():
        raise ValueError("bundle model parameters must be finite")


def _validate_metadata(metadata, models):
    if not isinstance(metadata, dict):
        raise ValueError("bundle metadata must be an object")
    required = {
        "bundle_format": BUNDLE_FORMAT,
        "bundle_version": BUNDLE_VERSION,
        "model_type": MODEL_TYPE,
        "horizons": list(HORIZONS),
        "feature_columns": V1_FEATURE_COLUMNS,
        "feature_configuration": "A2",
        "calendar_timezone": CALENDAR_TIMEZONE,
        "raw_pm25_is_feature": False,
        "weather_is_feature": False,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(f"bundle metadata {key!r} does not match contract")
    if not isinstance(metadata.get("model_version"), str) or not metadata["model_version"]:
        raise ValueError("bundle metadata model_version is required")
    input_digest = metadata.get("training_input_sha256")
    if not isinstance(input_digest, str) or len(input_digest) != 64 or any(character not in "0123456789abcdef" for character in input_digest):
        raise ValueError("bundle metadata training_input_sha256 is invalid")
    try:
        cutoff = pd.Timestamp(metadata.get("training_cutoff"))
    except (TypeError, ValueError) as error:
        raise ValueError("bundle metadata training_cutoff is invalid") from error
    if cutoff.tzinfo is None or cutoff.utcoffset() != pd.Timedelta(0) or cutoff.minute or cutoff.second or cutoff.microsecond or cutoff.nanosecond:
        raise ValueError("bundle metadata training_cutoff is invalid")
    versions = metadata.get("dependency_versions")
    if not isinstance(versions, dict) or any(not isinstance(versions.get(key), str) or not versions[key] for key in ("python", "sklearn", "numpy", "joblib")):
        raise ValueError("bundle metadata dependency_versions is invalid")
    if not isinstance(metadata.get("horizon_metadata"), dict):
        raise ValueError("bundle horizon_metadata is required")
    if set(models) != set(HORIZONS) or set(metadata["horizon_metadata"]) != {str(value) for value in HORIZONS}:
        raise ValueError("bundle must contain exactly horizons 1 through 6")
    for horizon in HORIZONS:
        entry = metadata["horizon_metadata"][str(horizon)]
        if not isinstance(entry, dict) or entry.get("target_column") != target_column_for_horizon(horizon):
            raise ValueError("bundle horizon target metadata does not match contract")
        if not isinstance(entry.get("training_row_count"), int) or entry["training_row_count"] <= 0:
            raise ValueError("bundle horizon training row count is invalid")
        cohort_digest = entry.get("cohort_sha256")
        if not isinstance(cohort_digest, str) or len(cohort_digest) != 64 or any(character not in "0123456789abcdef" for character in cohort_digest):
            raise ValueError("bundle horizon cohort digest is invalid")
        timestamps = {}
        for key in ("training_origin_start", "training_origin_end", "target_interval_start", "target_interval_end"):
            try:
                timestamp = pd.Timestamp(entry.get(key))
            except (TypeError, ValueError) as error:
                raise ValueError(f"bundle horizon {key} is invalid") from error
            if timestamp.tzinfo is None or timestamp.utcoffset() != pd.Timedelta(0):
                raise ValueError(f"bundle horizon {key} is invalid")
            timestamps[key] = timestamp.tz_convert("UTC")
        if timestamps["training_origin_start"] > timestamps["training_origin_end"] or timestamps["target_interval_start"] > timestamps["target_interval_end"] or timestamps["target_interval_end"] > cutoff:
            raise ValueError("bundle horizon time bounds are invalid")
        if entry.get("feature_schema_sha256") != feature_schema_sha256(horizon):
            raise ValueError("bundle horizon feature schema does not match contract")
        _validate_estimator(models[horizon])


def build_bundle(models, *, model_version, training_input_sha256, training_cutoff, horizon_metadata):
    models = {int(key): value for key, value in models.items()}
    metadata = {
        "bundle_format": BUNDLE_FORMAT,
        "bundle_version": BUNDLE_VERSION,
        "model_type": MODEL_TYPE,
        "model_version": model_version,
        "horizons": list(HORIZONS),
        "feature_columns": V1_FEATURE_COLUMNS.copy(),
        "feature_configuration": "A2",
        "calendar_timezone": CALENDAR_TIMEZONE,
        "raw_pm25_is_feature": False,
        "weather_is_feature": False,
        "training_input_sha256": training_input_sha256,
        "training_cutoff": str(training_cutoff),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dependency_versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
        "horizon_metadata": {},
    }
    for horizon in HORIZONS:
        entry = dict(horizon_metadata.get(horizon, horizon_metadata.get(str(horizon), {})))
        entry["target_column"] = target_column_for_horizon(horizon)
        entry["feature_schema_sha256"] = feature_schema_sha256(horizon)
        metadata["horizon_metadata"][str(horizon)] = entry
    _validate_metadata(metadata, models)
    return {"models": models, "metadata": metadata}


def save_bundle(path, bundle):
    joblib.dump(bundle, Path(path))


def load_bundle_bytes(raw):
    digest = hashlib.sha256(raw).hexdigest()
    loaded = joblib.load(io.BytesIO(raw))
    if not isinstance(loaded, dict) or set(loaded) != {"models", "metadata"}:
        raise ValueError("invalid multi-horizon bundle structure")
    models = {int(key): value for key, value in loaded["models"].items()}
    metadata = loaded["metadata"]
    _validate_metadata(metadata, models)
    versions = metadata["dependency_versions"]
    if versions.get("sklearn") != sklearn.__version__:
        raise ValueError("bundle sklearn version is incompatible with runtime")
    return ForecastBundle(models, metadata, digest)


def load_bundle(path):
    return load_bundle_bytes(Path(path).read_bytes())


def predict_trajectory(bundle, features):
    if len(features) != 1 or any(column not in features.columns for column in V1_FEATURE_COLUMNS):
        raise ValueError("trajectory features must contain one A2 row")
    ordered_features = features[V1_FEATURE_COLUMNS]
    values = ordered_features.iloc[0].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("trajectory features must be finite")
    predictions = []
    for horizon in HORIZONS:
        result = np.asarray(bundle.models[horizon].predict(ordered_features))
        if result.shape != (1,) or not math.isfinite(float(result[0])):
            raise ValueError("trajectory model output is invalid")
        predictions.append({"forecast_horizon_hours": horizon, "predicted_pm25": float(result[0])})
    return predictions
