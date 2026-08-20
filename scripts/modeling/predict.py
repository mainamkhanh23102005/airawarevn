from pathlib import Path

import joblib

from scripts.modeling.features import TARGET_COLUMN, V1_FEATURE_COLUMNS
from scripts.modeling.train import MODEL_TYPE


def load_artifact(path):
    artifact = joblib.load(Path(path))
    if set(artifact) != {"model", "metadata"}:
        raise ValueError("invalid model artifact")
    metadata = artifact["metadata"]
    if metadata.get("model_type") != MODEL_TYPE:
        raise ValueError("unexpected model type")
    if metadata.get("feature_columns") != V1_FEATURE_COLUMNS:
        raise ValueError("artifact feature list does not match V1 contract")
    if metadata.get("target_column") != TARGET_COLUMN:
        raise ValueError("artifact target does not match V1 contract")
    return artifact["model"], metadata


def predict_pm25_t_plus_6(model, metadata, dataframe):
    feature_columns = metadata.get("feature_columns")
    if feature_columns != V1_FEATURE_COLUMNS:
        raise ValueError("model feature list does not match V1 contract")
    missing_columns = [
        column for column in feature_columns if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(f"missing required feature columns: {missing_columns}")
    features = dataframe[feature_columns]
    if features.isna().any().any():
        raise ValueError("missing required feature values")
    return model.predict(features)
