"""Inference from raw rows; no training or notebook state is required."""

from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import load_artifact
from .data import read_csv, require_columns
from .schema import PROJECT


def predict_frame(bundle, frame):
    meta = bundle["metadata"]
    require_columns(frame, meta["required_features"])
    if frame.empty:
        raise ValueError("Prediction input must contain at least one row.")
    result = pd.DataFrame(index=frame.index)
    if "id" in frame:
        result["id"] = frame["id"]
    if PROJECT == "airbnb":
        result["predicted_price"] = bundle["model"].predict(frame)
    else:
        probability = np.asarray(bundle["model"].predict_proba(frame))[:, 1]
        result["probability_1"] = probability
        result["prediction"] = (probability >= meta["threshold"]).astype(int)
        result["predicted_label"] = result["prediction"].map(meta["class_labels"])
        if PROJECT == "ev":
            result["triage"] = np.where(
                probability >= 0.85,
                "high_probability_buyer",
                np.where(probability <= 0.15, "low_probability_buyer", "uncertain"),
            )
    return result


def predict_csv(model_path, data_path, output_path):
    output = Path(output_path)
    if output.resolve() in {Path(data_path).resolve(), Path(model_path).resolve()}:
        raise ValueError("Prediction output must differ from the input CSV and model file.")
    result = predict_frame(load_artifact(model_path), read_csv(data_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result
