"""Validated, serializable training settings."""

import json
import math
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("EV_DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("EV_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
TRAIN_PATH = DATA_DIR / "train.csv"
PREDICT_PATH = DATA_DIR / "new_rows.csv"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "run"
MODEL_PATH = DEFAULT_OUTPUT_DIR / "model.joblib"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.csv"

MODEL_NAMES = [
    "dummy",
    "logistic_regression",
    "random_forest",
    "gaussian_nb",
    "lightgbm",
    "xgboost",
    "catboost",
    "voting",
    "stacking",
]
FEATURE_SETS = ["original", "engineered"]


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    models: tuple[str, ...] = ("logistic_regression", "random_forest")
    feature_sets: tuple[str, ...] = ("engineered",)
    cv_folds: int = 3
    n_estimators: int = 200
    n_jobs: int = 2
    search_iterations: int = 0
    holdout_size: float = 0.15
    validation_size: float = 0.15
    primary_metric: str = "f1"
    min_recall: float = 0.65
    smote: bool = False

    def __post_init__(self):
        if (
            not self.models
            or len(set(self.models)) != len(self.models)
            or set(self.models) - set(MODEL_NAMES)
        ):
            raise ValueError(f"models must be a unique nonempty list from {MODEL_NAMES}")
        if (
            not self.feature_sets
            or len(set(self.feature_sets)) != len(self.feature_sets)
            or set(self.feature_sets) - set(FEATURE_SETS)
        ):
            raise ValueError(f"feature_sets must be a unique nonempty list from {FEATURE_SETS}")
        for name in [
            "seed",
            "cv_folds",
            "n_estimators",
            "n_jobs",
            "search_iterations",
        ]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        for name in ["holdout_size", "validation_size", "min_recall"]:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if self.seed < 0 or self.cv_folds < 2 or self.n_estimators < 1 or self.n_jobs < 1:
            raise ValueError(
                "Use a nonnegative seed, cv_folds >= 2, n_estimators >= 1 and n_jobs >= 1"
            )
        if self.search_iterations < 0:
            raise ValueError("search_iterations must be nonnegative")
        if not 0 < self.holdout_size < 1 or not 0 < self.validation_size < 1 - self.holdout_size:
            raise ValueError(
                "holdout_size and validation_size must be positive and sum to less than 1"
            )
        if not 0 <= self.min_recall <= 1:
            raise ValueError("min_recall must be between 0 and 1")
        if self.primary_metric not in ["f1", "f1_macro", "balanced_accuracy"]:
            raise ValueError("Unsupported primary_metric")
        for name in ["smote"]:
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be true or false")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_json(cls, path=None, **overrides):
        values = json.loads(Path(path).read_text()) if path else {}
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a JSON object")
        values.update({k: v for k, v in overrides.items() if v is not None})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        for name in ["models", "feature_sets"]:
            if name in values:
                if not isinstance(values[name], (list, tuple)):
                    raise ValueError(f"{name} must be a list")
                values[name] = tuple(values[name])
        return cls(**values)


FULL_PROFILE = {
    "models": [
        "logistic_regression",
        "random_forest",
        "lightgbm",
        "xgboost",
        "catboost",
        "voting",
        "stacking",
    ],
    "feature_sets": ["engineered", "original"],
    "n_estimators": 400,
    "cv_folds": 3,
    "search_iterations": 8,
    "n_jobs": 2,
}


def make_config(profile="default", **overrides):
    """Choose the lightweight or notebook-model profile, then apply explicit overrides."""
    if profile not in {"default", "full"}:
        raise ValueError("profile must be default or full")
    values = dict(FULL_PROFILE) if profile == "full" else {}
    values.update({key: value for key, value in overrides.items() if value is not None})
    for key in ["models", "feature_sets"]:
        if key in values:
            values[key] = tuple(values[key])
    return TrainConfig(**values)
