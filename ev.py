"""Paste this entire file into one notebook cell, or run it as a Python script.

Upload train.csv and test.csv and set the paths below. This fits only the
validation-selected weighted ensemble from ev.ipynb (cells 48-49), then predicts
the external test file. It does not repeat model, weight, or threshold searches.
Later runs reuse the saved model and need only test.csv.
"""

# ---------------------------- SETTINGS ----------------------------
TRAIN_PATH = "/content/train.csv"
TEST_PATH = "/content/test.csv"
OUTPUT_DIR = "/content/ev_results"
REUSE_SAVED_MODEL = True
DOWNLOAD_SUBMISSION = True

import csv
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

DEPENDENCIES = [
    "numpy>=1.24", "pandas>=2.0", "scikit-learn>=1.6",
    "lightgbm>=4.6.0", "xgboost>=3.1.3", "catboost>=1.2.8",
    "cloudpickle>=3.1",
]
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", *DEPENDENCIES]
)

import cloudpickle
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

TARGET = "Will_Buy_EV"
SEED = 42
THRESHOLD = 0.48
WEIGHTS = {"lightgbm": 0.70, "xgboost": 0.15, "catboost": 0.15}
NUM_COLS = [
    "Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]
CAT_COLS = [
    "Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
    "Subsidy_Available", "Range_Anxiety_Level",
]
RAW_FEATURES = NUM_COLS + CAT_COLS

# Exact component configurations used by the recorded winning ensemble.
# Its recorded holdout accuracy was 0.8976 and buyer F1 was 0.7016.
# Actual results are evaluated below, not assumed from those historical scores.
LGB_PARAMS = dict(
    n_estimators=700, learning_rate=0.04, num_leaves=31, max_depth=-1,
    min_child_samples=30, subsample=0.90, colsample_bytree=0.90,
    reg_alpha=0.05, reg_lambda=2.0, objective="binary",
    random_state=SEED, n_jobs=-1, verbosity=-1,
)
XGB_PARAMS = dict(
    n_estimators=700, max_depth=5, learning_rate=0.05,
    min_child_weight=3, gamma=0.05, subsample=0.90, colsample_bytree=0.90,
    reg_alpha=0.05, reg_lambda=2.0, objective="binary:logistic",
    eval_metric="logloss", tree_method="hist", random_state=SEED, n_jobs=-1,
)
CAT_PARAMS = dict(
    iterations=800, depth=7, learning_rate=0.05, loss_function="Logloss",
    eval_metric="AUC", l2_leaf_reg=5, random_seed=SEED,
    verbose=100, allow_writing_files=False,
)


def add_features(data):
    """The 11 engineered features used by the selected notebook ensemble."""
    data = data.copy()
    data["Total_Charging_Stations"] = (
        data["Charging_Stations_Near_Home"] + data["Charging_Stations_Near_Work"]
    )
    data["Stations_per_Commute"] = (
        data["Total_Charging_Stations"] / (data["Daily_Commute_km"] + 1)
    )
    data["Commute_per_Station"] = (
        data["Daily_Commute_km"] / (data["Total_Charging_Stations"] + 1)
    )
    data["Income_per_Car"] = (
        data["Annual_Income_USD"] / (data["Number_of_Cars_Owned"] + 1)
    )
    data["HomeCharge_Binary"] = (
        data["Home_Charging_Possible"].map({"Yes": 1, "No": 0}).fillna(0)
    )
    data["Subsidy_Binary"] = (
        data["Subsidy_Available"].map({"Yes": 1, "No": 0}).fillna(0)
    )
    data["HomeCharge_And_Subsidy"] = data["HomeCharge_Binary"] * data["Subsidy_Binary"]
    data["Concern_x_Subsidy"] = data["Environmental_Concern_Level"] * data["Subsidy_Binary"]
    data["Concern_x_HomeCharge"] = data["Environmental_Concern_Level"] * data["HomeCharge_Binary"]
    data["Commute_x_HomeCharge"] = data["Daily_Commute_km"] * data["HomeCharge_Binary"]
    data["Range_Anxiety_Ordinal"] = (
        data["Range_Anxiety_Level"].map({"Low": 0, "Medium": 1, "High": 2}).fillna(-1)
    )
    return data


class EVPreprocessor:
    """Keep training medians and encoded-column order for all future inputs."""

    def _engineer(self, raw):
        missing = sorted(set(RAW_FEATURES) - set(raw.columns))
        if missing:
            raise ValueError(f"Missing raw features: {missing}")
        data = raw[RAW_FEATURES].copy()
        for col in NUM_COLS:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data[NUM_COLS] = data[NUM_COLS].replace([np.inf, -np.inf], np.nan)
        for col in CAT_COLS:
            data[col] = data[col].astype("string").str.strip().replace("", pd.NA)
        # Match the notebook: engineer first, then impute each numeric feature.
        data = add_features(data)
        numeric = [col for col in data if col not in CAT_COLS]
        data[numeric] = data[numeric].replace([np.inf, -np.inf], np.nan)
        return data

    def fit(self, raw):
        data = self._engineer(raw)
        self.numeric_columns_ = [col for col in data if col not in CAT_COLS]
        self.medians_ = data[self.numeric_columns_].median().fillna(0.0)
        filled = self.transform(raw)
        self.encoded_columns_ = pd.get_dummies(filled, columns=CAT_COLS).columns.tolist()
        return self

    def transform(self, raw):
        data = self._engineer(raw)
        data[self.numeric_columns_] = data[self.numeric_columns_].fillna(self.medians_)
        for col in CAT_COLS:
            data[col] = data[col].fillna("Missing").astype(str)
        return data

    def encode(self, transformed):
        return pd.get_dummies(transformed, columns=CAT_COLS).reindex(
            columns=self.encoded_columns_, fill_value=0,
        )


def read_csv(path):
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Upload {path.name} to {path.parent}, or update its path.")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    names = [name.strip() for name in header]
    if not names:
        raise ValueError(f"{path.name} is empty.")
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate column names in {path.name}.")
    id_types = {raw: "string" for raw, name in zip(header, names) if name == "id"}
    frame = pd.read_csv(path, dtype=id_types, encoding="utf-8-sig")
    frame.columns = names
    if frame.empty:
        raise ValueError(f"{path.name} contains no data rows.")
    missing = sorted(set(RAW_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing raw features: {missing}")
    return frame


def labels_from(frame):
    if TARGET not in frame:
        raise ValueError(f"Training CSV must contain {TARGET}.")
    labels = frame[TARGET].astype("string").str.strip().str.lower().map({
        "no": 0, "yes": 1, "0": 0, "1": 1, "0.0": 0, "1.0": 1,
    })
    if labels.isna().any():
        raise ValueError(f"{TARGET} must contain non-missing Yes/No or 0/1 labels.")
    return labels.astype(np.int8)


def predict_probabilities(bundle, raw):
    probabilities = np.empty(len(raw), dtype=float)
    processor = bundle["preprocessor"]
    for start in range(0, len(raw), 50_000):
        end = min(start + 50_000, len(raw))
        transformed = processor.transform(raw.iloc[start:end])
        encoded = processor.encode(transformed)
        combined = np.zeros(len(transformed), dtype=float)
        for name, weight in bundle["weights"].items():
            model = bundle["models"][name]
            positive_index = int(np.flatnonzero(np.asarray(model.classes_) == 1)[0])
            inputs = transformed if name == "catboost" else encoded
            combined += weight * model.predict_proba(inputs)[:, positive_index]
        probabilities[start:end] = combined
    return probabilities


def evaluate(labels, probabilities, threshold, title):
    predictions = (probabilities >= threshold).astype(np.int8)
    both_classes = labels.nunique() == 2
    metrics = {
        "rows": len(labels), "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "buyer_precision": float(precision_score(labels, predictions, zero_division=0)),
        "buyer_recall": float(recall_score(labels, predictions, zero_division=0)),
        "buyer_f1": float(f1_score(labels, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if both_classes else None,
        "average_precision": (
            float(average_precision_score(labels, probabilities)) if both_classes else None
        ),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }
    print(f"\n{title}")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(classification_report(
        labels, predictions, labels=[0, 1], digits=5, zero_division=0,
        target_names=["Not Buyer (0)", "Buyer (1)"],
    ))
    return metrics


def fit_selected_model(train):
    y = labels_from(train)
    if y.nunique() != 2 or y.value_counts().min() < 30:
        raise ValueError("Training data needs at least 30 rows of each class.")
    # Same 70/15/15 split and random states as the selected notebook workflow.
    X_fit, X_temp, y_fit, y_temp = train_test_split(
        train[RAW_FEATURES], y, test_size=0.30, stratify=y, random_state=SEED,
    )
    X_val, X_holdout, y_val, y_holdout = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED,
    )
    processor = EVPreprocessor().fit(X_fit)
    fit_raw = processor.transform(X_fit)
    fit_encoded = processor.encode(fit_raw)
    val_raw = processor.transform(X_val)

    print(f"Training the selected ensemble on {len(X_fit):,} rows.")
    print(f"Validation: {len(X_val):,} | Internal holdout: {len(X_holdout):,}")
    models = {
        "lightgbm": LGBMClassifier(**LGB_PARAMS),
        "xgboost": XGBClassifier(**XGB_PARAMS),
        "catboost": CatBoostClassifier(**CAT_PARAMS),
    }
    print("Fitting LightGBM (ensemble weight 70%)...", flush=True)
    models["lightgbm"].fit(fit_encoded, y_fit)
    print("Fitting XGBoost (ensemble weight 15%)...", flush=True)
    models["xgboost"].fit(fit_encoded, y_fit)
    print("Fitting CatBoost (ensemble weight 15%)...", flush=True)
    # Preserve the original validation-based early stopping for this component.
    models["catboost"].fit(
        fit_raw, y_fit, cat_features=CAT_COLS, eval_set=(val_raw, y_val),
        early_stopping_rounds=100, use_best_model=True, verbose=100,
    )
    bundle = {
        "winner": "Weighted Soft Voting", "models": models,
        "preprocessor": processor, "weights": WEIGHTS.copy(),
        "threshold": THRESHOLD, "target": TARGET,
        "raw_features": RAW_FEATURES, "class_labels": {0: "No", 1: "Yes"},
        "seed": SEED, "catboost_tree_count": int(models["catboost"].tree_count_),
        "split_sizes": {
            "train": len(X_fit), "validation": len(X_val), "holdout": len(X_holdout),
        },
        "versions": {
            spec.split(">=")[0]: metadata.version(spec.split(">=")[0])
            for spec in DEPENDENCIES
        },
    }
    return bundle, X_holdout, y_holdout


def main():
    # Validate the external file before fitting any model.
    test = read_csv(TEST_PATH)
    has_test_labels = TARGET in test and test[TARGET].notna().any()
    test_labels = labels_from(test) if has_test_labels else None
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "model_bundle.pkl"

    if REUSE_SAVED_MODEL and model_path.is_file():
        print(f"Loading saved EV model: {model_path}")
        with model_path.open("rb") as handle:
            bundle = cloudpickle.load(handle)
        if (
            bundle.get("target") != TARGET
            or bundle.get("raw_features") != RAW_FEATURES
            or bundle.get("winner") != "Weighted Soft Voting"
        ):
            raise ValueError("Saved model differs; set REUSE_SAVED_MODEL = False.")
        print("Saved model loaded; no training is needed.")
    else:
        bundle, X_holdout, y_holdout = fit_selected_model(read_csv(TRAIN_PATH))
        with model_path.open("wb") as handle:
            cloudpickle.dump(bundle, handle)
        run_metadata = {
            key: value for key, value in bundle.items()
            if key not in {"models", "preprocessor"}
        }
        (out / "run_metadata.json").write_text(
            json.dumps(run_metadata, indent=2), encoding="utf-8",
        )
        (out / "requirements.txt").write_text(
            "\n".join(f"{name}=={version}" for name, version in bundle["versions"].items()) + "\n",
            encoding="utf-8",
        )
        holdout_metrics = evaluate(
            y_holdout, predict_probabilities(bundle, X_holdout),
            bundle["threshold"], "Internal labeled holdout evaluation",
        )
        (out / "holdout_metrics.json").write_text(
            json.dumps(holdout_metrics, indent=2), encoding="utf-8",
        )

    probabilities = predict_probabilities(bundle, test)
    predicted_classes = (probabilities >= bundle["threshold"]).astype(np.int8)
    submission = test[["id"]].copy() if "id" in test else pd.DataFrame(index=test.index)
    submission[TARGET] = np.where(predicted_classes == 1, "Yes", "No")
    details = submission.copy()
    details["probability_class_1"] = probabilities
    submission_path = out / "submission.csv"
    submission.to_csv(submission_path, index=False)
    details.to_csv(out / "test_predictions.csv", index=False)

    if test_labels is not None:
        test_metrics = evaluate(
            test_labels, probabilities, bundle["threshold"], "External labeled test evaluation",
        )
        (out / "test_metrics.json").write_text(
            json.dumps(test_metrics, indent=2), encoding="utf-8",
        )
    else:
        print(f"\nTest CSV has no true {TARGET} labels; predictions saved, metrics unavailable.")

    print(f"\nModel: {bundle['winner']} | Threshold: {bundle['threshold']:.2f}")
    print("Weights:", bundle["weights"])
    print(f"Predicted {len(test):,} rows; input row order and IDs preserved.")
    print(details.head(10).to_string(index=False))
    print(f"\nModel: {model_path}")
    print(f"Submission: {submission_path}")
    print(f"Probabilities: {out / 'test_predictions.csv'}")
    if DOWNLOAD_SUBMISSION:
        try:
            from google.colab import files
        except ImportError:
            pass
        else:
            files.download(str(submission_path))


if __name__ == "__main__":
    main()
