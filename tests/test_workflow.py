"""Behavioral tests for raw-data inference, isolation and portable exports."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from ev.artifacts import load_artifact
from ev.config import TrainConfig
from ev.data import load_training_data, read_csv
from ev.demo import make_demo_data
from ev.models import candidates
from ev.pipeline import run as train
from ev.predict import predict_frame
from ev.schema import CAT_COLS, NUM_COLS, RAW_FEATURES, TARGET
from ev.validation import split_data


@pytest.fixture
def csv_data(tmp_path):
    frame = make_demo_data(240)
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)
    return frame, path


def test_export_predict_in_fresh_process(csv_data, tmp_path):
    frame, path = csv_data
    config = TrainConfig(
        models=("logistic_regression",),
        feature_sets=("engineered",),
        cv_folds=2,
        n_estimators=8,
        n_jobs=1,
        search_iterations=1,
    )
    out = tmp_path / "run"
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    completed = subprocess.run(
        [
            sys.executable,
            str(scripts / "run_train.py"),
            "--data",
            str(path),
            "--output",
            str(out),
            "--models",
            *config.models,
            "--feature-sets",
            *config.feature_sets,
            "--cv-folds",
            "2",
            "--n-estimators",
            "8",
            "--n-jobs",
            "1",
            "--search-iterations",
            "1",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = {"model_path": out / "model.joblib"}
    bundle = load_artifact(result["model_path"])
    incoming = frame.drop(columns=[TARGET]).head(6).copy()
    incoming[CAT_COLS[0]] = "Never seen during training"
    incoming[NUM_COLS[0]] = np.nan
    incoming = incoming[incoming.columns[::-1]]
    expected = predict_frame(bundle, incoming)
    raw_path = tmp_path / "incoming.csv"
    incoming.to_csv(raw_path, index=False)
    prediction_path = tmp_path / "predictions.csv"
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts/run_predict.py"),
            "--model",
            str(result["model_path"]),
            "--data",
            str(raw_path),
            "--output",
            str(prediction_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    actual = pd.read_csv(prediction_path, dtype={"id": "string"})
    column = "probability_1"
    np.testing.assert_allclose(actual[column], expected[column], rtol=1e-10)
    assert actual["id"].tolist() == incoming["id"].tolist()
    if "prediction" in expected:
        assert (
            expected["prediction"]
            .eq(expected["probability_1"] >= bundle["metadata"]["threshold"])
            .all()
        )
    assert {
        "model.joblib",
        "metadata.json",
        "requirements.lock.txt",
        "metrics.json",
        "config.json",
        "holdout_predictions.csv",
        "cv_results.csv",
        "split_assignments.csv",
    } <= {p.name for p in out.iterdir()}
    report = subprocess.run(
        [sys.executable, str(scripts / "build_report.py"), "--run", str(out)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert report.returncode == 0, report.stderr
    assert "Development CV comparison" in (out / "REPORT.md").read_text()
    assert "<table" in (out / "REPORT.html").read_text()
    assert json.loads((out / "metadata.json").read_text())["project"] == "ev"
    with pytest.raises(ValueError, match="Missing required"):
        predict_frame(bundle, incoming.drop(columns=[RAW_FEATURES[0]]))
    with pytest.raises(FileExistsError):
        train(path, out, config)


def test_partitions_do_not_overlap(csv_data):
    _, path = csv_data
    data = load_training_data(path)
    config = TrainConfig(cv_folds=2)
    tr, va, te = split_data(data, config)
    assert set(tr).isdisjoint(va) and set(tr).isdisjoint(te) and set(va).isdisjoint(te)
    assert sorted([*tr, *va, *te]) == list(range(len(data.X)))
    if data.groups is not None:
        assert set(data.groups.iloc[tr]).isdisjoint(data.groups.iloc[va])
        assert set(data.groups.iloc[tr]).isdisjoint(data.groups.iloc[te])
        assert set(data.groups.iloc[va]).isdisjoint(data.groups.iloc[te])


def test_candidates_handle_missing_and_unseen_categories(csv_data):
    frame, path = csv_data
    data = load_training_data(path)
    config = TrainConfig(n_estimators=5, n_jobs=1)
    # Exercise every default candidate without spending time on repeated CV fits.
    for name, estimator in candidates(config).items():
        fitted = clone(estimator).fit(data.X.iloc[:180], data.y.iloc[:180])
        incoming = data.X.iloc[180:185].copy()
        incoming[CAT_COLS[0]] = "new category"
        incoming[NUM_COLS[0]] = np.nan
        prediction = fitted.predict(incoming)
        assert np.isfinite(prediction).all(), name


def test_duplicate_headers_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("age, age\n20,30\n")
    with pytest.raises(ValueError, match="unique"):
        read_csv(path)


def test_invalid_target_and_repeated_ids(csv_data):
    frame, path = csv_data
    frame.loc[1, "id"] = frame.loc[0, "id"]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Repeated IDs"):
        load_training_data(path)
    frame["id"] = [str(i) for i in range(len(frame))]
    frame[TARGET] = "invalid"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_training_data(path)


def test_unknown_configuration_key(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"cv_fold": 3}')
    with pytest.raises(ValueError, match="Unknown configuration"):
        TrainConfig.from_json(path)
