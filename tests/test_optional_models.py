"""Exercise notebook model families when the boosting extra is installed."""

import importlib.util

import joblib
import numpy as np
import pytest
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from ev.config import TrainConfig
from ev.data import load_training_data
from ev.demo import make_demo_data
from ev.models import candidates

BOOSTING = all(
    importlib.util.find_spec(package) is not None for package in ["lightgbm", "xgboost", "catboost"]
)


@pytest.mark.skipif(
    not BOOSTING, reason="Install the boosting extra to test notebook model families"
)
def test_optional_models_export_and_reload(tmp_path):
    source = tmp_path / "train.csv"
    make_demo_data(160).to_csv(source, index=False)
    data = load_training_data(source)
    config = TrainConfig(
        models=("lightgbm", "xgboost", "catboost", "voting", "stacking"),
        feature_sets=("engineered", "original"),
        n_estimators=5,
        n_jobs=1,
        cv_folds=2,
    )
    with threadpool_limits(limits=1):
        for candidate, model in candidates(config).items():
            fitted = clone(model).fit(data.X.iloc[:120], data.y.iloc[:120])
            expected = fitted.predict(data.X.iloc[120:])
            assert np.isfinite(expected).all(), candidate
            path = tmp_path / "candidate.joblib"
            joblib.dump(fitted, path)
            actual = joblib.load(path).predict(data.X.iloc[120:])
            np.testing.assert_allclose(actual, expected)


@pytest.mark.skipif(importlib.util.find_spec("imblearn") is None, reason="Install the smote extra")
def test_smote_pipeline_operates_on_raw_rows(tmp_path):
    source = tmp_path / "train.csv"
    make_demo_data(180).to_csv(source, index=False)
    data = load_training_data(source)
    config = TrainConfig(models=("logistic_regression",), smote=True, n_jobs=1)
    model = candidates(config)["logistic_regression__engineered"]
    with threadpool_limits(limits=1):
        model.fit(data.X.iloc[:130], data.y.iloc[:130])
    assert model.predict_proba(data.X.iloc[130:]).shape == (50, 2)
