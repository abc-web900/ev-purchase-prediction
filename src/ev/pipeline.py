"""Select on training CV, tune a threshold on validation, evaluate holdout once."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from threadpoolctl import threadpool_limits

from .artifacts import file_hash, load_artifact, save_artifact, write_json
from .config import DEFAULT_OUTPUT_DIR, TRAIN_PATH, TrainConfig
from .data import load_training_data, raw_schema
from .models import candidates, search_space
from .predict import predict_frame
from .schema import PROJECT, RAW_FEATURES, TARGET
from .validation import metric_summary, select_threshold, split_data


def run(data_path=None, output_dir=None, config=None):
    data_path = TRAIN_PATH if data_path is None else data_path
    output_dir = DEFAULT_OUTPUT_DIR if output_dir is None else output_dir
    config = config or TrainConfig()
    with threadpool_limits(limits=config.n_jobs):
        return _run(data_path, output_dir, config)


def _run(data_path, output_dir, config):
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            "Use a new or empty output directory to preserve previous run artifacts."
        )
    data = load_training_data(data_path)
    tr, va, te = split_data(data, config)
    X, y = data.X.iloc[tr], data.y.iloc[tr]
    folds = list(
        StratifiedKFold(config.cv_folds, shuffle=True, random_state=config.seed + 3).split(X, y)
    )
    scoring = {
        "f1": make_scorer(f1_score, zero_division=0),
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
        "balanced_accuracy": "balanced_accuracy",
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
    }
    models = candidates(config)
    out.mkdir(parents=True, exist_ok=True)
    assignments = pd.concat(
        [
            pd.DataFrame(
                {
                    "csv_row_index": data.X.iloc[ix].index,
                    "id": data.ids.iloc[ix].to_numpy(),
                    "partition": label,
                }
            )
            for ix, label in [(tr, "train"), (va, "validation"), (te, "holdout")]
        ],
        ignore_index=True,
    )
    assignments.to_csv(out / "split_assignments.csv", index=False)
    records, fold_records = [], []
    for name, model in list(models.items()):
        print(f"Cross-validating {name}", flush=True)
        start = time.perf_counter()
        params = search_space(name)
        if config.search_iterations and params:
            search = RandomizedSearchCV(
                model,
                params,
                n_iter=min(
                    config.search_iterations, int(np.prod([len(v) for v in params.values()]))
                ),
                scoring=scoring,
                refit=config.primary_metric,
                cv=folds,
                random_state=config.seed,
                n_jobs=1,
                error_score="raise",
                return_train_score=True,
            )
            search.fit(X, y)
            models[name] = clone(search.best_estimator_)
            results = search.cv_results_
            best = search.best_index_
            scores = {
                f"{kind}_{metric}": np.array(
                    [results[f"split{i}_{kind}_{metric}"][best] for i in range(config.cv_folds)]
                )
                for kind in ["train", "test"]
                for metric in scoring
            }
            pd.DataFrame(results).to_csv(out / f"search_{name}.csv", index=False)
        else:
            scores = cross_validate(
                model,
                X,
                y,
                cv=folds,
                scoring=scoring,
                n_jobs=1,
                error_score="raise",
                return_train_score=True,
            )
        records.append(
            {
                "candidate": name,
                "seconds": time.perf_counter() - start,
                **{m: float(np.mean(scores[f"test_{m}"])) for m in scoring},
                "primary_std": float(np.std(scores[f"test_{config.primary_metric}"], ddof=1)),
                "train_primary": float(np.mean(scores[f"train_{config.primary_metric}"])),
            }
        )
        fold_records.extend(
            {
                "candidate": name,
                "fold": i + 1,
                **{m: float(scores[f"test_{m}"][i]) for m in scoring},
            }
            for i in range(config.cv_folds)
        )
        pd.DataFrame(records).to_csv(out / "cv_results.csv", index=False)
    comparison = pd.DataFrame(records).sort_values(
        config.primary_metric, ascending=False, kind="stable"
    )
    comparison.to_csv(out / "cv_results.csv", index=False)
    pd.DataFrame(fold_records).to_csv(out / "cv_folds.csv", index=False)
    winner = comparison.iloc[0]["candidate"]
    model = clone(models[winner]).fit(X, y)
    p_validation = model.predict_proba(data.X.iloc[va])[:, 1]
    threshold, thresholds = select_threshold(
        data.y.iloc[va], p_validation, config.primary_metric, config.min_recall
    )
    thresholds.to_csv(out / "threshold_search.csv", index=False)
    # Freeze model and threshold here; do not refit after selecting the threshold.
    p_test = model.predict_proba(data.X.iloc[te])[:, 1]
    predictions = (p_test >= threshold).astype(int)
    metrics = {
        "validation": metric_summary(data.y.iloc[va], p_validation, threshold),
        "holdout": metric_summary(data.y.iloc[te], p_test, threshold),
        "holdout_at_0_5": metric_summary(data.y.iloc[te], p_test, 0.5),
        "confusion_matrix": confusion_matrix(data.y.iloc[te], predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            data.y.iloc[te], predictions, labels=[0, 1], output_dict=True, zero_division=0
        ),
    }
    labels = {0: "No", 1: "Yes"} if PROJECT == "ev" else {0: "Not addicted", 1: "Addicted"}
    metadata = {
        "selected_model": winner,
        "threshold": threshold,
        "class_labels": labels,
        "target": TARGET,
        "required_features": RAW_FEATURES,
        "input_schema": raw_schema(RAW_FEATURES),
        "config": config.to_dict(),
        "data": data.summary,
        "data_sha256": file_hash(data_path),
        "partitions": {"train": len(tr), "validation": len(va), "holdout": len(te)},
        "metrics": metrics,
        "fit_partition": "train",
        "selection_metric": config.primary_metric,
    }
    path = save_artifact(model, out, metadata)
    reloaded = load_artifact(path)
    actual = predict_frame(reloaded, data.X.iloc[te])["probability_1"].to_numpy()
    np.testing.assert_allclose(actual, p_test, rtol=1e-12, atol=1e-12)
    pd.DataFrame(
        {
            "csv_row_index": data.X.iloc[te].index,
            "id": data.ids.iloc[te].to_numpy(),
            "actual": data.y.iloc[te].to_numpy(),
            "probability_1": p_test,
            "prediction": predictions,
        }
    ).to_csv(out / "holdout_predictions.csv", index=False)
    write_json(out / "metrics.json", metrics)
    write_json(out / "config.json", config.to_dict())
    print(f"Selected {winner}; threshold={threshold:.3f}; exported {path}", flush=True)
    return {"model_path": path, "metadata": reloaded["metadata"]}
