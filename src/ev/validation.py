"""Binary evaluation and efficient validation-only threshold selection."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def metric_summary(y_true, probabilities, threshold=0.5):
    pred = (np.asarray(probabilities) >= threshold).astype(np.int8)

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, pred),
        "precision_1": precision_score(y_true, pred, zero_division=0),
        "recall_1": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "f1_0": f1_score(y_true, pred, pos_label=0, zero_division=0),
        "f1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1]),
        "brier_score": brier_score_loss(y_true, probabilities),
    }


def threshold_table(y_true, probabilities):
    """
    Efficient threshold search using sorted probabilities
    and cumulative confusion-matrix counts.
    """

    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(y_true, dtype=np.int64)

    order = np.argsort(p)
    p, y = p[order], y[order]

    prefix = np.r_[0, np.cumsum(y)]
    thresholds = np.linspace(0.0, 1.0, 201)

    cut = np.searchsorted(
        p,
        thresholds,
        side="left",
    )

    fn = prefix[cut]
    tp = y.sum() - fn
    tn = cut - fn
    fp = len(y) - cut - tp

    def divide(a, b):
        return np.divide(
            a,
            b,
            out=np.zeros_like(a, dtype=float),
            where=b != 0,
        )

    f1_1 = divide(
        2 * tp,
        2 * tp + fp + fn,
    )

    f1_0 = divide(
        2 * tn,
        2 * tn + fp + fn,
    )

    return pd.DataFrame(
        {
            "threshold": thresholds,
            "f1": f1_1,
            "f1_macro": (f1_1 + f1_0) / 2,
            "balanced_accuracy": (divide(tp, tp + fn) + divide(tn, tn + fp)) / 2,
            "precision_1": divide(tp, tp + fp),
            "recall_1": divide(tp, tp + fn),
        }
    )


def select_threshold(y, probabilities, metric="f1", min_recall=0.0):
    table = threshold_table(y, probabilities)
    table["distance_to_0.5"] = abs(table["threshold"] - 0.5)
    eligible = table.loc[table["recall_1"] >= min_recall]
    # Threshold 0 predicts every finite probability positive, so this is always feasible.
    order = (
        ["precision_1", metric, "distance_to_0.5"]
        if min_recall > 0
        else [metric, "distance_to_0.5"]
    )
    ascending = [False, False, True] if min_recall > 0 else [False, True]
    best = eligible.sort_values(order, ascending=ascending, kind="stable").iloc[0]
    return float(best["threshold"]), table


def split_data(data, config):
    trainval, holdout = train_test_split(
        np.arange(len(data.X)),
        test_size=config.holdout_size,
        stratify=data.y,
        random_state=config.seed,
    )
    train, validation = train_test_split(
        trainval,
        test_size=config.validation_size / (1 - config.holdout_size),
        stratify=data.y.iloc[trainval],
        random_state=config.seed + 1,
    )
    for positions in [train, validation, holdout]:
        if data.y.iloc[positions].nunique() != 2:
            raise ValueError(
                "Every partition must contain both classes; use more data or larger splits."
            )
    if data.y.iloc[train].value_counts().min() < config.cv_folds:
        raise ValueError("Too few training rows in the minority class for cv_folds.")
    return train, validation, holdout
