"""CSV validation, target parsing and auditable row identity."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import CAT_COLS, NUM_COLS, PROJECT, RAW_FEATURES, TARGET


def read_csv(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    columns = [c.strip() for c in header]
    if not columns or any(not c for c in columns) or len(set(columns)) != len(columns):
        raise ValueError("CSV must have unique, nonempty column names after trimming whitespace.")
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
        dtype={c: "string" for c in header if c.strip() in {"id", "host_id"}},
    )
    frame.columns = columns
    if frame.empty:
        raise ValueError("CSV contains no data rows.")
    return frame


def require_columns(frame, columns):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Inputs must be a pandas DataFrame of raw CSV columns.")
    if frame.columns.duplicated().any():
        raise ValueError("Input contains duplicate column names.")
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


@dataclass
class TrainingData:
    X: pd.DataFrame
    y: pd.Series
    ids: pd.Series
    groups: pd.Series | None
    summary: dict


def load_training_data(path, group_by_host=True):
    frame = read_csv(path)
    required = RAW_FEATURES + [TARGET]
    if PROJECT == "airbnb" and group_by_host:
        required += ["host_id"]
    require_columns(frame, required)
    original_rows = len(frame)
    frame = frame.drop_duplicates().copy()
    duplicates = original_rows - len(frame)
    if "id" in frame and frame["id"].dropna().duplicated().any():
        raise ValueError(
            "Repeated IDs remain after exact deduplication; resolve repeated records first."
        )
    removed = 0
    if PROJECT == "airbnb":
        values = frame[TARGET].astype("string").str.replace(r"[$,\s]", "", regex=True)
        y = pd.to_numeric(values, errors="coerce").astype(float)
        valid = np.isfinite(y) & (y > 0)
        removed = int((~valid).sum())
        frame, y = frame.loc[valid].copy(), y.loc[valid]
        if len(frame) < 100:
            raise ValueError("At least 100 valid, positive-price listings are required.")
        from .features import EXTRA_CAT, EXTRA_NUM, EXTRA_TEXT

        features = RAW_FEATURES + [c for c in EXTRA_NUM + EXTRA_CAT + EXTRA_TEXT if c in frame]
    else:
        values = frame[TARGET].astype("string").str.strip().str.lower()
        if PROJECT == "ev":
            values = values.replace({"yes": "1", "no": "0"})
        y = pd.to_numeric(values, errors="coerce")
        if y.isna().any() or not y.isin([0, 1]).all():
            raise ValueError(
                f"{TARGET} must contain non-missing binary labels (0/1"
                + (" or Yes/No)." if PROJECT == "ev" else ").")
            )
        y = y.astype(int)
        if y.nunique() != 2 or y.value_counts().min() < 30:
            raise ValueError("Both target classes need at least 30 rows.")
        features = RAW_FEATURES
    groups = None
    if PROJECT == "airbnb" and group_by_host:
        host = frame["host_id"].astype("string").str.strip().replace("", pd.NA)
        fallback = pd.Series([f"missing_host_row:{i}" for i in frame.index], index=frame.index)
        groups = ("host:" + host).fillna(fallback).astype(str)
    ids = frame["id"].copy() if "id" in frame else pd.Series(frame.index, index=frame.index)
    return TrainingData(
        frame[features].copy(),
        y,
        ids,
        groups,
        {
            "input_rows": original_rows,
            "exact_duplicates_removed": duplicates,
            "invalid_target_rows_removed": removed,
            "usable_rows": len(frame),
        },
    )


def raw_schema(columns):
    return {
        c: {
            "type": "number"
            if c in NUM_COLS
            else "category"
            if c in CAT_COLS
            else "string_or_number",
            "missing_values_allowed": True,
        }
        for c in columns
    }
