"""Persist the complete fitted workflow and a human-readable manifest."""

import hashlib
import importlib.metadata
import json
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from .schema import PROJECT

FORMAT_VERSION = 1


def write_json(path, value):
    def convert(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Cannot serialize {type(obj).__name__}")

    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment():
    versions = {}
    for package in [
        PROJECT,
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "joblib",
        "threadpoolctl",
        "lightgbm",
        "xgboost",
        "xgboost-cpu",
        "catboost",
        "imbalanced-learn",
    ]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def source_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def save_artifact(model, directory, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    versions = environment()
    metadata = {
        **metadata,
        "project": PROJECT,
        "format_version": FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "source_sha256": source_hash(),
        "versions": versions,
    }
    bundle = {"model": model, "metadata": metadata}
    pending = directory / "model.joblib.tmp"
    path = directory / "model.joblib"
    joblib.dump(bundle, pending, compress=3)
    pending.replace(path)
    write_json(directory / "metadata.json", {**metadata, "model_sha256": file_hash(path)})
    # Install this project from the same revision separately.
    (directory / "requirements.lock.txt").write_text(
        "\n".join(
            f"{name}=={version}" for name, version in sorted(versions.items()) if name != PROJECT
        )
        + "\n"
    )
    return path


def load_artifact(path):
    """Load only trusted joblib files; deserialization can execute Python code."""
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or not {"model", "metadata"} <= bundle.keys():
        raise ValueError("This file is not a supported model artifact.")
    meta = bundle["metadata"]
    if meta.get("project") != PROJECT or meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Expected a {PROJECT} model with artifact format {FORMAT_VERSION}.")
    if meta.get("source_sha256") != source_hash():
        warnings.warn(
            "Package source differs from the training code.", RuntimeWarning, stacklevel=2
        )
    current = environment()
    mismatched = [
        name for name, version in meta.get("versions", {}).items() if current.get(name) != version
    ]
    if mismatched:
        warnings.warn(
            f"Artifact dependency versions differ: {mismatched}. Recreate its training environment.",
            RuntimeWarning,
            stacklevel=2,
        )
    return bundle
