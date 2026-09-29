"""Data summaries, notebook-style associations, and saved figures."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .artifacts import write_json
from .config import OUTPUT_DIR, TRAIN_PATH
from .data import load_training_data, read_csv
from .schema import CAT_COLS, NUM_COLS, PROJECT, TARGET


def run(data_path=None, output_dir=None):
    """Explore validated raw data without fitting or exporting a predictive model."""
    data_path = TRAIN_PATH if data_path is None else data_path
    out = Path(OUTPUT_DIR / "eda" if output_dir is None else output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new or empty EDA output directory.")
    raw = read_csv(data_path)
    data = load_training_data(data_path, group_by_host=False)
    numeric = data.X[NUM_COLS].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    out.mkdir(parents=True, exist_ok=True)
    missing = raw.isna().sum().rename("missing_rows").to_frame()
    missing["missing_percent"] = 100 * missing["missing_rows"] / len(raw)
    missing.to_csv(out / "missing_values.csv", index_label="column")
    numeric.describe().T.to_csv(out / "numeric_summary.csv", index_label="feature")
    numeric.corr().to_csv(out / "numeric_correlations.csv", index_label="feature")
    numeric.cov().to_csv(out / "numeric_covariance.csv", index_label="feature")
    categories = []
    for column in CAT_COLS:
        counts = data.X[column].astype("string").fillna("(missing)").value_counts()
        categories.extend(
            {"feature": column, "category": str(value), "count": int(count)}
            for value, count in counts.items()
        )
    pd.DataFrame(categories).to_csv(out / "category_counts.csv", index=False)
    associations = []
    for column in NUM_COLS:
        valid = numeric[column].notna()
        values = numeric.loc[valid, column]
        labels = data.y.loc[valid]
        if values.nunique() < 2 or len(values) < 3:
            continue
        if PROJECT == "airbnb":
            if labels.nunique() < 2:
                continue
            result = stats.spearmanr(values, labels)
            test = "spearman"
        else:
            groups = [values.loc[labels == label] for label in [0, 1]]
            if min(map(len, groups)) < 2 or all(group.nunique() < 2 for group in groups):
                continue
            result = stats.f_oneway(*groups)
            test = "anova"
        associations.append(
            {
                "feature": column,
                "test": test,
                "statistic": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )
    if PROJECT != "airbnb":
        for column in CAT_COLS:
            table = pd.crosstab(data.X[column].astype("string").fillna("(missing)"), data.y)
            if min(table.shape) < 2:
                continue
            chi2, p_value, _, _ = stats.chi2_contingency(table)
            associations.append(
                {
                    "feature": column,
                    "test": "chi_square",
                    "statistic": float(chi2),
                    "p_value": float(p_value),
                }
            )
    pd.DataFrame(associations, columns=["feature", "test", "statistic", "p_value"]).to_csv(
        out / "target_associations.csv", index=False
    )
    if PROJECT == "airbnb":
        target_summary = {
            key: float(value)
            for key, value in data.y.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).items()
        }
    else:
        target_summary = {
            str(label): int(count) for label, count in data.y.value_counts().sort_index().items()
        }
    summary = {
        "project": PROJECT,
        **data.summary,
        "target": TARGET,
        "raw_columns": len(raw.columns),
        "predictor_columns": len(data.X.columns),
        "target_summary": target_summary,
    }
    write_json(out / "summary.json", summary)
    _plots(numeric, data.y, out)
    return {"summary": summary, "output_dir": out}


def _plots(numeric, target, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if PROJECT == "airbnb":
        ax.hist(target, bins=40, color="#25638d", edgecolor="white")
        ax.set_xlabel("Listing price (original CSV units)")
    else:
        counts = target.value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color="#25638d")
        ax.set_xlabel(TARGET)
    ax.set_ylabel("Rows")
    ax.set_title("Target distribution")
    fig.tight_layout()
    fig.savefig(out / "target_distribution.png", dpi=150)
    plt.close(fig)

    axes = numeric.hist(bins=25, figsize=(13, 10), color="#25638d", edgecolor="white")
    fig = np.asarray(axes).flat[0].figure
    fig.suptitle("Raw numeric predictors", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "numeric_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    grid = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=65, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    fig.colorbar(grid, ax=ax, label="Pearson correlation")
    ax.set_title("Numeric predictor correlations")
    fig.tight_layout()
    fig.savefig(out / "numeric_correlations.png", dpi=150)
    plt.close(fig)
