"""Build readable Markdown and HTML directly from saved evaluation results."""

import html
import json
from pathlib import Path

import pandas as pd

from .config import DEFAULT_OUTPUT_DIR
from .schema import PROJECT


def _value(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _table(frame):
    def cell(value):
        return _value(value).replace("|", r"\|").replace("\n", " ")

    rows = [
        "| " + " | ".join(map(cell, frame.columns)) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    rows += [
        "| " + " | ".join(map(cell, row)) + " |" for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join(rows)


def build(run_dir=None, output_dir=None):
    run_dir = Path(DEFAULT_OUTPUT_DIR if run_dir is None else run_dir)
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("project") != PROJECT:
        raise ValueError(f"Expected results for {PROJECT}.")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(run_dir / "cv_results.csv").head(15)
    out = Path(output_dir) if output_dir is not None else run_dir
    out.mkdir(parents=True, exist_ok=True)
    facts = pd.DataFrame(
        [
            {"Setting": "Selected model", "Value": meta["selected_model"]},
            {"Setting": "Created UTC", "Value": meta["created_utc"]},
            {"Setting": "Target", "Value": meta["target"]},
            {"Setting": "Fit partition", "Value": meta["fit_partition"]},
            {
                "Setting": "Train / validation / holdout rows",
                "Value": " / ".join(
                    str(meta["partitions"][k]) for k in ["train", "validation", "holdout"]
                ),
            },
        ]
    )
    if "threshold" in meta:
        facts.loc[len(facts)] = ["Validation-selected threshold", _value(meta["threshold"])]
    metric_rows = []
    for partition in ["holdout", "validation", "blend_validation", "holdout_at_0_5"]:
        if isinstance(metrics.get(partition), dict):
            metric_rows.extend(
                {"Partition": partition, "Metric": key, "Value": value}
                for key, value in metrics[partition].items()
            )
    scores = pd.DataFrame(metric_rows)
    text = (
        f"# {PROJECT} — training report\n\n"
        "Results read from this run's saved files; no training was performed to build this report.\n\n"
        "Interpret scores using the dataset actually supplied. Synthetic execution-check data cannot establish model quality.\n\n"
        f"## Run\n\n{_table(facts)}\n\n"
        f"## Evaluation\n\n{_table(scores)}\n\n"
        f"## Development CV comparison\n\n{_table(comparison)}\n\n"
        "CV scores were used for model selection. Holdout results are the final evaluation for this run.\n\n"
        "## Model export\n\n"
        "`model.joblib` contains the fitted preprocessing and estimator(s). "
        "`metadata.json` records inputs, configuration, dependency versions, and dataset/source hashes.\n"
    )
    markdown_path = out / "REPORT.md"
    markdown_path.write_text(text, encoding="utf-8")
    sections = [("Run", facts), ("Evaluation", scores), ("Development CV comparison", comparison)]
    body = "".join(
        f"<h2>{html.escape(title)}</h2>{frame.to_html(index=False, escape=True)}"
        for title, frame in sections
    )
    page = (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        f"<title>{html.escape(PROJECT)} — training report</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;color:#152536}"
        "table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}th,td{border-bottom:1px solid #cfd7df;padding:9px;text-align:left}"
        "th{background:#edf2f6}h2{margin-top:36px}@media print{body{margin:0}table{font-size:10px}}</style>"
        f"<h1>{html.escape(PROJECT)} — training report</h1>"
        "<p>Generated from saved run results. Synthetic-data scores only check execution.</p>"
        + body
        + "<p>The model export includes fitted preprocessing. See metadata.json for its input contract and dependencies.</p></html>"
    )
    html_path = out / "REPORT.html"
    html_path.write_text(page, encoding="utf-8")
    return markdown_path, html_path
