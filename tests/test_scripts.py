"""Direct scripts work independently of the caller's working directory."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_demo_and_eda_scripts(tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    csv = tmp_path / "demo.csv"
    out = tmp_path / "eda"
    for arguments in [
        ["make_demo_data.py", "--output", str(csv), "--rows", "120"],
        ["run_eda.py", "--data", str(csv), "--output", str(out)],
    ]:
        completed = subprocess.run(
            [sys.executable, str(scripts / arguments[0]), *arguments[1:]],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
    assert json.loads((out / "summary.json").read_text())["usable_rows"] == 120
    assert (out / "target_distribution.png").stat().st_size > 0
    assert (out / "numeric_correlations.png").stat().st_size > 0
    assert not pd.read_csv(out / "target_associations.csv").empty
