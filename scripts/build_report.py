"""Run directly with Python; no notebook or package installation is needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


import argparse

from ev import config, report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a Markdown and HTML report from an existing training run"
    )
    parser.add_argument("--run", type=Path, default=config.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        paths = report.build(args.run, args.output)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
