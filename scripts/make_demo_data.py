"""Run directly with Python; no notebook or package installation is needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


import argparse

from ev import config
from ev.demo import make_demo_data


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate synthetic rows for checking the Python scripts"
    )
    parser.add_argument("--output", type=Path, default=config.DATA_DIR / "demo.csv")
    parser.add_argument("--rows", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.exit(2, f"Error: Output already exists: {args.output}\n")
    try:
        frame = make_demo_data(args.rows, args.seed)
    except ValueError as exc:
        parser.exit(2, f"Error: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} synthetic rows to {args.output}; this is execution-check data.")


if __name__ == "__main__":
    main()
