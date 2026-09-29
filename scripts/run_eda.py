"""Run directly with Python; no notebook or package installation is needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


import argparse

from ev import config, eda


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize and plot the ev training data")
    parser.add_argument("--data", type=Path, default=config.TRAIN_PATH)
    parser.add_argument("--output", type=Path, default=config.OUTPUT_DIR / "eda")
    args = parser.parse_args(argv)
    try:
        result = eda.run(args.data, args.output)
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    for key, value in result["summary"].items():
        print(f"{key:30} {value}")
    print(f"EDA tables and figures saved to {result['output_dir']}")


if __name__ == "__main__":
    main()
