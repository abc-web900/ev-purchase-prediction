"""Run directly with Python; no notebook or package installation is needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


import argparse

from ev import config
from ev.predict import predict_csv


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Predict from raw CSV rows using an exported ev model"
    )
    parser.add_argument("--model", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--data", type=Path, default=config.PREDICT_PATH)
    parser.add_argument("--output", type=Path, default=config.PREDICTIONS_PATH)
    args = parser.parse_args(argv)
    try:
        predictions = predict_csv(args.model, args.data, args.output)
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
