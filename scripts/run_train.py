"""Run directly with Python; no notebook or package installation is needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

from ev import config, pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train, validate, and export the ev model")
    parser.add_argument("--data", type=Path, default=config.TRAIN_PATH)
    parser.add_argument("--output", type=Path, default=config.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--profile", choices=["default", "full"], default="default")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--feature-sets", nargs="+")
    for option in ["seed", "cv-folds", "n-estimators", "n-jobs", "search-iterations"]:
        parser.add_argument("--" + option, type=int)
    parser.add_argument("--smote", action="store_true", default=None)
    args = parser.parse_args(argv)
    overrides = {
        key: getattr(args, key)
        for key in [
            "models",
            "feature_sets",
            "seed",
            "cv_folds",
            "n_estimators",
            "n_jobs",
            "search_iterations",
        ]
    }
    overrides["smote"] = args.smote
    try:
        result = pipeline.run(args.data, args.output, config.make_config(args.profile, **overrides))
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"Model saved to {result['model_path']}")
    print(f"Evaluation saved to {args.output / 'metrics.json'}")


if __name__ == "__main__":
    main()
