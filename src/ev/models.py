"""Candidate pipelines with preprocessing fitted inside cross-validation."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .features import FeatureEngineer
from .schema import PROJECT


class CatBoostCompat(ClassifierMixin, BaseEstimator):
    """Expose sklearn tags and flat predictions across CatBoost versions."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = np.asarray(self.estimator_.classes_)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return np.asarray(self.estimator_.predict(X)).reshape(-1)

    def predict_proba(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)


def make_estimator(name, config):
    seed, trees, jobs = config.seed, config.n_estimators, config.n_jobs
    if name == "dummy":
        return DummyClassifier(strategy="constant", constant=1)
    if name == "logistic_regression":
        return LogisticRegression(max_iter=3000, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=trees, max_depth=16, min_samples_leaf=5, random_state=seed, n_jobs=jobs
        )
    if name == "gaussian_nb":
        return GaussianNB()
    try:
        if name == "lightgbm":
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                n_estimators=trees,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=30,
                subsample=0.9,
                subsample_freq=1,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=seed,
                n_jobs=jobs,
                verbosity=-1,
                deterministic=True,
                force_col_wise=True,
            )
        if name == "xgboost":
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=trees,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                tree_method="hist",
                eval_metric="logloss",
                random_state=seed,
                n_jobs=jobs,
            )
        if name == "catboost":
            from catboost import CatBoostClassifier

            return CatBoostCompat(
                CatBoostClassifier(
                    iterations=trees,
                    learning_rate=0.05,
                    depth=6,
                    loss_function="Logloss",
                    random_seed=seed,
                    verbose=False,
                    thread_count=jobs,
                    allow_writing_files=False,
                )
            )
    except ImportError as exc:
        raise ImportError(
            f"Model {name} needs optional dependencies: pip install -e '.[boosting]'"
        ) from exc
    raise ValueError(f"Unknown model: {name}")


def build_pipeline(name, feature_set, config):
    native = feature_set == "native"
    if native and name not in {"lightgbm", "xgboost"}:
        raise ValueError("Native missing-value features require LightGBM or XGBoost")
    if PROJECT == "ev":
        features = FeatureEngineer(engineered=feature_set != "original")
    else:
        features = FeatureEngineer(
            engineered=feature_set != "original",
            missing_flags=feature_set in {"engineered_missing", "selected", "native"},
            imputation="native" if native else "median",
        )
    preprocess = ColumnTransformer(
        [
            (
                "num",
                "passthrough"
                if native
                else SimpleImputer(strategy="median", keep_empty_features=True),
                make_column_selector(dtype_include=np.number),
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float32,
                    max_categories=128,
                ),
                make_column_selector(dtype_exclude=np.number),
            ),
        ],
        sparse_threshold=0,
    )
    steps = [("features", features), ("preprocess", preprocess)]
    if feature_set == "selected":
        selector = make_estimator("lightgbm", config)
        selector.set_params(n_estimators=min(100, config.n_estimators))
        steps.append(("selection", SelectFromModel(selector, threshold="mean")))
    if name in {"logistic_regression", "gaussian_nb"} or config.smote:
        steps.append(("scaler", StandardScaler()))
    pipeline_type = Pipeline
    if config.smote:
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbalancedPipeline
        except ImportError as exc:
            raise ImportError("SMOTE needs: pip install -e '.[smote]'") from exc
        pipeline_type = ImbalancedPipeline
        steps.append(("smote", SMOTE(random_state=config.seed, k_neighbors=3)))
    steps.append(("model", make_estimator(name, config)))
    return pipeline_type(steps)


def candidates(config):
    result = {}
    for name in dict.fromkeys(["dummy", *config.models]):
        if name in {"voting", "stacking"}:
            continue
        for variant in config.feature_sets:
            if variant == "native" and name not in {"lightgbm", "xgboost"}:
                continue
            if name == "dummy" and variant != config.feature_sets[0]:
                continue
            result[f"{name}__{variant}"] = build_pipeline(name, variant, config)
    for name in ["voting", "stacking"]:
        if name not in config.models:
            continue
        variant = next((v for v in config.feature_sets if v != "native"), "engineered")
        parts = [
            (family, build_pipeline(family, variant, config))
            for family in ["lightgbm", "xgboost", "catboost"]
        ]
        if name == "voting":
            result[name] = VotingClassifier(parts, voting="soft", n_jobs=1)
        else:
            inner = StratifiedKFold(
                n_splits=config.cv_folds, shuffle=True, random_state=config.seed + 5
            )
            result[name] = StackingClassifier(
                parts,
                final_estimator=LogisticRegression(max_iter=2000),
                cv=inner,
                stack_method="predict_proba",
                n_jobs=1,
            )
    if not result:
        raise ValueError("No compatible model and feature-set combinations")
    return result


def search_space(name):
    family = name.split("__")[0]
    return {
        "logistic_regression": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__class_weight": [None, "balanced"],
        },
        "random_forest": {"model__max_depth": [8, 16, None], "model__min_samples_leaf": [2, 5, 10]},
        "gaussian_nb": {"model__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]},
        "lightgbm": {
            "model__num_leaves": [15, 31, 63],
            "model__min_child_samples": [10, 30, 60],
            "model__learning_rate": [0.025, 0.05, 0.1],
        },
        "xgboost": {
            "model__max_depth": [3, 5, 7],
            "model__min_child_weight": [1, 3, 7],
            "model__learning_rate": [0.025, 0.05, 0.1],
        },
        "catboost": {
            "model__estimator__depth": [4, 6, 8],
            "model__estimator__l2_leaf_reg": [1.0, 3.0, 8.0],
        },
    }.get(family, {})
