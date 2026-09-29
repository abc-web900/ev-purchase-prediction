"""EV charging, affordability, anxiety, and demographic interactions."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .data import require_columns
from .schema import CAT_COLS, NUM_COLS, RAW_FEATURES


def add_features_v2(data):

    data = data.copy()

    # ========================================================
    # A. CHARGING INFRASTRUCTURE
    # ========================================================

    data["Total_Charging_Stations"] = (
        data["Charging_Stations_Near_Home"] + data["Charging_Stations_Near_Work"]
    )

    data["Stations_per_Commute"] = data["Total_Charging_Stations"] / (data["Daily_Commute_km"] + 1)

    data["Commute_per_Station"] = data["Daily_Commute_km"] / (data["Total_Charging_Stations"] + 1)

    data["Station_Difference"] = (
        data["Charging_Stations_Near_Home"] - data["Charging_Stations_Near_Work"]
    )

    data["Station_Difference_Abs"] = data["Station_Difference"].abs()

    data["Home_Station_Share"] = data["Charging_Stations_Near_Home"] / (
        data["Total_Charging_Stations"] + 1
    )

    data["Work_Station_Share"] = data["Charging_Stations_Near_Work"] / (
        data["Total_Charging_Stations"] + 1
    )

    data["Stations_per_Car"] = data["Total_Charging_Stations"] / (data["Number_of_Cars_Owned"] + 1)

    # ========================================================
    # B. INCOME / AFFORDABILITY
    # ========================================================

    data["Income_per_Car"] = data["Annual_Income_USD"] / (data["Number_of_Cars_Owned"] + 1)

    data["Income_per_Commute"] = data["Annual_Income_USD"] / (data["Daily_Commute_km"] + 1)

    # Log transforms
    data["Log_Income"] = np.log1p(data["Annual_Income_USD"].clip(lower=0))

    data["Log_Commute"] = np.log1p(data["Daily_Commute_km"].clip(lower=0))

    data["Log_Total_Stations"] = np.log1p(data["Total_Charging_Stations"].clip(lower=0))

    # ========================================================
    # C. BINARY FEATURES
    # ========================================================

    data["HomeCharge_Binary"] = data["Home_Charging_Possible"].map({"Yes": 1, "No": 0})

    data["Subsidy_Binary"] = data["Subsidy_Available"].map({"Yes": 1, "No": 0})

    # ========================================================
    # D. RANGE ANXIETY ORDINAL FEATURE
    # ========================================================

    range_map = {"Low": 0, "Medium": 1, "High": 2}

    data["Range_Anxiety_Ordinal"] = data["Range_Anxiety_Level"].map(range_map)

    # ========================================================
    # E. INTERACTION FEATURES
    # ========================================================

    data["HomeCharge_And_Subsidy"] = data["HomeCharge_Binary"] * data["Subsidy_Binary"]

    data["Concern_x_Subsidy"] = data["Environmental_Concern_Level"] * data["Subsidy_Binary"]

    data["Concern_x_HomeCharge"] = data["Environmental_Concern_Level"] * data["HomeCharge_Binary"]

    data["Commute_x_HomeCharge"] = data["Daily_Commute_km"] * data["HomeCharge_Binary"]

    data["Income_x_Subsidy"] = data["Log_Income"] * data["Subsidy_Binary"]

    data["Income_x_HomeCharge"] = data["Log_Income"] * data["HomeCharge_Binary"]

    data["Income_x_Concern"] = data["Log_Income"] * data["Environmental_Concern_Level"]

    data["Commute_x_RangeAnxiety"] = data["Daily_Commute_km"] * data["Range_Anxiety_Ordinal"]

    data["Stations_x_RangeAnxiety"] = (
        data["Total_Charging_Stations"] * data["Range_Anxiety_Ordinal"]
    )

    data["HomeCharge_x_RangeAnxiety"] = data["HomeCharge_Binary"] * data["Range_Anxiety_Ordinal"]

    data["Concern_x_RangeAnxiety"] = (
        data["Environmental_Concern_Level"] * data["Range_Anxiety_Ordinal"]
    )

    data["Age_x_Income"] = data["Age"] * data["Log_Income"]

    data["Age_x_Concern"] = data["Age"] * data["Environmental_Concern_Level"]

    # ========================================================
    # F. NON-LINEAR NUMERIC FEATURES
    #
    # Logistic Regression is linear, so these allow it
    # to model curved relationships.
    # ========================================================

    data["Age_Squared"] = data["Age"] ** 2

    data["Commute_Squared"] = data["Daily_Commute_km"] ** 2

    data["Concern_Squared"] = data["Environmental_Concern_Level"] ** 2

    # ========================================================
    # G. AGE BANDS
    # ========================================================

    data["Age_Band"] = pd.cut(
        data["Age"],
        bins=[-np.inf, 25, 35, 45, 55, 65, np.inf],
        labels=["<=25", "26-35", "36-45", "46-55", "56-65", "65+"],
    )

    # ========================================================
    # H. INCOME BANDS
    # ========================================================

    data["Income_Band"] = pd.cut(
        data["Annual_Income_USD"],
        bins=[-np.inf, 30000, 50000, 75000, 100000, 150000, np.inf],
        labels=["<30k", "30-50k", "50-75k", "75-100k", "100-150k", "150k+"],
    )

    # ========================================================
    # I. COMMUTE BANDS
    # ========================================================

    data["Commute_Band"] = pd.cut(
        data["Daily_Commute_km"],
        bins=[-np.inf, 5, 15, 30, 50, np.inf],
        labels=["<5", "5-15", "15-30", "30-50", "50+"],
    )

    return data


class FeatureEngineer(TransformerMixin, BaseEstimator):
    def __init__(self, engineered=True):
        self.engineered = engineered

    def _clean(self, X):
        require_columns(X, RAW_FEATURES)
        z = X[RAW_FEATURES].copy()
        for c in NUM_COLS:
            z[c] = pd.to_numeric(z[c], errors="coerce").astype(float)
        z[NUM_COLS] = z[NUM_COLS].replace([np.inf, -np.inf], np.nan)
        for c in CAT_COLS:
            z[c] = z[c].astype("string").str.strip().replace("", pd.NA)
        return z

    def fit(self, X, y=None):
        z = self._clean(X)
        self.medians_ = z[NUM_COLS].median().fillna(0.0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "medians_")
        z = self._clean(X)
        z[NUM_COLS] = z[NUM_COLS].fillna(self.medians_)
        for c in CAT_COLS:
            z[c] = z[c].fillna("Unknown").astype(object)
        if self.engineered:
            z = add_features_v2(z)
        for c in z.select_dtypes(exclude=np.number):
            z[c] = z[c].astype("string").fillna("Unknown").astype(object)
        numeric = z.select_dtypes(include=np.number).columns
        z[numeric] = z[numeric].replace([np.inf, -np.inf], np.nan)
        return z
