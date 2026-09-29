"""Synthetic survey rows for execution checks, not model-quality claims."""

import numpy as np
import pandas as pd


def make_demo_data(rows=400, seed=2026):
    if rows < 100:
        raise ValueError("Use at least 100 synthetic rows.")
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "id": [str(i) for i in range(rows)],
            "Age": rng.integers(18, 76, rows),
            "Annual_Income_USD": rng.uniform(20000, 160000, rows),
            "Daily_Commute_km": rng.uniform(1, 80, rows),
            "Number_of_Cars_Owned": rng.integers(0, 4, rows),
            "Charging_Stations_Near_Home": rng.integers(0, 16, rows),
            "Charging_Stations_Near_Work": rng.integers(0, 20, rows),
            "Environmental_Concern_Level": rng.integers(1, 6, rows).astype(float),
            "Gender": rng.choice(["Male", "Female", "Other"], rows),
            "City_Type": rng.choice(["Urban", "Suburban", "Rural"], rows),
            "Current_Car_Type": rng.choice(["Sedan", "SUV", "Hatchback"], rows),
            "Home_Charging_Possible": rng.choice(["Yes", "No"], rows),
            "Subsidy_Available": rng.choice(["Yes", "No"], rows),
            "Range_Anxiety_Level": rng.choice(["Low", "Medium", "High"], rows),
        }
    )
    score = (
        frame["Environmental_Concern_Level"]
        + (frame["Home_Charging_Possible"] == "Yes")
        + rng.normal(0, 1, rows)
    )
    frame["Will_Buy_EV"] = np.where(score > np.median(score), "Yes", "No")
    frame.loc[rng.choice(rows, rows // 12, replace=False), "Annual_Income_USD"] = np.nan
    return frame
