import sys
sys.path.append("src")

import numpy as np
import pandas as pd
from monitoring.drift_metrics import detect_drift_all_features

np.random.seed(42)

reference_df = pd.DataFrame({
    "MonthlyCharges": np.random.normal(70, 20, 500),
    "tenure": np.random.normal(30, 15, 500),
    "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], 500)
})

current_df = reference_df.copy()
current_df["MonthlyCharges"] = current_df["MonthlyCharges"] + 15
current_df["Contract"] = np.random.choice(["Month-to-month", "One year", "Two year"], 500)

results = detect_drift_all_features(
    reference_df, current_df,
    numeric_features=["MonthlyCharges", "tenure"],
    categorical_features=["Contract"]
)

for r in results:
    print(r)