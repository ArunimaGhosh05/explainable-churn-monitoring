import sys
sys.path.append("src")

import numpy as np
import pandas as pd
from monitoring.drift_simulator import get_scenario_data
from monitoring.drift_metrics import detect_drift_all_features

np.random.seed(1)
current_pool = pd.DataFrame({
    "MonthlyCharges": np.random.normal(70, 20, 1000),
    "tenure": np.random.normal(30, 15, 1000),
})
reference = pd.DataFrame({
    "MonthlyCharges": np.random.normal(70, 20, 1000),
    "tenure": np.random.normal(30, 15, 1000),
})

for scenario in ["S0", "S1"]:
    current = get_scenario_data(scenario, current_pool, important_feature="MonthlyCharges", seed=1)
    results = detect_drift_all_features(reference, current, numeric_features=["MonthlyCharges", "tenure"], categorical_features=[])
    print(f"\n--- {scenario} ---")
    for r in results:
        print(r)

# Test S2 separately since it needs an "unimportant_feature" argument
current_s2 = get_scenario_data(
    "S2",
    current_pool,
    unimportant_feature="tenure",   # placeholder — replace with A's real lowest-SHAP feature later
    std_multiplier=1.0,
    seed=1
)
results_s2 = detect_drift_all_features(
    reference, current_s2,
    numeric_features=["MonthlyCharges", "tenure"],
    categorical_features=[]
)
print("\n--- S2 ---")
for r in results_s2:
    print(r)