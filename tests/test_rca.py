import sys
sys.path.append("src")

import numpy as np
import pandas as pd
from monitoring.drift_simulator import get_scenario_data
from monitoring.drift_metrics import detect_drift_all_features
from monitoring.rca import compute_rca, top_k_contributors

np.random.seed(1)
current_pool = pd.DataFrame({
    "MonthlyCharges": np.random.normal(70, 20, 1000),
    "tenure": np.random.normal(30, 15, 1000),
    "SeniorCitizen": np.random.normal(0.2, 0.4, 1000),
})
reference = pd.DataFrame({
    "MonthlyCharges": np.random.normal(70, 20, 1000),
    "tenure": np.random.normal(30, 15, 1000),
    "SeniorCitizen": np.random.normal(0.2, 0.4, 1000),
})

# FAKE shap importance until A delivers the real one
# pretend MonthlyCharges is the most important feature, SeniorCitizen the least
fake_shap_importance = {
    "MonthlyCharges": 0.45,
    "tenure": 0.30,
    "SeniorCitizen": 0.05
}

# S1: drift injected on MonthlyCharges (important feature)
current_s1 = get_scenario_data("S1", current_pool, important_feature="MonthlyCharges", seed=1)
drift_results_s1 = detect_drift_all_features(
    reference, current_s1,
    numeric_features=["MonthlyCharges", "tenure", "SeniorCitizen"],
    categorical_features=[]
)

rca_df_s1 = compute_rca(drift_results_s1, fake_shap_importance)
print("--- S1 RCA (drift on important feature) ---")
print(rca_df_s1)
print("Top-3 by RCA score:", top_k_contributors(rca_df_s1, k=3, rank_column="rca_rank"))

# S2: drift injected on SeniorCitizen (unimportant feature)
current_s2 = get_scenario_data("S2", current_pool, unimportant_feature="SeniorCitizen", seed=1)
drift_results_s2 = detect_drift_all_features(
    reference, current_s2,
    numeric_features=["MonthlyCharges", "tenure", "SeniorCitizen"],
    categorical_features=[]
)

rca_df_s2 = compute_rca(drift_results_s2, fake_shap_importance)
print("\n--- S2 RCA (drift on unimportant feature) ---")
print(rca_df_s2)
print("Top-3 by RCA score:", top_k_contributors(rca_df_s2, k=3, rank_column="rca_rank"))