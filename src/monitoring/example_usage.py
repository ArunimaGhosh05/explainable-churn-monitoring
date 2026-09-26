"""
Example: how to call B's drift + RCA functions from C's pipeline.py
"""
import sys
sys.path.append("src")

import json
import pandas as pd
from monitoring.drift_metrics import detect_drift_all_features
from monitoring.rca import compute_rca

# 1. Load reference and current data (A provides these)
reference_df = pd.read_csv("data/raw/reference.csv")
current_df = pd.read_csv("data/raw/current.csv")   # or a scenario-modified version

# 2. Define feature lists (same ones B used)
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

# 3. Run drift detection -> this is the drift_report
drift_results = detect_drift_all_features(
    reference_df, current_df,
    numeric_features=NUMERIC_FEATURES,
    categorical_features=CATEGORICAL_FEATURES
)

# 4. Load A's real SHAP importance
with open("results/tables/shap_importance.json") as f:
    shap_importance = json.load(f)

# 5. Run RCA -> this is the rca_report
rca_df = compute_rca(drift_results, shap_importance)

# these two are what C's decision engine and LLM need
print("drift_results (list of dicts):", drift_results[:2])
print("\nrca_df (pandas DataFrame):")
print(rca_df.head())