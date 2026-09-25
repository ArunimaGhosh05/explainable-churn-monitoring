import sys
import os
sys.path.append("src")

import numpy as np
import pandas as pd
from monitoring.drift_simulator import get_scenario_data
from monitoring.drift_metrics import detect_drift_all_features
from monitoring.rca import compute_rca, top_k_contributors

# ---- CONFIG (swap these for real values once A delivers data) ----
SEED = 1
N_SEEDS = 5  # run each scenario multiple times, as planned
NUMERIC_FEATURES = ["MonthlyCharges", "tenure", "SeniorCitizen"]
CATEGORICAL_FEATURES = []  # add real categorical columns once using Telco data
IMPORTANT_FEATURE = "MonthlyCharges"      # placeholder until A gives real top SHAP feature
UNIMPORTANT_FEATURE = "SeniorCitizen"     # placeholder until A gives real lowest SHAP feature

# FAKE shap importance until A delivers the real one
FAKE_SHAP_IMPORTANCE = {
    "MonthlyCharges": 0.45,
    "tenure": 0.30,
    "SeniorCitizen": 0.05
}

os.makedirs("results/tables", exist_ok=True)


def make_fake_data(seed):
    """Placeholder data generator — replace with A's real reference/current-pool once ready."""
    np.random.seed(seed)
    reference = pd.DataFrame({
        "MonthlyCharges": np.random.normal(70, 20, 1000),
        "tenure": np.random.normal(30, 15, 1000),
        "SeniorCitizen": np.random.normal(0.2, 0.4, 1000),
    })
    current_pool = pd.DataFrame({
        "MonthlyCharges": np.random.normal(70, 20, 1000),
        "tenure": np.random.normal(30, 15, 1000),
        "SeniorCitizen": np.random.normal(0.2, 0.4, 1000),
    })
    return reference, current_pool


def run_all_scenarios():
    all_drift_rows = []
    all_rca_rows = []

    for seed in range(N_SEEDS):
        reference, current_pool = make_fake_data(seed)

        for scenario in ["S0", "S1", "S2"]:
            if scenario == "S0":
                current = get_scenario_data("S0", current_pool, seed=seed)
            elif scenario == "S1":
                current = get_scenario_data("S1", current_pool, important_feature=IMPORTANT_FEATURE, seed=seed)
            elif scenario == "S2":
                current = get_scenario_data("S2", current_pool, unimportant_feature=UNIMPORTANT_FEATURE, seed=seed)

            drift_results = detect_drift_all_features(
                reference, current,
                numeric_features=NUMERIC_FEATURES,
                categorical_features=CATEGORICAL_FEATURES
            )

            for r in drift_results:
                r["scenario"] = scenario
                r["seed"] = seed
                all_drift_rows.append(r)

            rca_df = compute_rca(drift_results, FAKE_SHAP_IMPORTANCE)
            rca_df["scenario"] = scenario
            rca_df["seed"] = seed
            all_rca_rows.append(rca_df)

    drift_df = pd.DataFrame(all_drift_rows)
    rca_df_all = pd.concat(all_rca_rows, ignore_index=True)

    drift_df.to_csv("results/tables/drift_results.csv", index=False)
    rca_df_all.to_csv("results/tables/rca_results.csv", index=False)

    print(f"Saved {len(drift_df)} drift rows to results/tables/drift_results.csv")
    print(f"Saved {len(rca_df_all)} RCA rows to results/tables/rca_results.csv")

    # quick summary: how often does the drifted feature rank #1 in RCA vs PSI-only?
    print("\n--- Quick summary ---")
    for scenario, target_feature in [("S1", IMPORTANT_FEATURE), ("S2", UNIMPORTANT_FEATURE)]:
        subset = rca_df_all[rca_df_all["scenario"] == scenario]
        rca_top1_hits = (subset[subset["rca_rank"] == 1]["feature"] == target_feature).sum()
        psi_top1_hits = (subset[subset["psi_only_rank"] == 1]["feature"] == target_feature).sum()
        print(f"{scenario}: injected feature = {target_feature}")
        print(f"  RCA ranked it #1 in {rca_top1_hits}/{N_SEEDS} seeds")
        print(f"  PSI-only ranked it #1 in {psi_top1_hits}/{N_SEEDS} seeds")


if __name__ == "__main__":
    run_all_scenarios()