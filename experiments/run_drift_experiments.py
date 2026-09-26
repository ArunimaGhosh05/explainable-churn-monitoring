import sys
import os
sys.path.append("src")

import numpy as np
import pandas as pd
from monitoring.drift_simulator import get_scenario_data
from monitoring.drift_metrics import detect_drift_all_features
from monitoring.rca import compute_rca, top_k_contributors

# ---- CONFIG (using A's real data and real SHAP importance) ----
N_SEEDS = 5  # run each scenario multiple times, as planned

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

IMPORTANT_FEATURE = "MonthlyCharges"      # numeric, high SHAP importance (0.393)
UNIMPORTANT_FEATURE = "SeniorCitizen"     # numeric, low SHAP importance (0.056)

# REAL SHAP importance from A's shap_importance.json
REAL_SHAP_IMPORTANCE = {
    "Contract": 0.9818646311759949,
    "tenure": 0.656414270401001,
    "InternetService": 0.4604584574699402,
    "MonthlyCharges": 0.3925558030605316,
    "TotalCharges": 0.270155668258667,
    "PaymentMethod": 0.24200856685638428,
    "TechSupport": 0.17802315950393677,
    "PaperlessBilling": 0.1404784917831421,
    "OnlineBackup": 0.11156638711690903,
    "OnlineSecurity": 0.10958608239889145,
    "StreamingMovies": 0.09686917066574097,
    "StreamingTV": 0.0679873526096344,
    "MultipleLines": 0.06600596010684967,
    "Dependents": 0.058571573346853256,
    "SeniorCitizen": 0.05641785264015198,
    "PhoneService": 0.05455981194972992,
    "gender": 0.048398979008197784,
    "Partner": 0.04728761687874794,
    "DeviceProtection": 0.02409215271472931
}

os.makedirs("results/tables", exist_ok=True)


def load_real_data():
    """Load A's real reference/current-pool splits."""
    reference = pd.read_csv("data/raw/reference.csv")
    current_pool = pd.read_csv("data/raw/current.csv")
    return reference, current_pool


def run_all_scenarios():
    all_drift_rows = []
    all_rca_rows = []

    reference, current_pool = load_real_data()  # load once, same data used across all seeds/scenarios

    for seed in range(N_SEEDS):
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

            rca_df = compute_rca(drift_results, REAL_SHAP_IMPORTANCE)
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

        # S0 false-alarm rate across all seeds
    print("\n--- S0 false-alarm check ---")
    s0_data = drift_df[drift_df["scenario"] == "S0"]
    for seed in range(N_SEEDS):
        seed_data = s0_data[s0_data["seed"] == seed]
        n_flagged = seed_data["drifted"].sum()
        n_total = len(seed_data)
        print(f"seed {seed}: {n_flagged}/{n_total} features flagged as drifted (ideally ~0)")

    avg_false_alarms = s0_data.groupby("seed")["drifted"].sum().mean()
    print(f"\nAverage false alarms per seed in S0: {avg_false_alarms:.1f} out of {len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)} features")

    # Save a clean summary table for the paper
    summary_rows = []
    for seed in range(N_SEEDS):
        seed_data = s0_data[s0_data["seed"] == seed]
        summary_rows.append({
            "seed": seed,
            "n_flagged": int(seed_data["drifted"].sum()),
            "n_total": len(seed_data),
            "false_alarm_rate": round(seed_data["drifted"].sum() / len(seed_data), 3)
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("results/tables/s0_false_alarm_summary.csv", index=False)
    print(f"\nSaved S0 false-alarm summary to results/tables/s0_false_alarm_summary.csv")

if __name__ == "__main__":
    run_all_scenarios()