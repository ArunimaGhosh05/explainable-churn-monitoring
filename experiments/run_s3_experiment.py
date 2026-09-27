import sys
import os
sys.path.append("src")

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from monitoring.drift_simulator import get_scenario_data
from monitoring.drift_metrics import detect_drift_all_features

N_SEEDS = 5

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

os.makedirs("results/tables", exist_ok=True)


def load_model():
    model = xgb.XGBClassifier()
    model.load_model("artifacts/xgb_model.json")
    return model


def encode_like_training(df):
    """Same encoding logic A used for training."""
    return pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True)


def run_s3():
    model = load_model()
    reference = pd.read_csv("data/raw/reference.csv")
    current_pool = pd.read_csv("data/raw/current.csv")

    # baseline: reference performance
    ref_encoded = encode_like_training(reference.drop(columns=["Churn"]))
    ref_encoded = ref_encoded.reindex(columns=model.get_booster().feature_names, fill_value=0)
    ref_pred = model.predict(ref_encoded)
    ref_proba = model.predict_proba(ref_encoded)[:, 1]

    ref_auc = roc_auc_score(reference["Churn"], ref_proba)
    ref_acc = accuracy_score(reference["Churn"], ref_pred)
    ref_f1 = f1_score(reference["Churn"], ref_pred)

    rows = []
    for seed in range(N_SEEDS):
        s3_data = get_scenario_data("S3", current_pool, flip_fraction=0.3, seed=seed)

        # confirm NO feature drift (this is the point of S3)
        drift_results = detect_drift_all_features(
            reference, s3_data.drop(columns=["Churn"]),
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES
        )
        n_feature_drifted = sum(r["drifted"] for r in drift_results)

        # check performance on the label-flipped data
        s3_encoded = encode_like_training(s3_data.drop(columns=["Churn"]))
        s3_encoded = s3_encoded.reindex(columns=model.get_booster().feature_names, fill_value=0)
        s3_pred = model.predict(s3_encoded)
        s3_proba = model.predict_proba(s3_encoded)[:, 1]

        s3_auc = roc_auc_score(s3_data["Churn"], s3_proba)
        s3_acc = accuracy_score(s3_data["Churn"], s3_pred)
        s3_f1 = f1_score(s3_data["Churn"], s3_pred)

        rows.append({
            "seed": seed,
            "n_features_drifted": n_feature_drifted,
            "ref_auc": round(ref_auc, 4),
            "s3_auc": round(s3_auc, 4),
            "auc_drop": round(ref_auc - s3_auc, 4),
            "ref_acc": round(ref_acc, 4),
            "s3_acc": round(s3_acc, 4),
            "ref_f1": round(ref_f1, 4),
            "s3_f1": round(s3_f1, 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv("results/tables/s3_concept_drift_results.csv", index=False)
    print(df)
    print(f"\nAverage features flagged as drifted in S3 (should be near S0's baseline, not elevated): {df['n_features_drifted'].mean():.1f}")
    print(f"Average AUC drop: {df['auc_drop'].mean():.4f}")


if __name__ == "__main__":
    run_s3()