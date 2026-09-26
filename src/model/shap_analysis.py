import sys
sys.path.append("src")

import pandas as pd
import xgboost as xgb
import shap
import json
from data.preprocessing import encode_for_model, CATEGORICAL_FEATURES, NUMERIC_FEATURES


def run_shap():
    model = xgb.XGBClassifier()
    model.load_model("artifacts/xgb_model.json")

    df = pd.read_csv("data/raw/reference.csv")
    df_encoded = encode_for_model(df)
    X = df_encoded.drop(columns=["Churn"])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap_df = pd.DataFrame(shap_values, columns=X.columns)
    shap_df.to_csv("results/tables/shap_raw.csv", index=False)

    # Combine one-hot dummy columns back into original feature names
    combined = {}
    all_original_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    for orig_col in all_original_features:
        matching = [c for c in shap_df.columns if c == orig_col or c.startswith(orig_col + "_")]
        if matching:
            combined[orig_col] = shap_df[matching].abs().sum(axis=1)

    combined_df = pd.DataFrame(combined)
    combined_df.to_csv("results/tables/shap_combined.csv", index=False)

    # Overall importance ranking: mean absolute SHAP value per original feature
    importance = combined_df.abs().mean().sort_values(ascending=False)
    importance.to_csv("results/tables/shap_importance_ranking.csv")

    # Save as JSON dict too — this is the exact format B's compute_rca() needs
    importance_dict = importance.to_dict()
    with open("results/tables/shap_importance.json", "w") as f:
        json.dump(importance_dict, f, indent=2)

    print("Top 10 features by SHAP importance:")
    print(importance.head(10))

    return importance_dict

if __name__ == "__main__":
    run_shap()