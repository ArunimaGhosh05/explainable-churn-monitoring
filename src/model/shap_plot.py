import sys
sys.path.append("src")

import pandas as pd
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")  # avoids display issues, just saves to file
import matplotlib.pyplot as plt
from data.preprocessing import encode_for_model


def save_shap_plot():
    model = xgb.XGBClassifier()
    model.load_model("artifacts/xgb_model.json")

    df = pd.read_csv("data/raw/reference.csv")
    df_encoded = encode_for_model(df)
    X = df_encoded.drop(columns=["Churn"])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("artifacts/shap_summary.png", bbox_inches="tight", dpi=150)
    plt.close()

    print("Saved SHAP summary plot to artifacts/shap_summary.png")

if __name__ == "__main__":
    save_shap_plot()