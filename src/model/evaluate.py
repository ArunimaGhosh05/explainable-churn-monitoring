import sys
sys.path.append("src")

import pandas as pd
import xgboost as xgb
import json
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from data.preprocessing import encode_for_model


def evaluate():
    model = xgb.XGBClassifier()
    model.load_model("artifacts/xgb_model.json")

    df = pd.read_csv("data/raw/reference.csv")
    df = encode_for_model(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, preds),
        "auc": roc_auc_score(y, probs),
        "f1": f1_score(y, preds),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
    }

    with open("results/tables/baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(metrics)
    return metrics

if __name__ == "__main__":
    evaluate()