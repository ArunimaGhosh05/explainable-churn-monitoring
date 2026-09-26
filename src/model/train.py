
import sys
sys.path.append("src")

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from data.preprocessing import encode_for_model


def train():
    train_df = pd.read_csv("data/raw/train.csv")
    train_df = encode_for_model(train_df)

    X = train_df.drop(columns=["Churn"])
    y = train_df["Churn"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    model.save_model("artifacts/xgb_model.json")
    X.columns.to_series().to_csv("artifacts/feature_names.csv", index=False)

    print("Model trained and saved.")
    print("Number of features:", X.shape[1])

if __name__ == "__main__":
    train()