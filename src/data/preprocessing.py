import pandas as pd
from sklearn.model_selection import train_test_split

CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


def load_clean_data(path="data/raw/telco.csv"):
    """Clean data but DO NOT one-hot encode — B's drift code needs original column names."""
    df = pd.read_csv(path)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    if not pd.api.types.is_numeric_dtype(df["Churn"]):
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def split_data(df):
    """Split into train / reference / current, keeping original column names."""
    train, temp = train_test_split(df, test_size=0.4, random_state=42, stratify=df["Churn"])
    reference, current = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["Churn"])
    return train, reference, current


def encode_for_model(df):
    """One-hot encode ONLY for XGBoost training — not for B's drift functions."""
    return pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True)


if __name__ == "__main__":
    df = load_clean_data()
    train, reference, current = split_data(df)

    train.to_csv("data/raw/train.csv", index=False)
    reference.to_csv("data/raw/reference.csv", index=False)
    current.to_csv("data/raw/current.csv", index=False)

    print("train:", train.shape)
    print("reference:", reference.shape)
    print("current:", current.shape)
    