import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def train_model(df: pd.DataFrame) -> LogisticRegression:
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def save_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


if __name__ == "__main__":
    data = load_data("data/dataset.csv")
    trained_model = train_model(data)
    save_model(trained_model, "models/model.pkl")
