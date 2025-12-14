import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

def load_data(path):
    return pd.read_csv(path)

def train_model(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    model = LogisticRegression()
    model.fit(X, y)
    return model

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    data = load_data("data/dataset.csv")
    model = train_model(data)
    joblib.dump(model, "models/model.pkl")
