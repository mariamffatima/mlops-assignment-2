import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import joblib
import os

os.makedirs("models", exist_ok=True)

# Load dataset
data = pd.read_csv("data/dataset.csv")
print("Columns:", data.columns.tolist())

# Encode categorical columns
le_location = LabelEncoder()
le_category = LabelEncoder()

data["location_enc"] = le_location.fit_transform(data["location"].astype(str))
data["category_enc"] = le_category.fit_transform(data["category"].astype(str))

# Features and target
X = data[["location_enc", "category_enc", "year"]]
y = pd.to_numeric(data["rank"], errors="coerce").fillna(0)

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "models/model.pkl")
print("Model saved successfully.")
