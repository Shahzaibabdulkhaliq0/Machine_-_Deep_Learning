from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import create_preprocessor


# -----------------------------
# 1. Project paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "models" / "random_forest_churn_pipeline.pkl"


# -----------------------------
# 2. Load data
# -----------------------------

df = pd.read_csv(DATA_PATH)


# -----------------------------
# 3. Data cleaning
# -----------------------------

# Remove customer identifier
df = df.drop(columns="customerID")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove missing values
df = df.dropna()

# Convert target to binary
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})


# -----------------------------
# 4. Separate features/target
# -----------------------------

X = df.drop(columns="Churn")
y = df["Churn"]


# -----------------------------
# 5. Train/Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------
# 6. Create preprocessor
# -----------------------------

preprocessor = create_preprocessor(X_train)


# -----------------------------
# 7. Create final model
# -----------------------------

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# -----------------------------
# 8. Create complete pipeline
# -----------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# -----------------------------
# 9. Train model
# -----------------------------

pipeline.fit(
    X_train,
    y_train
)


# -----------------------------
# 10. Save model
# -----------------------------

joblib.dump(
    pipeline,
    MODEL_PATH
)


print("Model trained and saved successfully.")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Model saved to: {MODEL_PATH}")

print("Shape after cleaning:", df.shape)
print(df.isnull().sum().sum())