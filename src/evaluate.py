import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from pathlib import Path


# -----------------------------
# 1. Load data
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "models" / "random_forest_churn_pipeline.pkl"

df = pd.read_csv(DATA_PATH)

# -----------------------------
# 2. Data cleaning
# -----------------------------

df = df.drop(columns="customerID")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna()

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})


# -----------------------------
# 3. Separate features/target
# -----------------------------

X = df.drop(columns="Churn")
y = df["Churn"]


# -----------------------------
# 4. Recreate same test split
# -----------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------
# 5. Load trained model
# -----------------------------

model = joblib.load(MODEL_PATH)

# -----------------------------
# 6. Predictions
# -----------------------------

y_pred = model.predict(X_test)

y_proba = model.predict_proba(X_test)[:, 1]


# -----------------------------
# 7. Metrics
# -----------------------------

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_proba))


# -----------------------------
# 8. Classification report
# -----------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Stayed", "Churned"]
    )
)


# -----------------------------
# 9. Confusion Matrix
# -----------------------------

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)