from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# -----------------------------
# 1. Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "random_forest_churn_pipeline.pkl"


# -----------------------------
# 2. Load model
# -----------------------------

model = joblib.load(MODEL_PATH)


# -----------------------------
# 3. FastAPI app
# -----------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting telecom customer churn",
    version="1.0.0"
)


# -----------------------------
# 4. Input schema
# -----------------------------

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# -----------------------------
# 5. Health check
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


# -----------------------------
# 6. Prediction endpoint
# -----------------------------

@app.post("/predict")
def predict(data: CustomerData):

    input_data = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return {
        "prediction": int(prediction),
        "churn": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(probability), 4)
    }