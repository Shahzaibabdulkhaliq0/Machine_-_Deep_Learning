from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Body

app = FastAPI(title="House Price Prediction API")

# Load trained model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "house_price_model.pkl"

model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}


@app.post("/predict")
def predict(data: dict = Body(...)):
    input_data = pd.DataFrame([data])

    prediction = model.predict(input_data)

    return {
        "predicted_price": float(prediction[0])
    }