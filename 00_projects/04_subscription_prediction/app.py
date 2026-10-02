from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Load the trained model pipeline
model = joblib.load("bank_subscription_model.pkl")


# Define customer input
class CustomerData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    previously_contacted: int


@app.post("/predict")
def predict(data: CustomerData):

    # Convert input into DataFrame
    customer_data = pd.DataFrame([data.model_dump()])

    # Create the same features used during model training
    customer_data["balance_log"] = (
        np.sign(customer_data["balance"])
        * np.log1p(np.abs(customer_data["balance"]))
    )

    customer_data["campaign_log"] = np.log1p(
        customer_data["campaign"]
    )

    customer_data["age_group"] = pd.cut(
        customer_data["age"],
        bins=[17, 25, 35, 45, 55, 65, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
    )

    customer_data["balance_positive"] = (
        customer_data["balance"] > 0
    ).astype(int)

    # Convert pdays = -1 into missing value
    customer_data["pdays"] = customer_data["pdays"].replace(-1, np.nan)

    # Make prediction
    prediction = model.predict(customer_data)[0]

    # Get subscription probability
    probability = model.predict_proba(customer_data)[0][1]

    return {
        "prediction": int(prediction),
        "subscription_probability": float(probability)
    }