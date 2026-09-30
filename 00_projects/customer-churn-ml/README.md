# Customer Churn Prediction

## 📌 Project Overview

An end-to-end machine learning classification project that predicts whether a telecom customer is likely to churn based on customer demographics, services, contract details, and billing information.

The project covers the complete machine learning workflow from data cleaning to model deployment through a REST API.

---

## 🎯 Problem Statement

Customer churn is a major business problem for telecom companies.

The goal is to build a binary classification model that predicts whether a customer will churn.

- `0` → Customer stays
- `1` → Customer churns

---

## 📊 Dataset

**Dataset:** IBM Telco Customer Churn

Original dataset:

- Rows: 7,043
- Columns: 21

After cleaning:

- Rows: 7,032
- Features: 19
- Target: `Churn`

### Data Cleaning

- Removed `customerID`
- Converted `TotalCharges` from string to numeric
- Removed rows with missing `TotalCharges`
- Converted target:
  - `Yes` → `1`
  - `No` → `0`

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Joblib
- FastAPI
- Uvicorn
- Jupyter Notebook

---

## 🔄 Machine Learning Workflow

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Train/Test Split
     ↓
Feature Preprocessing
     ↓
Baseline Modeling
     ↓
Model Comparison
     ↓
Hyperparameter Tuning
     ↓
Cross-Validation
     ↓
Model Evaluation
     ↓
Error Analysis
     ↓
Threshold Analysis
     ↓
Feature Importance
     ↓
Final Model
     ↓
Model Serialization
     ↓
FastAPI Prediction API