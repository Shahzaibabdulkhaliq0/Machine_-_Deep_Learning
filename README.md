# Customer Churn Prediction

## 📌 Project Overview

A machine learning classification project that predicts whether a telecom customer is likely to churn based on customer demographics, services, contract details, and billing information.

The project follows an end-to-end machine learning workflow:

* Data cleaning
* Exploratory Data Analysis (EDA)
* Feature preprocessing
* Model training
* Hyperparameter tuning
* Cross-validation
* Model evaluation
* Error analysis
* Threshold analysis
* Feature importance
* Model serialization

---

## 🎯 Problem Statement

Customer churn is a major business problem for telecom companies.

The goal of this project is to build a binary classification model that predicts:

* `0` → Customer stays
* `1` → Customer churns

---

## 📊 Dataset

**Dataset:** IBM Telco Customer Churn

Original dataset:

* Rows: 7,043
* Columns: 21

After cleaning:

* Rows: 7,032
* Features: 19
* Target: `Churn`

The `customerID` column was removed because it is an identifier and does not provide predictive value.

`TotalCharges` was converted from string to numeric and rows with missing values were removed.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Joblib
* Jupyter Notebook

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
Model Training
     ↓
Hyperparameter Tuning
     ↓
Cross-Validation
     ↓
Model Evaluation
     ↓
Error & Threshold Analysis
     ↓
Final Model
     ↓
Model Serialization
```

---

## ⚙️ Preprocessing

### Numerical Features

Standardized using:

```text
StandardScaler
```

### Categorical Features

Encoded using:

```text
OneHotEncoder
```

with:

```text
handle_unknown="ignore"
drop="first"
```

All preprocessing is included inside the Scikit-learn pipeline to prevent data leakage.

---

## 🤖 Models Evaluated

The following classification models were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost

Hyperparameter tuning was performed using cross-validation.

---

## 📈 Final Model

The final production pipeline uses:

**Random Forest Classifier**

Configuration:

```text
n_estimators = 100
max_depth = 10
min_samples_split = 10
min_samples_leaf = 1
class_weight = balanced
random_state = 42
```

---

## 📊 Test Set Performance

The production training/evaluation pipeline uses:

```text
Training samples: 5,625
Test samples: 1,407
```

Current test performance:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.7925 |
| Precision | 0.5727 |
| Recall    | 0.8636 |
| F1 Score  | 0.6887 |
| ROC-AUC   | 0.8949 |

### Classification Report

| Class   | Precision | Recall |   F1 |
| ------- | --------: | -----: | ---: |
| Stayed  |      0.94 |   0.77 | 0.84 |
| Churned |      0.57 |   0.86 | 0.69 |

The model identifies a high proportion of customers who churn, while producing some false-positive predictions.

---

## 🔍 Error Analysis

Confusion Matrix:

```text
                Predicted
                Stay  Churn

Actual Stay      792    241
Actual Churn      51    323
```

The model correctly identifies **323 churned customers** in the test set and misses **51 churned customers**.

---

## ⭐ Feature Importance

Important features identified by the Random Forest model include:

* Tenure
* Total Charges
* Contract type
* Monthly Charges
* Internet Service
* Payment Method
* Online Security
* Tech Support

Feature importance indicates model contribution and should not be interpreted as causal relationships.

---

## 📁 Project Structure

```text
customer-churn-ml/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── models/
│   └── random_forest_churn_pipeline.pkl
│
├── reports/
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/train.py
```

### 3. Evaluate the model

```bash
python src/evaluate.py
```

---

## 💡 Key Learning Outcomes

This project demonstrates practical experience with:

* Binary classification
* Imbalanced classification
* Data preprocessing pipelines
* One-hot encoding
* Feature scaling
* Model comparison
* Hyperparameter tuning
* Cross-validation
* Evaluation metrics
* Confusion matrix analysis
* Threshold analysis
* Feature importance
* Model serialization with Joblib
* Reproducible ML project structure

---

## 🚀 Future Improvements

* Probability threshold optimization using a validation set
* Model explainability using SHAP
* REST API deployment
* Docker containerization
* Cloud deployment
* Automated model retraining
