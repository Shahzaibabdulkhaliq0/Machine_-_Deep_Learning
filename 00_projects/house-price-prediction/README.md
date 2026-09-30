# 🏠 House Price Prediction

An end-to-end Machine Learning regression project that predicts house sale prices using the **Ames Housing Dataset**.

The project covers the complete ML workflow — from data understanding and preprocessing to model comparison, hyperparameter tuning, error analysis, model saving, and FastAPI deployment.

---

## 📌 Project Overview

The objective of this project is to build a regression model that predicts the `SalePrice` of residential properties based on their features.

### 🎯 Problem Type

**Supervised Learning → Regression**

### 🎯 Target Variable

`SalePrice`

---

## 📊 Dataset

**Dataset:** Ames Housing / Kaggle House Prices

| Property | Value |
|---|---:|
| Rows | 1,460 |
| Original Columns | 81 |
| Target | `SalePrice` |
| Duplicate Rows | 0 |

The dataset contains numerical and categorical features describing residential properties, including:

- Overall Quality
- Living Area
- Garage
- Basement
- Lot Area
- Year Built
- Neighborhood
- Bathrooms
- Rooms
- And many other property characteristics

---

## 🔄 Machine Learning Workflow

```text
Data Loading
     ↓
Data Understanding
     ↓
Exploratory Data Analysis
     ↓
Missing Value Handling
     ↓
Train-Test Split
     ↓
Data Preprocessing
     ↓
Model Comparison
     ↓
Cross-Validation
     ↓
Hyperparameter Tuning
     ↓
Final Model Evaluation
     ↓
Error Analysis
     ↓
Feature Importance
     ↓
Model Saving
     ↓
FastAPI Deployment
🧹 Data Preprocessing

The following preprocessing steps were performed:

Handled missing values
Filled categorical feature-absence values with None
Used median imputation for LotFrontage
Filled MasVnrArea missing values with 0
Filled missing Electrical with the mode
Handled missing GarageYrBlt
Removed the Id column
Standardized numerical features
One-Hot Encoded categorical features
Used ColumnTransformer
Used Scikit-learn Pipeline
🤖 Models Compared

The following regression models were evaluated:

Model	MAE	RMSE	R²
Lasso	18,554.33	29,740.65	0.8847
Ridge	19,137.85	29,749.92	0.8846
ElasticNet	20,007.82	36,134.58	0.8298
Random Forest	17,760.91	29,310.36	0.8880
Gradient Boosting	17,446.17	28,228.72	0.8961
🏆 Final Model
Tuned Gradient Boosting Regressor

Hyperparameter tuning was performed using GridSearchCV with 5-fold cross-validation.

Best Parameters
n_estimators = 200
learning_rate = 0.1
max_depth = 3
min_samples_split = 5
min_samples_leaf = 2
Best Cross-Validation RMSE
28,009.71
📈 Final Test Performance

The tuned Gradient Boosting model was evaluated on the unseen test set.

Metric	Score
MAE	16,628.98
RMSE	26,658.63
R²	0.9073
Interpretation

The final model achieved an R² of 0.9073, meaning it explains approximately 90.7% of the variation in house sale prices on the test set.

🔍 Error Analysis

Residual analysis was performed using:

Actual vs Predicted values
Residual calculation
Residual distribution
Residual plot
Largest prediction errors
Inspection of high-error properties

The residual analysis showed that prediction errors were generally centered around zero, with increasing error spread for higher-priced houses.

Some high-error cases were inspected individually. Their feature values were valid and plausible, so no obvious data corruption was identified.

⭐ Feature Importance

The final Gradient Boosting model identified the following features as the most influential:

Feature	Importance
OverallQual	0.4968
GrLivArea	0.1547
GarageCars	0.0405
TotalBsmtSF	0.0377
BsmtFinSF1	0.0364
1stFlrSF	0.0311
2ndFlrSF	0.0262
LotArea	0.0181
YearBuilt	0.0164
BsmtQual_Ex	0.0154

OverallQual was the most influential feature in the trained model.

💾 Model Saving

The complete preprocessing + model pipeline was saved using Joblib:

models/house_price_model.pkl

The saved pipeline was successfully loaded again and tested for prediction.

🚀 FastAPI Deployment

A FastAPI application was created to serve predictions.

Start the API
uvicorn src.api:app --reload
API URL
http://127.0.0.1:8000
Swagger Documentation
http://127.0.0.1:8000/docs

The /predict endpoint accepts the required house features and returns the predicted sale price.

📁 Project Structure
house-price-prediction/
│
├── data/
│   └── raw/
│       └── train.csv
│
├── models/
│   └── house_price_model.pkl
│
├── notebooks/
│   └── 01_data_understanding.ipynb
│
├── src/
│   ├── __init__.py
│   └── api.py
│
├── .gitignore
├── README.md
└── requirements.txt
🛠️ Tech Stack
Technology	Purpose
Python	Programming
Pandas	Data Processing
NumPy	Numerical Computing
Matplotlib	Visualization
Seaborn	Data Visualization
Scikit-learn	Machine Learning
Jupyter Notebook	Experimentation
FastAPI	API Deployment
Uvicorn	API Server
Joblib	Model Saving
Git & GitHub	Version Control
▶️ How to Run
1. Clone the repository
git clone https://github.com/Shahzaibabdulkhaliq0/Machine_-_Deep_Learning.git
2. Navigate to the project
cd Machine_-_Deep_Learning/00_projects/house-price-prediction
3. Install dependencies
pip install -r requirements.txt
4. Start FastAPI
uvicorn src.api:app --reload
5. Open Swagger
http://127.0.0.1:8000/docs
📌 Key Learning Outcomes

Through this project, the following Machine Learning concepts were practiced:

Regression
Exploratory Data Analysis
Missing Value Handling
Feature Preprocessing
One-Hot Encoding
Feature Scaling
Train-Test Split
Model Comparison
Cross-Validation
Hyperparameter Tuning
GridSearchCV
Model Evaluation
Residual Analysis
Feature Importance
Scikit-learn Pipelines
Model Serialization
FastAPI Deployment
👤 Project

House Price Prediction — Machine Learning Regression Project

Built as part of an end-to-end Machine Learning project portfolio.