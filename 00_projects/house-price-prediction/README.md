\# House Price Prediction



Machine Learning regression project for predicting house sale prices using the Ames Housing dataset.



\## Project Overview



This project builds an end-to-end house price prediction system using regression techniques.



\### Workflow



1\. Data Loading

2\. Data Understanding

3\. Exploratory Data Analysis

4\. Missing Value Handling

5\. Train-Test Split

6\. Data Preprocessing

7\. Model Comparison

8\. Cross-Validation

9\. Hyperparameter Tuning

10\. Model Evaluation

11\. Error Analysis

12\. Feature Importance

13\. Model Saving

14\. FastAPI Deployment



\## Dataset



\- Dataset: Ames Housing / Kaggle House Prices

\- Rows: 1460

\- Original Columns: 81

\- Target: `SalePrice`



\## Models Tested



\- Ridge Regression

\- Lasso Regression

\- ElasticNet

\- Random Forest Regressor

\- Gradient Boosting Regressor



\## Final Model



Tuned Gradient Boosting Regressor.



\### Test Performance



\- MAE: 16,628.98

\- RMSE: 26,658.63

\- R²: 0.9073



\## Preprocessing



\- Missing values handled

\- Numerical features standardized

\- Categorical features One-Hot Encoded

\- `Id` removed

\- Preprocessing implemented using `ColumnTransformer`

\- Complete workflow implemented using Scikit-learn Pipeline



\## API



FastAPI is used for prediction.



Run:



```bash

uvicorn src.api:app --reload

