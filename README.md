# Customer Segmentation using K-Means Clustering

## Project Overview

This project uses RFM analysis and K-Means clustering to segment customers based on their purchasing behavior.

The goal is to identify groups of customers with different levels of engagement and spending.

## Dataset

The project uses the Online Retail dataset containing transactional data from an online retail business.

Dataset Source:
UCI Machine Learning Repository

## RFM Features

Three customer-level features were created:

- Recency — Number of days since the customer's last purchase
- Frequency — Number of unique purchases made by the customer
- Monetary — Total amount spent by the customer

## Project Workflow

1. Data Loading
2. Data Understanding
3. Data Quality Audit
4. Data Cleaning
5. Exploratory Data Analysis
6. Feature Engineering
7. RFM Feature Engineering
8. Outlier Analysis
9. Log Transformation
10. Feature Scaling
11. K-Means Clustering
12. Elbow Method
13. Silhouette Score
14. Cluster Profiling
15. Customer Segmentation
16. Business Insights

## Machine Learning

- Unsupervised Learning
- K-Means Clustering
- Elbow Method
- Silhouette Score

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Results

The final K-Means model identified two customer segments.

### Cluster 0 — Low-Activity Customers

- Customers: 2,683
- Average Recency: 132.70 days
- Average Frequency: 1.68 purchases
- Average Monetary Value: 498.22

### Cluster 1 — High-Value Customers

- Customers: 1,655
- Average Recency: 24.80 days
- Average Frequency: 8.48 purchases
- Average Monetary Value: 4,562.22

## Conclusion

The project demonstrates how RFM analysis and K-Means clustering can be used to identify meaningful customer segments from transactional data.

The resulting segments provide insights into differences in customer activity, purchase frequency, and spending behavior.

## Project Structure

```text
customer-segmentation-ml/
│
├── data/
├── models/
├── notebooks/
├── outputs/
├── src/
├── .gitignore
├── README.md
└── requirements.txt