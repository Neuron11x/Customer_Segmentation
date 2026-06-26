# Customer_Segmentation (Telco Customer Churn Analysis & Prediction)

A machine learning project that analyzes telecom customer data to predict churn and segment customers using Random Forest classification and K-Means clustering.

## Overview

This notebook covers the full data science pipeline — from exploratory data analysis to model tuning and customer segmentation — using a telecom customer churn dataset. The goal is to identify customers likely to churn and understand what drives that behavior.

## Dataset

**File:** `Telco_customer_churn.xlsx`

Key features used in analysis:
- `Tenure Months` — how long the customer has been with the company
- `Monthly Charges` — monthly billing amount
- `Total Charges` — total amount billed
- `Contract` — contract type (Month-to-month, One year, Two year)
- `Internet Service` — type of internet service
- `Payment Method` — how the customer pays
- `Tech Support` — whether the customer has tech support
- `Churn Label` / `Churn Value` — target variable

## Project Structure

### 1. Exploratory Data Analysis (EDA)
- Distribution plots for tenure and monthly charges
- Box plots comparing churned vs. retained customers
- Count plots for categorical features (Contract, Internet Service, Payment Method, Tech Support)
- Correlation matrix across numerical features
- Cross-tabulation of contract type vs. churn rate

### 2. Data Cleaning & Preprocessing
- Converts `Total Charges` to numeric (handles non-numeric values via coercion)
- Fills missing `Total Charges` values with 0 (affects customers with 0 tenure)
- Drops irrelevant columns: `CustomerID`, `Count`, `Country`, `State`, geographic coordinates, `Churn Score`, `CLTV`, `Churn Reason`, `City`
- One-hot encodes all categorical features using `pd.get_dummies`

### 3. Machine Learning — Churn Prediction

Three progressively improved Random Forest models:

| Approach | Description |
|---|---|
| Baseline | `RandomForestClassifier(n_estimators=100)` |
| Balanced | Added `class_weight='balanced'` to handle class imbalance |
| Tuned | `n_estimators=300`, `max_depth=10`, `class_weight='balanced'` |

**Hyperparameter search** — grid search over `n_estimators ∈ [100, 500]` and `max_depth ∈ [5, 25]`, ranked by accuracy and recall.

**Final model** evaluation includes:
- Accuracy score
- Confusion matrix
- Classification report (precision, recall, F1)
- 5-fold cross-validation (accuracy and recall)
- ROC-AUC score and ROC curve

### 4. Feature Importance Analysis
- Extracts feature importances from the tuned model
- Bar plot of all features ranked by importance
- Drops low-importance features (`Phone Service_Yes`, `Multiple Lines_No phone service`) for a leaner model

### 5. Customer Segmentation (K-Means Clustering)
Segments customers using:
- `Tenure Months`, `Monthly Charges`, `Total Charges`, `Churn Probability`

Steps:
1. Standard scaling with `StandardScaler`
2. Elbow method to determine optimal `k` (tested k = 1–15)
3. Final model: `KMeans(n_clusters=3)`

**Resulting segments:**

| Cluster | Label |
|---|---|
| 0 | Budget Loyal Customers |
| 1 | High Risk New Customers |
| 2 | Loyal Premium Customers |

Scatter plots visualize clusters across Tenure, Monthly Charges, Total Charges, and Churn Probability.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
openpyxl
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

## Usage

1. Place `Telco_customer_churn.xlsx` in `/content/` (or update the file path in the notebook).
2. Run all cells in order — the notebook is self-contained.
3. Model outputs (metrics, plots, cluster summaries) are displayed inline.

## Key Findings

- Customers on **month-to-month contracts** churn at significantly higher rates.
- **Higher monthly charges** correlate with increased churn risk.
- **Shorter tenure** is a strong predictor of churn — new customers are the highest risk group.
- The **balanced Random Forest** improves recall on the minority churn class at a small cost to overall accuracy.
- K-Means clustering identifies three actionable customer segments for targeted retention strategies.
