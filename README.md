# Customer Churn Prediction

## Project Overview

Customer churn is one of the most critical challenges faced by subscription-based businesses. Retaining existing customers is significantly more cost-effective than acquiring new ones, making churn prediction a valuable business application of machine learning.

This project develops and evaluates multiple machine learning models to predict customer churn using the Telco Customer Churn dataset. The solution incorporates a complete machine learning pipeline, including data preprocessing, feature engineering, model comparison, hyperparameter tuning, threshold optimization, and model interpretability.

---

## Business Objective

The objective of this project is to identify customers who are likely to discontinue a service so that retention strategies can be applied proactively.

### Business Benefits

* Reduce customer attrition
* Improve customer lifetime value
* Enable targeted retention campaigns
* Optimize marketing and customer support efforts

---

## Dataset

The dataset contains customer demographic information, account details, service subscriptions, and billing information.

### Target Variable

| Variable | Description                                  |
| -------- | -------------------------------------------- |
| Churn    | Whether a customer left the company (Yes/No) |

### Dataset Characteristics

* Records: 7,032 customers
* Features: 19 predictive features
* Target: Binary Classification
* Churn Rate: ~26.6%

---

## Application Screeshots 


---
## Project Workflow

### 1. Data Cleaning

* Converted `TotalCharges` to numeric format
* Removed invalid records containing missing values
* Selected relevant numerical and categorical features

### 2. Data Preprocessing

Implemented a production-ready preprocessing pipeline using:

* Missing value imputation
* Standard scaling for numerical features
* One-Hot Encoding for categorical variables
* ColumnTransformer for automated feature transformation

### 3. Train-Test Split

* Stratified train-test split
* 70% Training Data
* 30% Testing Data

### 4. Model Development

The following models were trained and evaluated:

1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier

### 5. Hyperparameter Tuning

Random Forest was optimized using:

* GridSearchCV
* Cross-validation
* ROC-AUC as the optimization metric

### 6. Model Explainability

Implemented SHAP (SHapley Additive Explanations) to understand feature contributions and improve model interpretability.

---

## Model Performance

| Model                 | ROC-AUC | F1 Score | Precision | Recall |
| --------------------- | ------- | -------- | --------- | ------ |
| Logistic Regression   | 0.86    | 0.64     | 0.52      | 0.84   |
| Random Forest (Tuned) | 0.85    | 0.65     | 0.56      | 0.78   |
| XGBoost               | 0.84    | 0.62     | 0.52      | 0.76   |

---

## Final Model

### Tuned Random Forest Classifier

The tuned Random Forest model was selected as the final model due to its strong balance between Precision, Recall, F1 Score, and interpretability.

### Best Hyperparameters

```python
{
    "max_depth": 10,
    "min_samples_leaf": 5,
    "min_samples_split": 2,
    "n_estimators": 400
}
```

### Final Performance

| Metric    | Score |
| --------- | ----- |
| ROC-AUC   | 0.845 |
| Accuracy  | 0.772 |
| Precision | 0.55  |
| Recall    | 0.73  |
| F1 Score  | 0.63  |

---

## Key Drivers of Churn

Feature importance analysis identified the following major churn indicators:

* Customer Tenure
* Contract Type
* Total Charges
* Monthly Charges
* Internet Service Type
* Online Security
* Technical Support
* Payment Method

SHAP analysis was used to validate and interpret these findings.

---

## Threshold Optimization

Instead of relying solely on the default classification threshold (0.50), multiple thresholds were evaluated to analyze the trade-off between:

* Precision
* Recall
* F1 Score

This enables business stakeholders to select an operating threshold based on organizational priorities.

---

## Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-Learn
* XGBoost

### Model Explainability

* SHAP

### Model Persistence

* Joblib

---

## Project Structure

```text
Customer-Churn-Prediction/
│
├── data/
│   └── Telco_Customer_Churn.csv
│
├── notebooks/
│   └── churn_prediction.ipynb
│
├── artifacts/
│   ├── logistic_model.pkl
│   ├── rf_model.pkl
│   └── xgb_model.pkl
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Key Achievements

* Built an end-to-end machine learning pipeline for customer churn prediction.
* Implemented automated preprocessing using Scikit-Learn Pipelines.
* Compared multiple classification algorithms.
* Performed hyperparameter optimization using GridSearchCV.
* Applied threshold tuning for business-driven decision making.
* Used SHAP for model interpretability and feature impact analysis.
* Saved trained models for future deployment.

---

## Author

**Priyesh Kumar Kashyap**

Machine Learning Engineer | Data Science Enthusiast
