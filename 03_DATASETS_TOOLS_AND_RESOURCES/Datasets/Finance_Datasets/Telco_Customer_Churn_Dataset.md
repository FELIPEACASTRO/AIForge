# Telco Customer Churn Dataset

## Description
A dataset focused on customer retention programs, originally from IBM. It contains detailed information about the customers of a fictional telecommunications company, including whether they cancelled their service in the last month (Churn), the services they subscribed to, account information (tenure, contract type, payment method, monthly charges and total charges), and demographic data (gender, age group, and whether they have partners and dependents). It is widely used for churn prediction modeling.

## Statistics
The raw dataset contains 7,043 rows (customers) and 21 columns (features). The main file is `WA_Fn-UseC_Telco-Customer-Churn.csv`, with a size of 977.5 kB. The most recent version on Kaggle is "Version 1". The dataset is widely used, with more than 2.59 million views and 446 thousand downloads.

## Features
The dataset has 21 columns, including: `customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure` (months of tenure), `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, and the target variable `Churn`.

## Use Cases
Customer Churn Prediction in telecommunications companies. Exploratory Data Analysis (EDA) to identify factors that lead to cancellation. Development and comparison of Machine Learning models (such as Logistic Regression, Decision Trees, Random Forest, and XGBoost) for binary classification. Creation of focused customer retention programs.

## Integration
The dataset can be downloaded directly from Kaggle in CSV format. For use in Python, it is common to use the `pandas` library to load the file and begin analysis and modeling. Loading example: `import pandas as pd; df = pd.read_csv('WA_Fn-UseC_Telco-Customer-Churn.csv')`. Preprocessing is frequently required, such as handling missing values and encoding categorical variables, before applying Machine Learning models.

## URL
[https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
