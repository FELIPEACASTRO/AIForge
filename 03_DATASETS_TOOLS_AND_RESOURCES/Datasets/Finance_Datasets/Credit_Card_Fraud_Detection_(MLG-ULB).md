# Credit Card Fraud Detection (MLG-ULB)

## Description
This dataset presents credit card transactions made by European cardholders in September 2013. It contains 284,807 transactions, of which 492 are fraudulent. The dataset is highly imbalanced, with the positive class (fraud) representing only 0.172% of all transactions. Due to confidentiality concerns, the features are transformed using Principal Component Analysis (PCA), except for 'Time' (time elapsed since the first transaction) and 'Amount' (transaction value).

## Statistics
284,807 transactions in total. 492 fraud cases (0.172%). 31 columns (features + label). File size: approximately 150 MB. Classic and most-cited version, used in numerous papers from 2017 to 2025. A more recent version (2023) with more than 550,000 records is also available on Kaggle.

## Features
Numeric features transformed by PCA (V1-V28); original 'Time' and 'Amount'. The response variable is 'Class' (1 for fraud, 0 for legitimate). The high class imbalance is notable.

## Use Cases
Development and evaluation of fraud detection models, such as Neural Networks, Support Vector Machines (SVM), and *ensemble* methods (e.g., XGBoost, Random Forest). Research on techniques for handling imbalanced data and cost-sensitive learning. Risk analysis and security of financial transactions.

## Integration
The dataset can be downloaded directly from Kaggle. For use in Python, it is common to use the `pandas` library for loading and `scikit-learn` for preprocessing and modeling. Due to the imbalance, techniques such as *oversampling* (SMOTE) or *undersampling* are frequently applied. A Kaggle account is required to download the `creditcard.csv` file.

## URL
[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
