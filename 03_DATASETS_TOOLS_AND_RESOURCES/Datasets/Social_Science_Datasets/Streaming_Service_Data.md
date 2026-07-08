# Streaming Service Data

## Description
The "Streaming Service Data" dataset is a comprehensive collection of 5,000 customer records from a streaming service, designed for consumer behavior analysis and *churn* prediction. It captures various demographic, behavioral, and transactional aspects of customers, such as age, gender, subscription length, region, payment method, number of support tickets raised, satisfaction score, discounts offered, recent activity, and monthly spend. The main goal of the dataset is to provide valuable *insights* into customer retention and satisfaction patterns.

## Statistics
**Samples:** 5,000 customer records.
**Variables:** 12 columns.
**File Size:** 314.59 kB (Streaming.csv).
**Versions:** The current version is V1, updated about a year ago (information based on the source's update date in 2024).

## Features
**Data Structure:**
*   **Records:** 5,000 customers.
*   **Variables:** 12 columns, including Customer_ID, Age, Gender, Subscription_Length, Region, Payment_Method, Support_Tickets_Raised, Satisfaction_Score, Discount_Offered, Last_Activity, Monthly_Spend, and Churned.
*   **Data Types:** Contains categorical variables (Gender, Region, Payment_Method) and numerical variables (Age, Subscription_Length, Scores, Spend).
*   **Characteristics:** Includes missing values in some variables (Age and Satisfaction_Score), which is useful for practicing data preprocessing. The `Churned` field (1 = yes, 0 = no) is the main target variable for classification models.

## Use Cases
*   **Churn Prediction:** Building Machine Learning models (such as Logistic Regression, Decision Trees, or *Random Forests*) to predict which customers are most likely to cancel their subscription.
*   **Sentiment and Satisfaction Analysis:** Studying the correlation between the Satisfaction Score and other behavioral variables.
*   **Customer Segmentation:** Using *clustering* algorithms to identify groups of customers with similar profiles and spending behaviors.
*   **Marketing Optimization:** Evaluating the effectiveness of the discounts offered and the frequency of customer activity.

## Integration
The dataset can be downloaded directly from the Kaggle platform. After downloading, the `Streaming.csv` file can be loaded into any data analysis environment (such as Python with Pandas or R) for exploration and modeling. Integration is simple, requiring only reading the CSV file.

**Download Example (Kaggle CLI):**
```bash
kaggle datasets download -d akashanandt/streaming-service-data
```
**Usage Example (Python/Pandas):**
```python
import pandas as pd
df = pd.read_csv('Streaming.csv')
print(df.head())
```

## URL
[https://www.kaggle.com/datasets/akashanandt/streaming-service-data](https://www.kaggle.com/datasets/akashanandt/streaming-service-data)
