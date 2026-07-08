# California Housing Dataset

## Description
The **California Housing Dataset** is a classic regression dataset widely used in machine learning to predict the median house value in California districts. The data was collected during the 1990 U.S. Census. The goal is to predict the median house value (in units of $100,000) based on eight geographic and socioeconomic features. Although the data is from 1990, it remains a fundamental resource for teaching and testing regression algorithms, and is frequently referenced in recent research and tutorials (2023-2025) as a benchmark.

## Statistics
- **Total Samples (n_samples):** 20,640
- **Dimensionality (n_features):** 8
- **Size:** Approximately 1.5 MB (in CSV format)
- **Versions:** The dataset is static (based on the 1990 census), but is constantly updated and repackaged in new versions of libraries such as scikit-learn (e.g., version 1.7.2).
- **Target Variable (Target):** Median House Value (real 0.15 - 5.0, in units of $100,000).

## Features
The dataset contains 8 predictor attributes (features) for each census block:
1. **MedInc**: Median income in the block (in tens of thousands of dollars).
2. **HouseAge**: Median house age in the block.
3. **AveRooms**: Average number of rooms per household.
4. **AveBedrms**: Average number of bedrooms per household.
5. **Population**: Block population.
6. **AveOccup**: Average household occupancy.
7. **Latitude**: Latitude position of the block.
8. **Longitude**: Longitude position of the block.
The target variable is the **Median House Value**, expressed in units of $100,000.

## Use Cases
- **Predictive Regression:** This is the primary use case, focused on predicting the median house value.
- **Exploratory Data Analysis (EDA):** Used to practice data visualization and understanding techniques.
- **Model Comparison:** Serves as a standard benchmark for comparing the performance of different regression algorithms (e.g., Linear Regression, Random Forests, Gradient Boosting).
- **Feature Engineering:** Used to demonstrate and test the creation of new features from existing ones (e.g., rooms-per-population ratio).
- **Tutorials and Teaching:** Widely used in Machine Learning courses and tutorials due to its ease of access and clean structure.

## Integration
The dataset is most commonly accessed through the **scikit-learn** library in Python, which facilitates its integration into machine learning projects.

**Usage Instructions (Python/scikit-learn):**
1. **Library Installation:** Make sure scikit-learn is installed: `pip install scikit-learn`
2. **Loading the Dataset:** Use the `fetch_california_housing` function to download and load the dataset directly.

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load the dataset
housing = fetch_california_housing(as_frame=True)

# Create a DataFrame for visualization
df = housing.frame

# Display the first rows
print(df.head())

# X (features) and y (target) variables
X = housing.data
y = housing.target
```

The dataset is also available on platforms such as Kaggle and Hugging Face, usually in CSV format.

## URL
[https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
