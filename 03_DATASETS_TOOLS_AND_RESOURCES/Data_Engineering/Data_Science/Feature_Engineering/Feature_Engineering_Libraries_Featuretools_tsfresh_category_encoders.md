# Feature Engineering Libraries: Featuretools, tsfresh, category_encoders

## Description

**Featuretools** is an open-source Python library for **Automated Feature Engineering (AFE)**. Its core value proposition is to simplify the process of creating features from relational and temporal data, allowing data scientists to focus on modeling. It uses a technique called **Deep Feature Synthesis (DFS)** to automatically generate hundreds of features from multiple data tables, eliminating the need for time-consuming manual coding. The library is also notable for its ability to generate natural language descriptions for the created features, making the process more transparent and interpretable.

## Statistics

**Featuretools:**
*   **GitHub Stars:** 7,562
*   **PyPI Downloads (Overall):** 752,916

**tsfresh:**
*   **GitHub Stars:** 9,011
*   **PyPI Downloads (Overall):** 1,919,582

**category_encoders:**
*   **GitHub Stars:** 2,467
*   **PyPI Downloads (Overall):** 12,615,710

## Features

**Featuretools:** Deep Feature Synthesis (DFS), Support for relational and temporal data (EntitySets), Automatic feature generation, Natural language feature descriptions, Customizable feature primitives, Integration with Dask and Spark for scalability.

**tsfresh:** Extraction of more than 100 time-series features, Automatic selection of relevant features (based on statistical tests), Support for multivariate time series, Integration with pandas DataFrames, Ability to handle time series of different lengths.

**category_encoders:** Implementation of more than 15 categorical encoding methods (including TargetEncoder, BinaryEncoder, HashingEncoder, etc.), scikit-learn-compatible interface (fit/transform), Support for pandas DataFrames, Efficiently handles high-cardinality variables.

## Use Cases

**Featuretools:** Churn Prediction, Fraud detection in financial transactions, Recommendation systems, Equipment failure prediction (Predictive Maintenance) on sensor and log data.

**tsfresh:** Classification and clustering of time series (for example, on sensor data, medical signals such as ECG, or financial data), Vibration-based machine failure analysis, Energy usage pattern analysis.

**category_encoders:** Any Machine Learning problem involving categorical variables (for example, regression and classification models), especially in **high-cardinality** scenarios (many unique values) where One-Hot Encoding becomes inefficient. Common in data science competitions (Kaggle).

## Integration

**Featuretools:**
Installation: `pip install featuretools`
Usage example (DFS):
```python
import featuretools as ft
import pandas as pd

# 1. Create EntitySet
data = {'clientes': pd.DataFrame({'id': [1, 2, 3], 'idade': [30, 40, 50]}),
        'transacoes': pd.DataFrame({'id': [1, 2, 3, 4], 'cliente_id': [1, 1, 2, 3], 'valor': [10, 20, 5, 15]})}
es = ft.EntitySet(id="dados_comerciais")
es = es.add_dataframe(dataframe_name="clientes", dataframe=data['clientes'], index="id")
es = es.add_dataframe(dataframe_name="transacoes", dataframe=data['transacoes'], index="id")
es = es.add_relationship(ft.Relationship(es["clientes"]["id"], es["transacoes"]["cliente_id"]))

# 2. Run Deep Feature Synthesis (DFS)
feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name="clientes", max_depth=2)
print(feature_matrix)
```

**tsfresh:**
Installation: `pip install tsfresh`
Usage example (Feature extraction):
```python
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
import pandas as pd

# Example data (long format)
df_ts = pd.DataFrame({'id': [1, 1, 1, 2, 2], 'tempo': [1, 2, 3, 1, 2], 'valor': [10, 15, 12, 5, 8]})

# Extract features (using a minimal parameter set)
settings = MinimalFCParameters()
features = extract_features(df_ts, column_id="id", column_sort="tempo", column_value="valor", default_fc_parameters=settings)
print(features)
```

**category_encoders:**
Installation: `pip install category-encoders`
Usage example (Target Encoding):
```python
import category_encoders as ce
import pandas as pd

# Example data
data = pd.DataFrame({'cor': ['vermelho', 'azul', 'verde', 'vermelho', 'azul'], 'alvo': [1, 0, 1, 1, 0]})
X = data['cor']
y = data['alvo']

# Apply Target Encoder
encoder = ce.TargetEncoder(cols=['cor'])
X_encoded = encoder.fit_transform(X, y)
print(X_encoded)
```

## URL

Featuretools: https://featuretools.alteryx.com/ | tsfresh: https://tsfresh.readthedocs.io/ | category_encoders: https://contrib.scikit-learn.org/category_encoders/