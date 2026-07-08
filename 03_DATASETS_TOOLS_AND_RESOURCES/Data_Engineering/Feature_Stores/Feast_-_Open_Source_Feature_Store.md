# Feast - Open Source Feature Store

## Description

Feast (Feature Store) is an open-source operational data system designed to manage and serve Machine Learning (ML) features at production scale. It acts as a unified data access layer, reusing existing data infrastructure to ensure feature consistency between training (offline) and serving (online) environments, eliminating the critical problem of 'training-serving skew'. Its unique value proposition lies in providing point-in-time correctness, ensuring that models are trained only with data that would have been available at inference time, and decoupling ML from the underlying data infrastructure [1].

## Statistics

Feast is an active open-source project, initially developed in collaboration between Gojek and Google [2]. Although specific production usage metrics (such as latency or throughput) depend on the underlying infrastructure (for example, Redis, DynamoDB for the online store), Feast's architecture is optimized for low-latency (sub-millisecond) serving for real-time inference [3].

## Features

Feast offers a robust set of features for the ML feature lifecycle:

*   **Python SDK and CLI:** Tools to define, manage, and interact programmatically with features.
*   **Offline and Online Storage:** Manages an offline store (for historical batch training data) and a low-latency online store (for real-time inference serving).
*   **Point-in-Time Correctness:** Battle-tested data join logic to avoid data leakage during the creation of the training dataset.
*   **Feature Reuse and Discovery:** Centralized catalog for feature definition, promoting collaboration between teams.
*   **Feature Server (Optional):** A hosted service for reading and writing feature data, useful for non-Python languages [1].

## Use Cases

Feast is broadly applicable across various ML domains that require consistent and up-to-date features:

*   **Recommendation Engines:** Personalization of online recommendations using historical user/item features and serving features in real time.
*   **Risk Scorecards:** Online fraud detection and credit scoring, using features that compare historical transaction patterns.
*   **NLP/RAG:** Storage and indexing of text embedding vectors for efficient similarity search in Retrieval-Augmented Generation (RAG) systems.
*   **Time Series Forecasting:** Management of temporal features and creation of time-based aggregations for demand forecasting and anomaly detection [4].

## Integration

Integration with Feast involves defining features, materializing data into the online store, and retrieving features for training and inference. The process begins with the initialization of a feature repository and the definition of entities and FeatureViews in a Python file (for example, `example_repo.py`).

**Feature Definition Example (Python):**

```python
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64
from datetime import timedelta

# 1. Define Entity
driver = Entity(name="driver", description="Driver ID", value_type=ValueType.INT64)

# 2. Define Data Source
driver_stats_source = FileSource(
    path="data/driver_stats.parquet",
    timestamp_field="event_timestamp",
)

# 3. Define FeatureView
driver_hourly_stats_fv = FeatureView(
    name="driver_hourly_stats",
    entities=[driver],
    ttl=timedelta(days=1),
    schema=[
        Field(name="conv_rate", dtype=Float32),
        Field(name="acc_rate", dtype=Float32),
        Field(name="avg_daily_trips", dtype=Int64),
    ],
    source=driver_stats_source,
)
```

**Feature Retrieval Example for Inference (Python):**

```python
from feast import FeatureStore

# Connect to the Feature Store
store = FeatureStore(repo_path=".")

# Retrieve online features for real-time inference
feature_vector = store.get_online_features(
    features=[
        "driver_hourly_stats:conv_rate",
        "driver_hourly_stats:acc_rate",
        "driver_hourly_stats:avg_daily_trips",
    ],
    entity_rows=[
        {"driver_id": 1001},
        {"driver_id": 1002},
    ],
).to_dict()

print(feature_vector)
# Expected output (example): {'driver_id': [1001, 1002], 'conv_rate': [0.5, 0.8], ...}
```

## URL

https://feast.dev/