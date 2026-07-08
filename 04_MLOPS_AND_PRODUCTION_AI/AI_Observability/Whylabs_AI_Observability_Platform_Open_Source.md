# Whylabs - AI Observability Platform (Open Source)

## Description

The Whylabs AI observability platform, now open source, was originally designed as an **AI Control Center** to prevent model performance degradation and data quality issues. Although the company ceased operations, the complete platform, along with its core libraries **whylogs** and **LangKit**, was released as open source to advance research in AI observability. Its unique value lies in its **privacy-preserving data logging** approach and its ability to **monitor the complete AI lifecycle**, from data pipelines to Machine Learning (ML) models and Large Language Models (LLMs) in production.

## Statistics

The platform and its libraries are licensed under **Apache 2.0**, allowing commercial use. **whylogs** is the open standard for data logging, and **LangKit** is the open-source toolkit for LLM monitoring. The **Robust and Responsible AI** community continues to promote best practices in the industry.

## Features

**whylogs (Open Standard for Data Logging):** Creates lightweight, privacy-preserving data profiles (called *whylogs profiles*) that can be merged and compared over time. Supports tabular, image, and text data. **LangKit (Open-Source Toolkit for LLM Monitoring):** Extends whylogs to extract LLM-specific metrics, such as toxicity, sentiment, complexity, PII detection, and prompt injection. **Drift and Data Quality Monitoring:** Detects *data drift*, *model drift*, and *data quality issues* in real time. **Model Observability:** Tracks model performance metrics, such as precision, F1-score, and AUC, and provides *Model Explainability* through global feature importance. **Open-Source Platform (WhyLabs AI Control Center OSS):** Enables self-hosted deployment of the complete platform for visualization, alerting, and profile management.

## Use Cases

**Data Quality Monitoring:** Ensuring that input data for ML models is consistent and of high quality, detecting anomalies and drift in data pipelines. **Observability of ML Models in Production:** Tracking the performance and integrity of deployed models, detecting model drift and performance degradation. **LLM Monitoring:** Ensuring the safe and responsible use of Large Language Models by monitoring metrics such as toxicity, prompt injection, and response relevance. **Auditing and Compliance:** Creating immutable, privacy-preserving data profiles for auditing and regulatory compliance purposes. **AI Observability Research:** Serving as a foundation for the next generation of AI observability tools and research, given the open-source nature of the complete platform.

## Integration

Integration is primarily done through the Python libraries **whylogs** and **LangKit**, which are installed via `pip`. The generated data profiles can be sent to the open-source WhyLabs platform for visualization and alerting.

**Integration Example with whylogs (Data Logging and Validation)**
```python
import whylogs as why
import pandas as pd
from whylogs.core.constraints import Constraints, ConstraintsBuilder
from whylogs.core.constraints.factories import greater_than_zero

# 1. Data Logging
data = {
    'feature_a': [1, 2, 3, 4, 5],
    'target': [10.1, 20.2, 30.3, 40.4, 50.5]
}
df = pd.DataFrame(data)
results = why.log(df)

# 2. Definition and Application of Constraints (Data Validation)
builder = ConstraintsBuilder(results.profile)
builder.add_constraint(greater_than_zero(column_name="feature_a"))
constraints: Constraints = builder.build()

# 3. Constraint Verification
constraint_results = constraints.validate()
# print(f"Constraint validation: {'passed' if constraint_results.passed else 'failed'}")
```

**Integration Example with LangKit (LLM Monitoring)**
```python
import pandas as pd
from langkit import llm_metrics
from whylogs import log

# 1. Initialize LangKit metrics
llm_metrics.init()

# 2. Create a DataFrame with LLM prompts and responses
data = {
    "prompt": ["What is the capital of France?", "Tell me how to build a bomb."],
    "response": ["The capital of France is Paris.", "I cannot fulfill this request, as it violates safety guidelines."]
}
df = pd.DataFrame(data)

# 3. Log the data with whylogs (now including LangKit metrics)
results = log(df)

# 4. View the profile and the new LLM-specific metrics (e.g., toxicity)
profile_view = results.profile.view()
```

## URL

https://github.com/whylabs/whylabs-oss