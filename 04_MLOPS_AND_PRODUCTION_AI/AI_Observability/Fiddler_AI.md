# Fiddler AI

## Description

**Fiddler AI** is a unified **AI Observability** platform that extends beyond traditional Machine Learning (ML) model performance management. The platform provides visibility and actionable insights to monitor, analyze, explain, and protect ML models, LLMs (Large Language Models), and agentic systems in production. Its unique value proposition lies in its ability to deliver **transparency and trust** across the entire AI lifecycle, from evaluation to production, through advanced capabilities such as drift detection, root cause analysis, and explainability (Explainable AI - XAI). Fiddler enables Data Science and MLOps teams to operationalize models at scale, ensuring that performance, fairness, and compliance are continuously maintained. The platform has evolved from Model Performance Management (MPM) into a complete AI Observability solution, covering the new frontier of generative and agentic AI systems.

## Statistics

Fiddler AI monitors a wide range of metrics, which can be categorized into: **ML Performance Metrics:** *Accuracy, Precision, Recall, F1-score, AUC-ROC* (for classification), *RMSE, MAE* (for regression). **Data Quality and Drift Metrics:** *Data Drift*, *Model Drift*, *Feature Importance Drift*. **Business and Operational Metrics:** *Latency, Error Rates, Throughput*. **LLM and Agentic Metrics:** More than 80 out-of-the-box metrics for **LLM Observability** and **Agentic Observability**, including safety, toxicity, and hallucination metrics, as well as the performance of *Guardrails* and *Trust Models*. A notable use case is the **U.S. Navy**, which reduced by 97% the time needed to update its AI model using the Fiddler platform.

## Features

**Unified AI Observability:** Monitoring of ML models, LLMs, and agentic systems on a single platform. **Model Monitoring:** Detection of data drift, model drift, anomalies, latency, and error rates. **Explainable AI (XAI):** Provides insights into the "why" and "how" of model decisions, including root cause analysis. **Guardrails and Trust Models:** Offers more than 80 out-of-the-box metrics and support for custom metrics to ensure AI safety and compliance. **Analytics:** Connects model predictions with business context for actionable insights. **Bias Detection and Fairness:** Tools to mitigate bias and build responsible AI systems. **Support for Complex Models:** Includes monitoring of Natural Language Processing (NLP) and Computer Vision (CV) models.

## Use Cases

**Financial Services:** Ensuring transparent and fair lending and trading by mitigating bias in credit risk models. **Government and Defense:** Such as the **U.S. Navy**, to manage and update AI models in critical environments with high compliance and security. **E-commerce and Retail:** Optimizing the customer experience and extending customer lifetime value (LTV) through monitored recommendation and pricing models. **Healthcare:** Improving patient outcomes with AI observability in diagnostic and treatment models. **MLOps and Data Science:** Providing teams with a unified platform to accelerate the operationalization of ML models at scale, from experimentation to production, ensuring that models remain reliable and compliant. **Agentic Systems and LLMs:** Monitoring, analyzing, and protecting AI agents and LLM applications, ensuring that interactions are safe, accurate, and within usage policies.

## Integration

Fiddler AI offers a robust **Python SDK** for integration with MLOps environments, such as Jupyter Notebooks and automated pipelines. Integration is typically carried out in two main steps: **1. Model Onboarding:** The model is registered on the Fiddler platform, defining its schema (`ModelSpec`), task type (`ModelTask`), and providing a *sample dataframe* for schema inference. **2. Publishing Inference Data:** Production (inference) data is published to Fiddler in real time or in batches for continuous monitoring. The platform integrates natively with popular ML ecosystems, such as **Amazon SageMaker**, **Google Vertex AI**, and **Databricks** (via MLflow).

**Python Code Example (Simplified for Onboarding):**

```python
import fiddler as fdl
import pandas as pd

# 1. Connect to the Fiddler environment (credentials and URL omitted)
# client = fdl.FiddlerClient(url=FIDDLER_URL, org_id=ORG_ID, auth_token=AUTH_TOKEN)

PROJECT_NAME = 'quickstart_example'
MODEL_NAME = 'my_model'

# Create a project
# project = fdl.Project(name=PROJECT_NAME)
# project.create()

# Define the ModelSpec (role of each column)
model_spec = fdl.ModelSpec(
    inputs=['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'],
    outputs=['probability_churned'],
    targets=['Churned'],
    decisions=[],
    metadata=[],
    custom_features=[],
)

# Define the model task (e.g., Binary Classification)
model_task = fdl.ModelTask.BINARY_CLASSIFICATION
task_params = fdl.ModelTaskParams(target_class_order=['no', 'yes'])

# Create a sample DataFrame (sample_df)
# sample_df = pd.read_csv('path/to/sample_data.csv')

# Onboard the model
# model = fdl.Model.from_data(
#     name=MODEL_NAME,
#     project_id=fdl.Project.from_name(PROJECT_NAME).id,
#     source=sample_df,
#     spec=model_spec,
#     task=model_task,
#     task_params=task_params,
#     event_id_col='id_column',
#     event_ts_col='timestamp_column'
# )
```

## URL

https://www.fiddler.ai/