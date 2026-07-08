# Evidently AI

## Description

Evidently is an open-source, Python-based observability framework for Machine Learning (ML) and Large Language Models (LLM). Its primary function is to evaluate, test, and monitor the quality and performance of any AI-powered system or data pipeline, spanning from traditional tabular data to Generative AI (Gen AI) applications. Its unique value proposition lies in its modularity, offering more than 100 built-in metrics for drift detection, model evaluation, and data quality testing, making it an essential tool for MLOps.

## Statistics

**GitHub Repository Statistics (evidentlyai/evidently):** Stars: 6.8k, Forks: 740, Commits: 2,715. **Monitoring Metrics:** Supports more than 100 metrics, including PSI, K-L divergence for Data Drift; Accuracy, F1 Score for Classification; MAE, RMSE for Regression; and Sentiment, Toxicity for LLM/Text.

## Features

**Reports:** Computes and summarizes data, ML, and LLM quality evaluations, ideal for exploratory analysis and debugging. Can be exported as JSON, a Python dictionary, HTML, or viewed in the UI. **Test Suites:** Transforms Reports into pass/fail conditions, ideal for regression testing, CI/CD, and data validation. **Monitoring Dashboard (UI):** A service to visualize metrics and test results over time, available for self-hosting (open-source) or via Evidently Cloud. **Extensive Metrics:** Includes more than 100 metrics for Data Drift (PSI, K-L divergence), Data Quality, Classification, Regression, Ranking, and LLM/Text (Sentiment, Toxicity, RAG Relevance).

## Use Cases

**Model Validation:** Ensure that ML and LLM models are ready for production. **Drift Detection:** Identify shifts in data distribution (Data Drift) or in model performance (Model Drift) in production. **Regression Testing:** Use Test Suites in CI/CD pipelines to ensure that new data or model versions do not introduce problems. **LLM Observability:** Monitor the quality of LLM responses, including sentiment, toxicity, and relevance in RAG systems. **Exploratory Analysis:** Use Reports to understand the quality of new datasets or the performance of models in experiments.

## Integration

Evidently is a Python library that integrates easily with other MLOps tools. Installation is done via `pip install evidently`.

**Data Drift Detection Example (Python):**
```python
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn import datasets

# 1. Prepare reference and current data
iris_data = datasets.load_iris(as_frame=True)
iris_frame = iris_data.frame
reference_data = iris_frame.iloc[:60]
current_data = iris_frame.iloc[60:]

# 2. Create and run the Data Drift Report
report = Report([
    DataDriftPreset(method="psi")
])
drift_report = report.run(reference_data=reference_data, current_data=current_data)

# 3. Save the result as HTML
drift_report.save_html("data_drift_report.html")
```
**Integrations with Other Tools:** MLflow, Neptune, Airflow, Kubeflow (to log metrics and reports) and CI/CD pipelines (such as GitHub Actions) for pre-deployment validation.

## URL

https://www.evidentlyai.com/