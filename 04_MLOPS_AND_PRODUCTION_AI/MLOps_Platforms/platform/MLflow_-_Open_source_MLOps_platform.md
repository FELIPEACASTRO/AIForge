# MLflow - Open source MLOps platform

## Description

**MLflow** is an open-source platform designed to manage the complete machine learning (ML) lifecycle, from experimentation to deployment. Its unique value proposition is to provide a **unified solution** for MLOps, covering Experiment Tracking, Code Packaging, Model Management, and a Centralized Model Registry. It has recently expanded to include robust support for **Generative AI (GenAI) applications** and **LLMs (Large Language Models)**, offering tools for LLM evaluation and tracking, consolidating itself as the foundation for MLOps at scale.

## Statistics

**Monthly Downloads:** More than 30 Million (2025 statistic); **GitHub Stars:** More than 19,000; **Contributors:** More than 850 developers; **Notable Enterprise Users:** BNP Paribas, Thales, Unilever France, Carmax, DoorDash, Walmart, Oracle.

## Features

MLflow Tracking (API and UI for logging parameters, metrics, and artifacts); MLflow Projects (standard format for packaging data science code, ensuring reproducibility); MLflow Models (convention for packaging models for deployment on various platforms); MLflow Model Registry (centralized repository for managing models, versions, and transition stages); System Metrics Tracking (logging of CPU, GPU, and memory statistics); LLM Evaluation (simple API for evaluating LLMs with custom metrics).

## Use Cases

**Experiment Tracking:** Comparison and hyperparameter optimization across hundreds of model runs; **MLOps at Scale:** Management of the model lifecycle in production, from training to deployment and monitoring; **Reproducibility:** Ensuring that any team member can reproduce the results of a previous experiment; **Multi-Platform Deployment:** Packaging a model for deployment across various environments (Docker, Azure ML, AWS SageMaker, Databricks); **LLM Applications:** Evaluation and monitoring of language model performance.

## Integration

The primary integration is done through the Python API, compatible with most ML frameworks (Scikit-learn, TensorFlow, PyTorch).

**Integration Example (Python - MLflow Tracking):**
```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Start a new MLflow run
with mlflow.start_run():
    # Define and log parameters
    n_estimators = 100
    mlflow.log_param("n_estimators", n_estimators)

    # Training and metric simulation
    mse = 0.55 # Simulated value
    
    # Log the metric
    mlflow.log_metric("mse", mse)

    # Log the model (uncomment for real use)
    # mlflow.sklearn.log_model(rf, "random_forest_model")
    
    print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
```

## URL

http://mlflow.org/