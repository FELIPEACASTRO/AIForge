# MLOps Projects

This directory covers complete project examples for production machine learning: data ingestion, training, evaluation, deployment, monitoring, rollback, and governance.

## Project Types

- End-to-end model pipelines.
- Batch scoring systems.
- Real-time inference APIs.
- RAG and agent systems with evaluation and observability.
- Feature-store backed models.
- Drift-aware retraining workflows.
- CI/CD and GitOps examples for ML systems.

## Required Project Metadata

Every project should record problem statement, dataset source, model family, training command, evaluation metric, serving path, monitoring signals, failure modes, cost notes, and reproducibility status.

## Source Families

- MLflow, Kubeflow, TFX, Airflow, Dagster, Flyte, DVC, CML, BentoML, KServe, Seldon, Ray Serve, Evidently, and OpenTelemetry.
- Cloud reference architectures from AWS, Azure, Google Cloud, Databricks, Snowflake, and Kubernetes.

## Reference Links

- MLflow model registry: https://mlflow.org/docs/latest/ml/model-registry/
- MLflow tracking: https://mlflow.org/docs/latest/ml/tracking/
- Kubeflow Pipelines: https://www.kubeflow.org/docs/components/pipelines/overview/
- KServe: https://kserve.github.io/website/
- BentoML: https://docs.bentoml.com/
- Evidently: https://docs.evidentlyai.com/

## Routing Rules

- Put reusable platform notes in `../../../04_MLOPS_AND_PRODUCTION_AI/`.
- Put specific datasets in `../../../03_DATASETS_TOOLS_AND_RESOURCES/Datasets/`.
- Put domain projects in their vertical directories when the domain is the primary value.
