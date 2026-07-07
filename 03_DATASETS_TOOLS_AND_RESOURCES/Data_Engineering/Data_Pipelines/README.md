# Data Pipelines

This directory covers repeatable pipelines that move, validate, transform, and publish data for analytics, machine learning, and AI systems.

## Scope

- Batch, streaming, event-driven, and hybrid data pipelines.
- ELT, ETL, reverse ETL, CDC, orchestration, lineage, and dependency management.
- ML-specific pipelines for training data, feature generation, evaluation sets, and inference logs.
- Reproducibility patterns for dataset versioning and rebuildable assets.

## Source Families

- Apache Airflow, Dagster, Prefect, Flyte, Kubeflow Pipelines, TFX, dbt, Spark, Flink, Beam, Kafka, and DVC.
- Cloud-native pipeline systems on AWS, Azure, Google Cloud, Databricks, Snowflake, and Kubernetes.
- Data-quality tools such as Great Expectations, Soda, Deequ, TFDV, and Evidently.

## Reference Links

- Apache Airflow: https://airflow.apache.org/docs/
- Kubeflow Pipelines overview: https://www.kubeflow.org/docs/components/pipelines/overview/
- Kubeflow pipeline concept: https://www.kubeflow.org/docs/components/pipelines/concepts/pipeline/
- TensorFlow Extended: https://www.tensorflow.org/tfx
- Dagster: https://docs.dagster.io/
- DVC: https://dvc.org/doc

## Production Checklist

- Explicit source, schema, owner, and update cadence.
- Idempotent jobs and deterministic rebuilds.
- Data contracts, validation gates, lineage, and rollback.
- Backfill plan, failure handling, observability, and cost controls.
- Privacy, retention, and access boundaries.

## Routing Rules

- Put exploratory analysis in `../Data_Analysis/`.
- Put feature reuse and online/offline serving in `../../../04_MLOPS_AND_PRODUCTION_AI/MLOps_Platforms/feature_store/`.
- Put storage engines in `../../Storage_and_Databases/`.
- Put deployment of model services in `../../../04_MLOPS_AND_PRODUCTION_AI/Deployment/`.
