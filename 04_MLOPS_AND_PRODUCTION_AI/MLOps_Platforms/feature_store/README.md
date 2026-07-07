# Feature Store

This directory covers feature-store systems that define, manage, discover, materialize, and serve features for machine-learning training and inference.

## Content Map

| Subdirectory | Scope |
|---|---|
| `Feature_Pipelines/` | Pipelines that compute, validate, materialize, backfill, and monitor reusable features. |

## What To Track

- Feature definitions, entities, joins, freshness, point-in-time correctness, and training-serving skew.
- Offline store, online store, registry, transformation logic, validation, and access controls.
- Batch and streaming feature computation.
- Feature ownership, documentation, lineage, reuse, and deprecation.

## Source Families

- Feast, Tecton, Vertex AI Feature Store, Databricks Feature Store, SageMaker Feature Store, Hopsworks, and Snowflake feature-store patterns.
- Kubeflow, Airflow, Spark, Flink, dbt, and warehouse-native feature engineering.
- Data validation and drift tools such as Great Expectations, Evidently, TFDV, and whylogs.

## Reference Links

- Feast: https://feast.dev/
- Feast documentation: https://docs.feast.dev/
- Tecton: https://docs.tecton.ai/
- SageMaker Feature Store: https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html
- Databricks Feature Store: https://docs.databricks.com/en/machine-learning/feature-store/index.html

## Routing Rules

- Put theoretical feature engineering in `../../../01_AI_FUNDAMENTALS_AND_THEORY/Feature_Engineering/`.
- Put generic pipeline orchestration in `../../../03_DATASETS_TOOLS_AND_RESOURCES/Data_Engineering/Data_Pipelines/`.
- Put storage-engine notes in `../../../03_DATASETS_TOOLS_AND_RESOURCES/Storage_and_Databases/`.
- Put model-serving notes in `../../Deployment/`.
