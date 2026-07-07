# Model Registry Solutions

This directory covers model registries and artifact governance for ML and AI systems.

## Scope

- Model versioning, promotion, lineage, approvals, model cards, deployment linkage, rollback, and governance metadata.
- Track model name, version, stage, artifact URI, training run, dataset, metrics, approval owner, and deployment endpoint.

## Reference Links

- MLflow Model Registry: https://mlflow.org/docs/latest/ml/model-registry
- Azure ML model management: https://learn.microsoft.com/en-us/azure/machine-learning/concept-model-management-and-deployment
- Vertex AI Model Registry: https://cloud.google.com/vertex-ai/docs/model-registry/introduction
- SageMaker Model Registry: https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html

## Routing Rules

- Put experiment tracking in experiment-tracking folders.
- Put deployment details in model-serving/deployment folders.
