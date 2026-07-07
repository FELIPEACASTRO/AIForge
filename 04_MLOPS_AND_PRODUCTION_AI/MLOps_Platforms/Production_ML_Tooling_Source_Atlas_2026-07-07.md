# Production ML Tooling Source Atlas - 2026-07-07

This atlas enriches the MLOps platform directory with source routes for experiment tracking, orchestration, feature stores, serving, monitoring, data quality, model registries, and production AI operations. It is designed to keep tool evidence separate from vendor claims.

## Lifecycle Tooling

| Source | Primary coverage | Local routing |
|---|---|---|
| [MLflow documentation](https://mlflow.org/docs/latest/index.html) | Tracking, projects, models, registry, deployments, and evaluation. | `Experiment_Tracking/`, `Model_Registry_Solutions/`, `Deployment/` |
| [Kubeflow documentation](https://www.kubeflow.org/docs/) | ML pipelines, notebooks, training operators, and Kubernetes-native ML workflows. | `Workflow_Orchestration/`, `GPU_Infrastructure_and_Scheduling/` |
| [KServe documentation](https://kserve.github.io/website/) | Kubernetes model serving, inference graphs, transformers, and predictors. | `Model_Serving/`, `Deployment/Kubernetes/` |
| [BentoML documentation](https://docs.bentoml.com/) | Model packaging, services, runners, deployment, and inference APIs. | `Model_Serving/`, `Deployment/model_serving/` |
| [Seldon documentation](https://docs.seldon.ai/home) | Kubernetes model serving, inference graphs, explainers, and canary patterns. | `Model_Serving/`, `AB_Testing_and_Canary/` |
| [Ray Serve documentation](https://docs.ray.io/en/latest/serve/index.html) | Scalable model serving and Python-native deployments. | `Model_Serving/`, distributed inference. |
| [Feast documentation](https://docs.feast.dev/) | Feature stores, online/offline features, registries, and training-serving consistency. | `feature_store/`, `Feature_Pipelines/` |
| [DVC documentation](https://dvc.org/doc) | Data versioning, pipelines, experiments, model registry, and remotes. | Data versioning, experiment tracking, reproducibility. |
| [lakeFS documentation](https://docs.lakefs.io/) | Data lake version control and branch/merge workflows. | Data versioning and lakehouse governance. |

## Observability, Quality, And Evaluation

| Source | Primary coverage | Local routing |
|---|---|---|
| [Evidently documentation](https://docs.evidentlyai.com/) | Data drift, model monitoring, test suites, and evaluation reports. | `AI_Observability/`, `Drift_Monitoring/` |
| [WhyLabs documentation](https://docs.whylabs.ai/) | Data and ML observability, profiles, monitors, and alerts. | `AI_Observability/`, `Drift_Monitoring/` |
| [Great Expectations documentation](https://docs.greatexpectations.io/docs/) | Data quality tests, expectations, checkpoints, and validation docs. | Data quality and pipeline validation. |
| [Weights and Biases documentation](https://docs.wandb.ai/) | Experiment tracking, sweeps, artifacts, model registry, and reports. | `Experiment_Tracking/`, model registry. |
| [Neptune documentation](https://docs.neptune.ai/) | Experiment tracking, metadata, model registry, and collaboration. | `Experiment_Tracking/`, model registry. |
| [Langfuse documentation](https://langfuse.com/docs) | LLM observability, traces, prompts, evals, and datasets. | `LLMOps_and_Prompt_Management/`, `AI_Observability/` |
| [Promptfoo documentation](https://www.promptfoo.dev/docs/intro/) | Prompt and model evals, red-team checks, CI, and regression tests. | `LLMOps_and_Prompt_Management/`, guardrails. |

## Orchestration And Pipelines

| Source | Primary coverage | Local routing |
|---|---|---|
| [Apache Airflow documentation](https://airflow.apache.org/docs/) | DAG orchestration, scheduling, providers, and production workflow operations. | `Workflow_Orchestration/` |
| [Dagster documentation](https://docs.dagster.io/) | Software-defined assets, data pipelines, orchestration, and lineage. | `Workflow_Orchestration/`, data engineering. |
| [Prefect documentation](https://docs.prefect.io/) | Workflow orchestration, deployments, work pools, and automations. | `Workflow_Orchestration/` |
| [Metaflow documentation](https://docs.metaflow.org/) | ML/data science workflows, artifacts, scaling, and deployment patterns. | `Workflow_Orchestration/`, experiment pipelines. |
| [Flyte documentation](https://docs.flyte.org/) | Workflow automation, typed tasks, scaling, and ML pipelines. | `Workflow_Orchestration/`, distributed training. |
| [Argo Workflows documentation](https://argo-workflows.readthedocs.io/) | Kubernetes-native workflow orchestration. | Kubernetes deployment and pipelines. |

## Tool Evaluation Checklist

| Field | Requirement |
|---|---|
| Lifecycle stage | Data, feature, training, tracking, registry, serving, monitoring, eval, or governance. |
| Integration | Supported frameworks, cloud/Kubernetes dependency, API/CLI/SDK, and storage backends. |
| Evidence | Official docs, example repo, release notes, license, security model, and production references. |
| Operations | Scaling mode, failure handling, audit trail, RBAC, cost controls, and rollback strategy. |
| AIForge route | Exact directory owner and cross-links to data/model/application directories. |
