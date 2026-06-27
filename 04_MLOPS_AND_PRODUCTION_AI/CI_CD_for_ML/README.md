# CI CD for ML

> Continuous integration and continuous delivery for machine learning (CI/CD/CT): pipeline-as-code that automatically tests, builds, validates, and promotes **code + data + models** through reproducible, gated stages — extending DevOps with data validation, model evaluation, and continuous training (CT).

## Why it matters

Unlike traditional software, ML systems change along three axes — code, data, and model — so a green unit-test suite is not enough to ship safely. CI/CD for ML adds data/schema validation, automated model evaluation against thresholds, and promotion gates so that retraining and redeployment are reproducible, auditable, and reversible. Done well, it shortens the loop from experiment to production while preventing silent regressions, training/serving skew, and model decay. It is the connective tissue between experiment tracking, model registries, orchestration, and serving.

## Taxonomy

| Sub-area | What it covers | Representative tools |
|---|---|---|
| **CI for ML** | Lint/test code, run on PR, reproduce training on a runner, post metric reports/plots back to the PR | CML, GitHub Actions, GitLab CI, Jenkins |
| **Data & model versioning** | Version large datasets/artifacts alongside Git; reproducible pipeline DAGs | DVC, LakeFS, Pachyderm, Git LFS |
| **Continuous Training (CT)** | Automated/triggered retraining pipelines (schedule, drift, new data) | Kubeflow Pipelines, Airflow, Argo Workflows, ZenML |
| **Model evaluation gates** | Threshold checks, eval suites, fairness/regression tests before promotion | MLflow, Evidently, Deepchecks, custom CI steps |
| **Model registry & promotion** | Stage transitions (Staging → Production), approvals, lineage, aliases | MLflow Registry, BentoML, SageMaker / Vertex registries |
| **GitOps for models** | Declarative deploy manifests reconciled to a cluster | Argo CD, Flux, KServe, Seldon |
| **CD / serving rollout** | Package, canary/shadow, rollback of model services | KServe, Seldon Core, BentoML, AB Testing & Canary |

## Key tools

| Tool | Role | Link |
|---|---|---|
| CML (Continuous Machine Learning) | CI for ML — metric/plot reports in PRs, provision cloud runners | https://github.com/iterative/cml |
| DVC | Data & model version control, reproducible pipelines | https://github.com/iterative/dvc |
| GitHub Actions | General-purpose CI/CD runner for ML workflows | https://docs.github.com/en/actions |
| GitLab CI/CD | Integrated CI/CD with ML model registry support | https://docs.gitlab.com/ee/ci/ |
| Jenkins | Extensible automation server for ML pipelines | https://github.com/jenkinsci/jenkins |
| Kubeflow Pipelines | Kubernetes-native ML pipeline DAGs (CI/CD/CT) | https://github.com/kubeflow/pipelines |
| Argo Workflows | CNCF container-native workflow engine | https://github.com/argoproj/argo-workflows |
| Argo CD | GitOps continuous delivery for Kubernetes | https://github.com/argoproj/argo-cd |
| ZenML | Portable MLOps pipelines / CI for ML | https://github.com/zenml-io/zenml |
| MLflow | Experiment tracking + Model Registry / promotion | https://github.com/mlflow/mlflow |
| BentoML | Build, ship, and serve models as production services | https://github.com/bentoml/BentoML |
| Seldon Core | Model deployment & rollout on Kubernetes | https://github.com/SeldonIO/seldon-core |
| KServe | Serverless model inference & canary on K8s | https://github.com/kserve/kserve |
| Pachyderm | Data-driven pipelines with versioned data lineage | https://github.com/pachyderm/pachyderm |

## Validation & gate frameworks

| Tool | Use in the pipeline | Link |
|---|---|---|
| Great Expectations | Data/schema validation gate before training | https://github.com/great-expectations/great_expectations |
| Deepchecks | Data & model validation test suites | https://github.com/deepchecks/deepchecks |
| Evidently | Data/model quality + drift checks in CI | https://github.com/evidentlyai/evidently |
| TFX / TensorFlow Data Validation | Production ML pipeline components & data checks | https://github.com/tensorflow/tfx |

## Reference architecture (CD4ML)

A widely cited blueprint is Thoughtworks' **Continuous Delivery for Machine Learning (CD4ML)**, which decomposes the loop into discoverable, versioned, reproducible stages:

1. **Reproducible model training** — versioned data + code, pipeline-as-code (DVC/Kubeflow).
2. **Model evaluation & experimentation** — tracked metrics, comparison, eval gates (MLflow).
3. **Model serving** — packaged artifact promoted from a registry (BentoML/KServe).
4. **Testing & quality** — data validation, model validation, integration tests in CI (Deepchecks/Evidently).
5. **Orchestration & deployment** — automated, gated rollout (Argo/GitOps), canary/shadow.
6. **Monitoring & continuous training** — drift triggers retraining, closing the loop.

Reference: https://martinfowler.com/articles/cd4ml.html

## Key papers

| Paper | Year | Link |
|---|---|---|
| Machine Learning Operations (MLOps): Overview, Definition, and Architecture | 2022 | https://arxiv.org/abs/2205.02302 |
| On Continuous Integration / Continuous Delivery for Automated Deployment of ML Models using MLOps | 2022 | https://arxiv.org/abs/2202.03541 |
| Machine Learning Operations: A Survey on MLOps Tool Support | 2022 | https://arxiv.org/abs/2202.10169 |
| The Pipeline for the Continuous Development of AI Models — Current State of Research and Practice | 2023 | https://arxiv.org/abs/2301.09001 |
| Automating the Training and Deployment of Models in MLOps by Integrating Systems with ML | 2024 | https://arxiv.org/abs/2405.09819 |
| Hidden Technical Debt in Machine Learning Systems (Sculley et al.) | 2015 | https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html |

## Best practices (skimmable)

- **Version everything** — code in Git, data/models via DVC or a registry; a build must be reproducible from a commit hash.
- **Gate promotion, never click-to-prod** — transition Staging → Production only when an eval threshold passes and an approval/CI check succeeds.
- **Test the three axes** — unit tests for code, schema/distribution checks for data, metric/regression/fairness checks for models.
- **Make CI report metrics** — post evaluation tables and plots back to the PR (CML pattern) so reviewers see model impact, not just diffs.
- **Separate CI (build/test) from CD (deploy)** and from CT (retrain) — different triggers, different gates.
- **Prefer GitOps for deploy** — declarative manifests reconciled by Argo CD/Flux give auditable, reversible rollouts.
- **Guard against skew & decay** — wire drift/monitoring signals as retraining triggers.

## Cross-references in AIForge

- [MLOps Platforms](../MLOps_Platforms/README.md) — end-to-end platforms that bundle CI/CD/CT.
- [Workflow Orchestration — Airflow/Argo/Kubeflow DAGs that pipelines run on.
- [Model Registry Solutions — where promotion gates and stage transitions live.
- [AB Testing and Canary — progressive rollout strategies for deployed models.
- [Drift Monitoring](../Drift_Monitoring/README.md) — signals that trigger continuous training.

## Sources

- CML — https://github.com/iterative/cml
- DVC — https://github.com/iterative/dvc
- Kubeflow Pipelines — https://github.com/kubeflow/pipelines
- ZenML — https://github.com/zenml-io/zenml
- Argo Workflows / Argo CD — https://github.com/argoproj/argo-workflows · https://github.com/argoproj/argo-cd
- MLflow Model Registry — https://mlflow.org/docs/latest/ml/model-registry/
- BentoML — https://github.com/bentoml/BentoML
- Thoughtworks CD4ML — https://martinfowler.com/articles/cd4ml.html
- Databricks CI/CD for ML — https://docs.databricks.com/aws/en/machine-learning/mlops/ci-cd-for-ml
- JFrog: What is CI/CD for ML — https://jfrog.com/learn/mlops/cicd-for-machine-learning/
- arXiv 2205.02302, 2202.03541, 2202.10169, 2301.09001, 2405.09819

_Expanded from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
