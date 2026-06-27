# Orchestration

> Data-pipeline orchestration is the practice of scheduling, sequencing, executing, and monitoring interdependent data tasks (and the assets they produce) across distributed compute, with retries, lineage, and observability built in.

## Why it matters

Modern data and ML stacks are made of dozens of interdependent steps — ingestion, transformation (dbt/Spark), validation, model training, and reverse-ETL — that must run in the right order, recover from failures, and reprocess history on demand. An orchestrator turns this tangle into a versioned, observable, and reproducible system instead of a pile of cron jobs. The field has shifted from purely *task-centric* DAGs (run step A, then B) toward *asset-centric* and *event-driven* models that make data lineage, freshness, and quality first-class concerns.

## Taxonomy

| Approach | Core abstraction | Representative tools | Best for |
|---|---|---|---|
| Task/DAG-centric | Tasks wired into a Directed Acyclic Graph | Airflow, DolphinScheduler | Scheduled batch ETL, broad operator ecosystem |
| Asset-centric | Software-defined assets + lineage graph | Dagster | Data platforms where lineage/quality is primary |
| Pythonic / dynamic flows | Decorated Python functions, dynamic tasks | Prefect, Flyte | Dynamic, parametrized, recovery-heavy workflows |
| Declarative YAML / event-driven | YAML workflow + triggers | Kestra, Argo Workflows | Polyglot teams, event-driven, K8s-native |
| Durable execution (code-as-workflow) | Stateful, fault-tolerant functions | Temporal | Long-running, stateful application workflows |
| Notebook / visual builder | Block-based notebook + UI | Mage | Fast iteration, blended data-science + engineering |

## Key tools

| Tool | What it is | Model | Link |
|---|---|---|---|
| Apache Airflow | Industry-standard Python DAG scheduler; v3 adds DAG versioning, Task Execution API/SDK, asset-aware & event-driven scheduling | Task/DAG | https://github.com/apache/airflow |
| Dagster | Asset-aware orchestrator; software-defined assets, native dbt integration, single lineage graph | Asset | https://github.com/dagster-io/dagster |
| Prefect | Pythonic workflow framework focused on simplicity and failure recovery | Pythonic flows | https://github.com/PrefectHQ/prefect |
| Flyte | Kubernetes-native, strongly-typed workflows for data + ML (originated at Lyft) | Pythonic/typed | https://github.com/flyteorg/flyte |
| Kestra | Declarative YAML, event-driven orchestration platform with rich UI | Declarative YAML | https://github.com/kestra-io/kestra |
| Mage AI | Notebook-style + visual pipeline builder blending code blocks | Notebook/visual | https://github.com/mage-ai/mage-ai |
| Apache DolphinScheduler | Low-code, high-performance distributed DAG platform (Web UI / Python SDK / Open API) | Task/DAG | https://github.com/apache/dolphinscheduler |
| Argo Workflows | Container-native, Kubernetes CRD-based parallel job/DAG engine | Declarative YAML / K8s | https://github.com/argoproj/argo-workflows |
| Temporal | Durable-execution engine for fault-tolerant, long-running stateful workflows-as-code | Durable execution | https://github.com/temporalio/temporal |

## Key concepts & docs

| Concept | Where it lives | Link |
|---|---|---|
| DAGs, scheduling, executors, triggerer | Airflow docs | https://airflow.apache.org/docs/apache-airflow/stable/index.html |
| Software-defined assets, asset checks | Dagster docs | https://docs.dagster.io/ |
| Data-aware / asset-based scheduling | Airflow scheduling guide | https://www.astronomer.io/docs/learn/scheduling-in-airflow |
| Flows, tasks, deployments | Prefect docs | https://docs.prefect.io/ |
| YAML flow definitions, triggers | Kestra docs | https://kestra.io/docs |
| Workflows & DAG templates on K8s | Argo Workflows docs | https://argo-workflows.readthedocs.io/ |

## Key papers

| Year | Title | Link |
|---|---|---|
| 2024 | Reproducible data science over data lakes: replayable data pipelines with Bauplan and Nessie (SIGMOD DEEM) | https://arxiv.org/abs/2404.13682 |
| 2024 | An Empirical Investigation on the Challenges in Scientific Workflow Systems Development | https://arxiv.org/abs/2411.10890 |
| 2025 | Prompt2DAG: A Modular Methodology for LLM-Based Data Enrichment Pipeline Generation | https://arxiv.org/abs/2509.13487 |
| 2025 | DataFlow: An LLM-Driven Framework for Unified Data Preparation and Workflow Automation | https://arxiv.org/abs/2512.16676 |

## Choosing an orchestrator (rules of thumb)

- **Broadest ecosystem / scheduled batch ETL** → Airflow (largest operator/provider catalog, mature community).
- **Data-platform with lineage + dbt at the center** → Dagster (assets unify dbt models and Python pipelines in one graph).
- **Dynamic, recovery-heavy Python pipelines** → Prefect or Flyte (Flyte if Kubernetes + strong typing matter).
- **Polyglot / event-driven / declarative YAML** → Kestra or Argo Workflows.
- **Long-running, stateful application workflows** → Temporal (durable execution, not a batch scheduler).

## Cross-references in AIForge

- [Apache NiFi](../Apache_NiFi/) — flow-based data routing and ingestion within the same pillar.
- [Data Pipelines](../) — parent section covering pipeline patterns and tooling.
- [Data Engineering](../../) — the broader pillar for storage, processing, and quality.
- ML-workflow orchestration (Airflow + Prefect framed for ML) lives under pillar `04` MLOps; this page is the data-engineering DAG/asset home.

## Sources

- https://github.com/apache/airflow
- https://airflow.apache.org/docs/apache-airflow/stable/index.html
- https://dagster.io/blog/dagster-airflow
- https://docs.dagster.io/
- https://github.com/flyteorg/flyte
- https://kestra.io/resources/data/airflow-alternatives
- https://github.com/apache/dolphinscheduler
- https://argoproj.github.io/workflows/
- https://www.pracdata.io/p/state-of-workflow-orchestration-ecosystem-2025
- https://arxiv.org/abs/2404.13682
- https://arxiv.org/abs/2411.10890
- https://arxiv.org/abs/2509.13487

_Expanded from the seed gap-sweep entry. Contributions welcome (see CONTRIBUTING.md)._
