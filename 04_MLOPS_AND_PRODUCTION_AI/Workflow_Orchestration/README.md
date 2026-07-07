# Workflow Orchestration

This directory covers orchestration of data, ML, LLM, agent, and production AI workflows.

## Scope

- DAGs, pipelines, schedules, retries, backfills, lineage, artifacts, dependencies, event triggers, and human approval gates.
- Track owner, trigger, inputs, outputs, idempotency, retries, observability, and rollback path.

## Reference Links

- Apache Airflow: https://airflow.apache.org/docs/
- Dagster: https://docs.dagster.io/
- Prefect: https://docs.prefect.io/
- Kubeflow Pipelines: https://www.kubeflow.org/docs/components/pipelines/overview/
- Flyte: https://docs.flyte.org/

## Routing Rules

- Put data pipeline concepts in the data-engineering pipeline folder.
- Put deployment-specific serving workflows in model-serving or deployment folders.
