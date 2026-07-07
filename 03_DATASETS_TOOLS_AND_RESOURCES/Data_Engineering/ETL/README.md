# ETL

This directory covers extract-transform-load patterns for AI, analytics, and ML workflows.

## Scope

- Source extraction, schema normalization, validation, joins, transformations, loading, backfills, and incremental processing.
- Track source, destination, schedule, schema, owner, data quality checks, and downstream consumers.

## Reference Links

- dbt documentation: https://docs.getdbt.com/docs/introduction
- Apache Airflow: https://airflow.apache.org/docs/
- Dagster: https://docs.dagster.io/
- Apache Spark: https://spark.apache.org/docs/latest/

## Routing Rules

- Put ML-specific ETL in `ML/`.
- Put orchestration in data pipelines or workflow orchestration.
