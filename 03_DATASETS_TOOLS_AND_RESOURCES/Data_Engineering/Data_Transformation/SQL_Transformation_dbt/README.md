# SQL Transformation dbt

> SQL-first transformation tools (dbt, SQLMesh, Dataform) implement the "T" of modern ELT: they let analytics engineers define data models as versioned SQL `SELECT` statements, then compile, test, document, and orchestrate them into materialized tables and views inside the warehouse.

## Why it matters

dbt pioneered "analytics engineering" by bringing software-engineering practices — modularity, version control, testing, CI/CD, and lineage — to SQL transformation, and is the de-facto standard for the T in ELT. A newer wave (SQLMesh, SDF, Dataform, Malloy) adds semantic SQL understanding via real parsers, enabling column-level lineage, virtual data environments, and far cheaper incremental rebuilds. These tools sit upstream of analytics, BI, and ML feature pipelines, so their modeling choices shape downstream data quality and cost.

## Taxonomy

| Sub-area | What it covers | Representative tools |
|---|---|---|
| Templated SQL transformation | Jinja-templated `SELECT` models compiled to DDL/DML | dbt-core, Dataform |
| Semantic / parser-based transformation | Real SQL AST, column lineage, virtual envs, transpilation | SQLMesh, SDF, Malloy |
| Adapters / engines | Warehouse-specific compilation & execution | dbt-snowflake, dbt-bigquery, dbt-spark, dbt-databricks |
| Testing & data quality | Schema/data tests, expectations, anomaly checks | dbt tests, dbt-expectations, elementary |
| Linting & parsing | Style enforcement, AST parsing, dialect transpiling | SQLFluff, SQLGlot |
| Semantic layer / metrics | Centralized metric definitions over models | dbt Semantic Layer (MetricFlow), Malloy |
| Packages & codegen | Reusable macros, model/source scaffolding | dbt_utils, audit_helper, codegen |

## Key tools

| Tool | Role | Link |
|---|---|---|
| dbt-core | Templated SQL transformation framework (the standard) | https://github.com/dbt-labs/dbt-core |
| dbt Documentation | Official docs, references, guides | https://docs.getdbt.com/ |
| dbt-adapters | v1 warehouse adapters (Snowflake, BigQuery, Spark, Redshift, Postgres) | https://github.com/dbt-labs/dbt-adapters |
| SQLMesh | Parser-based transformation w/ virtual data environments | https://github.com/TobikoData/sqlmesh |
| SQLMesh Docs | Concepts, incremental models, comparisons | https://sqlmesh.readthedocs.io/ |
| Dataform | SQL data-ops framework for BigQuery (now in Google Cloud) | https://github.com/dataform-co/dataform |
| Malloy | Semantic modeling + query language over SQL engines | https://github.com/malloydata/malloy |
| SQLGlot | No-dependency SQL parser, transpiler & optimizer (31+ dialects) | https://github.com/tobymao/sqlglot |
| SQLFluff | Dialect-flexible SQL linter & auto-formatter (Jinja/dbt aware) | https://github.com/sqlfluff/sqlfluff |

## Key adapters (dbt)

| Adapter | Platform | Link |
|---|---|---|
| dbt-snowflake | Snowflake | https://github.com/dbt-labs/dbt-snowflake |
| dbt-bigquery | Google BigQuery | https://github.com/dbt-labs/dbt-bigquery |
| dbt-spark | Apache Spark | https://github.com/dbt-labs/dbt-spark |
| dbt-redshift | Amazon Redshift | https://github.com/dbt-labs/dbt-redshift |
| dbt-postgres | PostgreSQL | https://github.com/dbt-labs/dbt-postgres |
| dbt-databricks | Databricks (Delta) | https://github.com/databricks/dbt-databricks |
| dbt-trino | Trino / Starburst | https://github.com/starburstdata/dbt-trino |
| dbt-duckdb | DuckDB (local/embedded) | https://github.com/duckdb/dbt-duckdb |

Supported platforms reference: https://docs.getdbt.com/docs/supported-data-platforms

## Key packages

| Package | Purpose | Link |
|---|---|---|
| dbt_utils | Cross-database utility macros & generic tests | https://github.com/dbt-labs/dbt-utils |
| dbt-expectations | Great Expectations-style data tests for dbt | https://github.com/calogica/dbt-expectations |
| codegen | Auto-generate models, sources & YAML | https://github.com/dbt-labs/dbt-codegen |
| audit_helper | Compare/audit relations during refactors | https://github.com/dbt-labs/dbt-audit-helper |
| elementary | Data observability & anomaly tests on top of dbt | https://github.com/elementary-data/elementary |
| Package Hub | Discovery registry for dbt packages | https://hub.getdbt.com/ |

## Key papers

| Paper | Year | Venue / ID |
|---|---|---|
| LINEAGEX: A Column Lineage Extraction System for SQL | 2025 | arXiv:2505.23133 — https://arxiv.org/abs/2505.23133 (ICDE 2025 demo) |
| CrackSQL: A Hybrid SQL Dialect Translation System Powered by LLMs | 2025 | arXiv:2504.00882 — https://arxiv.org/abs/2504.00882 |
| RISE: Rule-Driven SQL Dialect Translation via Query Reduction | 2026 | arXiv:2601.05579 — https://arxiv.org/abs/2601.05579 |
| SQLFlex: Dialect-Agnostic SQL Parsing via LLM-Based Segmentation | 2026 | arXiv:2603.16155 — https://arxiv.org/abs/2603.16155 |
| R-Bot: An LLM-based Query Rewrite System | 2024 | arXiv:2412.01661 — https://arxiv.org/abs/2412.01661 |

> Note: SQL transformation tooling (dbt/SQLMesh/Dataform) is largely industry-driven; the academic literature clusters around the adjacent problems these tools solve — column lineage, dialect transpilation, and query rewriting.

## Benchmarks & references

| Resource | Notes | Link |
|---|---|---|
| SQLMesh ↔ dbt comparison | Feature/architecture comparison from the maintainers | https://sqlmesh.readthedocs.io/en/stable/comparisons/ |
| dbt vs SDF — what changes | Context after dbt Labs acquired SDF Labs | https://www.tobikodata.com/blog/dbt-sdf |
| dbt Semantic Layer (MetricFlow) | Metric definitions co-located with models | https://github.com/dbt-labs/metricflow |

## Cross-references in AIForge

- [Data Quality Tools (Great Expectations, Deequ, Soda)](../../Data_Quality/Data_Quality_Tools_Great_Expectations_Deequ_Soda.md) — testing layer downstream of transformation models
- [ETL](../../ETL/) — extract/load stages that feed dbt/SQLMesh models
- [Feature Engineering](../../Feature_Engineering/) — feature pipelines built on transformed tables
- [Data Catalog](../../Data_Catalog/) — lineage & metadata consuming dbt manifests
- [Data Pipelines](../../Data_Pipelines/) — orchestration (Airflow/Dagster/Prefect) that schedules transformation runs

## Sources

- dbt-core — https://github.com/dbt-labs/dbt-core
- dbt docs / supported platforms — https://docs.getdbt.com/docs/supported-data-platforms
- dbt-adapters — https://github.com/dbt-labs/dbt-adapters
- SQLMesh — https://github.com/TobikoData/sqlmesh • docs https://sqlmesh.readthedocs.io/
- SQLMesh comparisons — https://sqlmesh.readthedocs.io/en/stable/comparisons/
- Dataform — https://github.com/dataform-co/dataform
- Malloy — https://github.com/malloydata/malloy
- SQLGlot — https://github.com/tobymao/sqlglot
- SQLFluff — https://github.com/sqlfluff/sqlfluff
- dbt Package Hub — https://hub.getdbt.com/
- LINEAGEX — https://arxiv.org/abs/2505.23133
- CrackSQL — https://arxiv.org/abs/2504.00882

_Expanded from the verified high-value gap seed. Contributions welcome (see CONTRIBUTING.md)._
