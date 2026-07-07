# Data Transformation

This directory covers transformations that reshape, enrich, normalize, or aggregate data for AI systems.

## Scope

- Type conversion, joins, aggregation, normalization, denormalization, enrichment, pivots, feature extraction, and modality conversion.
- Track source, target schema, deterministic logic, validation tests, and downstream assumptions.

## Reference Links

- pandas documentation: https://pandas.pydata.org/docs/
- Polars documentation: https://docs.pola.rs/
- dbt documentation: https://docs.getdbt.com/docs/introduction
- Apache Spark SQL: https://spark.apache.org/docs/latest/sql-programming-guide.html

## Routing Rules

- Put ETL orchestration in ETL and data-pipeline directories.
- Put ML-specific feature transforms in feature-engineering folders.
