# Data Engineering

This directory covers the data lifecycle that supports machine learning and AI systems: acquisition, cleaning, labeling, validation, feature engineering, versioning, streaming, transformation, storage, and data-quality monitoring.

## Content Map

| Subdirectory | Scope |
|---|---|
| `Data_Sources/` | Source discovery, open data, APIs, scraping inputs, public portals, and provenance. |
| `Data_Preprocessing/` | Cleaning, normalization, missing values, deduplication, parsing, tokenization, and data preparation. |
| `Data_Quality/` | Validation checks, schema constraints, drift, anomaly checks, and quality gates. |
| `Data_Labeling/` and `Data_Annotation/` | Human labeling, active learning, weak supervision, QA, guidelines, and label governance. |
| `Feature_Engineering/` | Tabular, text, image, geospatial, medical, financial, and domain-specific features. |
| `Feature_Stores/` | Feature serving, offline/online stores, Feast, Tecton, and ML feature governance. |
| `Data_Pipelines/`, `ETL/`, and `Data_Transformation/` | Batch and streaming transforms, orchestration, lineage, and reproducible pipelines. |
| `Data_Versioning/` | DVC, LakeFS, Delta Lake, object-store versioning, and dataset release management. |
| `Web_Scraping/` | Scrapy, crawlers, extraction, crawling ethics, robots directives, and legal constraints. |

## Prompt Files

This directory also contains data-workflow prompt files for EDA, cleaning, transformation, forecasting, statistics, topic modeling, sentiment analysis, cohort analysis, and visualization. Keep these prompts close to data engineering because they support analyst and data-science workflows.

## Next Enrichment

Add a data pipeline readiness checklist covering source license, schema, data contracts, quality checks, lineage, reproducibility, and model-evaluation split discipline.
