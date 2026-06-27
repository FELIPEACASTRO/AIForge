# Feature Pipelines

> Feature pipelines are the data-engineering layer of a feature store: they read raw data, apply batch / streaming / on-demand transformations, and materialize point-in-time-correct features into offline (training) and online (serving) stores.

## Why it matters

Feature pipelines are where most production-ML bugs are born: training/serving skew, label leakage from non-point-in-time joins, and stale online features all originate in how features are computed and materialized. A well-designed feature pipeline guarantees that the value backfilled for a historical timestamp equals what would have been served online at that instant, and that the same transformation logic runs in training and inference. This is a classic MLOps pillar; the AIForge `feature_store` folder previously covered only Tecton, so this page expands to the open-source and declarative ecosystem (Feast, Chronon, Hopsworks, Featureform).

## Taxonomy

| Transformation type | When it runs | Freshness | Typical engine | Example |
|---|---|---|---|---|
| **Batch** | Scheduled (hourly/daily) over warehouse/lake | Hours–days | Spark, SQL, Snowflake, BigQuery | 30-day rolling purchase count |
| **Streaming** | Continuously on an event stream | Seconds–minutes | Flink, Spark Structured Streaming, Kafka Streams | Clicks in last 5 min |
| **On-demand / request-time** | At inference, using request payload | Real-time | Python (Pandas/native), UDFs | Distance(user_loc, merchant_loc) |
| **Materialization / serving** | Sync offline → online store | N/A | Feature store writers | Push latest aggregates to Redis/DynamoDB |

Core correctness concepts:

- **Point-in-time (as-of) join** — for each labeled row at time `t`, fetch only feature values with event-time `<= t` to prevent future-data leakage.
- **Online/offline consistency** — identical transformation logic and values across the training (offline) and serving (online) stores.
- **Backfill** — recompute historical feature values for new/changed feature definitions over long time ranges.
- **Materialization** — moving computed features from compute into low-latency online storage.

## Key tools / frameworks

| Tool | What it is | Transformations | Link |
|---|---|---|---|
| Feast | Open-source feature store; registry + offline/online stores | Batch, streaming (stream FV), on-demand (ODFV) | https://github.com/feast-dev/feast |
| Chronon (Airbnb/Stripe) | Declarative feature platform; one definition for batch + streaming | GroupBy aggregations, Join, streaming, scalable backfills | https://github.com/airbnb/chronon |
| Hopsworks | Open-source feature store with online (RonDB) + offline stores | Spark/Python pipelines, model-dependent transformations | https://github.com/logicalclocks/hopsworks |
| Featureform | Virtual feature store; defines features on existing infra | SQL/DataFrame transformations, on-demand | https://github.com/featureform/featureform |
| Feathr (LinkedIn) | Feature platform / DSL (now LF AI & Data) | Sliding-window aggregations, derived features | https://github.com/feathr-ai/feathr |
| Tecton | Managed feature platform; Python DSL | Batch, Stream FV + Stream Ingest API, on-demand | https://docs.tecton.ai/ |
| Databricks Feature Engineering | Feature tables on Delta Lake + Unity Catalog | Batch/streaming, point-in-time lookups | https://docs.databricks.com/aws/en/machine-learning/feature-store/ |
| Vertex AI Feature Store | Managed feature serving on BigQuery | BigQuery-backed offline, online serving | https://cloud.google.com/vertex-ai/docs/featurestore |

### Compute / streaming engines commonly wired into pipelines

| Engine | Role | Link |
|---|---|---|
| Apache Flink | Low-latency streaming feature computation | https://github.com/apache/flink |
| Apache Spark (Structured Streaming) | Batch + micro-batch streaming features | https://github.com/apache/spark |
| Apache Kafka | Event transport for streaming features | https://github.com/apache/kafka |
| dbt | SQL-based batch feature transformations in the warehouse | https://github.com/dbt-labs/dbt-core |

## Reference implementations & docs

| Resource | Link |
|---|---|
| Feast — Feature Transformation architecture | https://docs.feast.dev/getting-started/architecture/feature-transformation |
| Feast — Building streaming features (tutorial) | https://docs.feast.dev/tutorials/building-streaming-features |
| Feast — On-demand feature views | https://docs.feast.dev/reference/beta-on-demand-feature-view |
| Chronon — declarative feature engineering (Airbnb Eng blog) | https://medium.com/airbnb-engineering/chronon-a-declarative-feature-engineering-framework-b7b8ce796e04 |
| Tecton — Stream Feature View with Stream Ingest API | https://docs.tecton.ai/docs/0.8/defining-features/feature-views/stream-feature-view/stream-feature-view-with-stream-ingest-api |
| Hopsworks — feature store architecture concepts | https://docs.hopsworks.ai/ |
| Databricks — point-in-time feature lookups | https://www.databricks.com/blog/what-feature-store-complete-guide-ml-feature-engineering |
| featurestore.org — community catalog | https://www.featurestore.org/ |

## Key papers

| Paper | Venue / ID | Link |
|---|---|---|
| The Hopsworks Feature Store for Machine Learning | SIGMOD 2024 (Companion), DOI 10.1145/3626246.3653389 | https://dl.acm.org/doi/10.1145/3626246.3653389 |
| RALF: Accuracy-Aware Scheduling for Feature Store Maintenance | VLDB 2024 (PVLDB Vol.17) | https://www.vldb.org/pvldb/vol17/p563-wooders.pdf |
| Managed Geo-Distributed Feature Store: Architecture and System Design | arXiv:2305.20077 | https://arxiv.org/abs/2305.20077 |
| Machine Learning Operations (MLOps): Overview, Definition, and Architecture | arXiv:2205.02302 | https://arxiv.org/abs/2205.02302 |
| Automated data processing and feature engineering for DL and big data: a survey | arXiv:2403.11395 | https://arxiv.org/abs/2403.11395 |

## Cross-references in AIForge

- [Feature Store overview (Tecton)](../Tecton.md) — managed feature platform sibling page.
- [MLOps Platforms](../../README.md) — parent pillar index.
- [Experiment Tracking](../../Experiment_Tracking/) — pairs feature lineage with run/metric tracking.
- [Observability](../../observability/) — monitoring feature freshness, drift, and training/serving skew.

## Sources

- https://github.com/feast-dev/feast — Feast repository
- https://docs.feast.dev/getting-started/architecture/feature-transformation — Feast transformations
- https://docs.feast.dev/tutorials/building-streaming-features — Feast streaming features
- https://github.com/airbnb/chronon — Chronon repository
- https://medium.com/airbnb-engineering/chronon-a-declarative-feature-engineering-framework-b7b8ce796e04 — Chronon design (Airbnb)
- https://docs.tecton.ai/ — Tecton documentation
- https://www.hopsworks.ai/dictionary/online-offline-feature-store-consistency — online/offline consistency
- https://www.databricks.com/blog/what-feature-store-complete-guide-ml-feature-engineering — Databricks feature store guide
- https://www.vldb.org/pvldb/vol17/p563-wooders.pdf — RALF (VLDB 2024)
- https://arxiv.org/abs/2305.20077 — Managed Geo-Distributed Feature Store
- https://arxiv.org/abs/2205.02302 — MLOps overview/architecture
- https://www.featurestore.org/ — feature store community catalog

_Seed section — curated from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
