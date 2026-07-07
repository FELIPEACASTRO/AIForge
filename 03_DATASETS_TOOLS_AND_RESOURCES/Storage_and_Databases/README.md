# Storage And Databases

This directory covers storage systems used by AI, ML, analytics, RAG, feature engineering, and production data products.

## Content Map

| Subdirectory | Scope |
|---|---|
| `Vector_Databases/` | Embedding indexes, ANN search, hybrid retrieval, RAG memory, metadata filters, and vector operations. |
| `Data_Warehouses/` | Analytical warehouses, SQL engines, lakehouse systems, and enterprise reporting data. |
| `Data_Lakes/` | Object storage, lakehouse formats, Delta/Iceberg/Hudi patterns, and raw/bronze/silver/gold layouts. |
| `Document_Databases/` | JSON/document stores, operational stores, metadata catalogs, and semi-structured retrieval. |
| `Time_Series_Databases/` | Sensor data, metrics, forecasting, anomaly detection, and temporal feature stores. |
| `In_Memory_Databases/` | Caching, session state, online features, low-latency inference support, and queues. |
| `Distributed_File_Systems/` | HDFS, cloud filesystems, parallel storage, and large-scale ML training data access. |
| `Storage_Infrastructure/` | Object stores, backup, replication, governance, access control, encryption, and retention. |

## AI-Specific Evaluation Questions

- Does the system support the access pattern: training, batch inference, online inference, RAG, or observability?
- Can it preserve provenance, lineage, schema, and dataset versions?
- Does it support privacy, security, tenant isolation, deletion, and audit requirements?
- Does it integrate with the model serving, feature store, and evaluation stack?

## Next Enrichment

Add comparison pages for vector databases, lakehouse formats, RAG storage design, and AI data-governance trade-offs.
