# Key-Value Stores

This directory covers key-value stores used in AI systems.

## Scope

- Caches, feature lookup, session state, rate-limit state, embeddings metadata, queues, and low-latency serving support.
- Track consistency model, latency, TTL, memory/disk behavior, replication, eviction, and data sensitivity.

## Reference Links

- Redis documentation: https://redis.io/docs/latest/
- KeyDB documentation: https://docs.keydb.dev/
- Amazon DynamoDB: https://docs.aws.amazon.com/amazondynamodb/
- FoundationDB: https://apple.github.io/foundationdb/

## Routing Rules

- Put vector-specific stores in vector database directories.
- Put feature-serving concerns in feature-store folders.
