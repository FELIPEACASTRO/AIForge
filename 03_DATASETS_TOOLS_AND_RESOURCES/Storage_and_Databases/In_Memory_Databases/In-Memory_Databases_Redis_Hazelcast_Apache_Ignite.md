# In-Memory Databases: Redis, Hazelcast, Apache Ignite

## Description

**Redis** is the world's fastest in-memory database, acting as an open-source NoSQL data-structure store. Its unique value proposition lies in its ultra-low latency speed (sub-millisecond) and the versatility of its data structures (Strings, Hashes, Lists, Sets, Sorted Sets, etc.), making it ideal for caching, message queues, and user sessions. **Hazelcast** is an In-Memory Data Grid (IMDG) and In-Memory Computing Platform, focused on providing elastic scalability and high availability for distributed applications. Its value proposition is to transform the RAM of a cluster of servers into a single distributed data pool, enabling parallel processing and distributed transactions. **Apache Ignite** is an In-Memory Computing Platform (IMCP) and a Distributed Database, which stands out for offering HTAP (Hybrid Transactional and Analytical Processing) capabilities and a multi-tier data store (memory and disk). Its value proposition is to act as a data acceleration layer for existing systems (cache) or as a complete distributed database with support for SQL, ACID transactions, and native persistence.

## Statistics

**Redis:** Sub-millisecond (often sub-microsecond) latency for read/write operations. High throughput, reaching hundreds of thousands of operations per second on modern hardware. **Hazelcast:** Microsecond read/write latency. Designed for elastic scalability, supporting clusters with terabytes of in-memory data and high throughput in multi-threaded scenarios. **Apache Ignite:** Microsecond read/write latency. Delivers real-time HTAP (Hybrid Transactional and Analytical Processing) performance, with the ability to scale to petabytes of data using the multi-tier architecture (memory and disk).

## Features

**Redis:** Rich, atomic data structures (Lists, Sets, Hashes, Streams, etc.), optional persistence (RDB/AOF), asynchronous replication, transactions, Pub/Sub, and modules for advanced functionality such as Vector Search and JSON. **Hazelcast:** Distributed data structures (Maps, Queues, Topics, Semaphores), distributed transactions, stream processing with Hazelcast Jet, high availability and fault tolerance, and client support in multiple languages. **Apache Ignite:** Distributed database with SQL support (ANSI-99), ACID transactions, optional native on-disk persistence, Data Grid, Compute Grid (distributed processing), Service Grid, and support for distributed Machine Learning.

## Use Cases

**Redis:** Session caching, message queues, counters and rate limiting, real-time leaderboards, and geospatial data storage. **Hazelcast:** Second-level caching for Hibernate, high-volume transaction processing (e.g., payments), state management for microservices, and real-time event processing. **Apache Ignite:** Data Acceleration Layer for legacy databases, HTAP platform for real-time analytics and transactions, and data storage for distributed Machine Learning applications.

## Integration

**Redis (Python - `redis-py`):**
```python
import redis

# Connection to the Redis server
r = redis.Redis(decode_responses=True)

# SET (Set a value)
r.set('user:100:name', 'Alice')

# GET (Get a value)
name = r.get('user:100:name')
print(f"The name of user 100 is {name}")

# INCR (Increment a counter)
r.incr('page_views')
```

**Hazelcast (Java - `Hazelcast Client`):**
```java
/*
import com.hazelcast.client.HazelcastClient;
import com.hazelcast.client.config.ClientConfig;
import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.map.IMap;

public class HazelcastIntegration {
    public static void main(String[] args) {
        ClientConfig clientConfig = new ClientConfig();
        clientConfig.setClusterName("dev");
        clientConfig.getNetworkConfig().addAddress("127.0.0.1:5701");

        HazelcastInstance client = HazelcastClient.newHazelcastClient(clientConfig);
        IMap<String, String> cities = client.getMap("cities");

        // PUT (Insert a value)
        cities.put("BR", "Brasília");

        // GET (Get a value)
        String capitalBR = cities.get("BR");
        System.out.println("The capital of Brazil is " + capitalBR);

        client.shutdown();
    }
}
*/
```

**Apache Ignite (Python - `pyignite Thin Client`):**
```python
from pyignite import Client

# Connection to the Ignite server
client = Client()
client.connect('127.0.0.1', 10800)

# Get or create a Cache
cache = client.get_or_create_cache('myCache')

# PUT (Insert a value)
cache.put(1, 'Ignite Value 1')

# GET (Get a value)
value1 = cache.get(1)
print(f"The value of key 1 is {value1}")

client.close()
```

## URL

Redis: https://redis.io/ | Hazelcast: https://hazelcast.com/ | Apache Ignite: https://ignite.apache.org/