# Apache Kafka

## Description

An open-source, distributed event streaming platform used by thousands of companies for high-performance data pipelines, streaming analytics, and data integration. Its unique value proposition lies in its ability to provide a low-latency, highly scalable, and fault-tolerant data backbone, acting as an "operating system" for real-time data.

## Statistics

Used by more than 80% of Fortune 100 companies. Key metrics include: Throughput (messages/second), Latency (end-to-end time), Consumer Lag, and Broker Resource Utilization. Topic partitioning is the fundamental unit of scalability.

## Features

High throughput and low latency; Durability and fault tolerance (data replication); Horizontal scalability (broker clusters and topic partitioning); Publish/subscribe messaging model; Stream processing (Kafka Streams and KSQL); Connectors (Kafka Connect) for integration with external systems.

## Use Cases

Building real-time data pipelines; Website activity tracking (Web Tracking); Monitoring application metrics and logs; Event Sourcing and CQRS; Stream Processing for fraud detection or real-time analytics.

## Integration

Integration is typically done through clients in various languages (Java, Python, Node.js, Go) or via Kafka Connect.
**Producer Example in Python (using `confluent-kafka`):**
```python
from confluent_kafka import Producer

conf = {'bootstrap.servers': 'localhost:9092'}
producer = Producer(conf)

def delivery_report(err, msg):
    if err is not None:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to topic {msg.topic()} [{msg.partition()}]')

producer.produce('my_topic', key='key', value='my message', callback=delivery_report)
producer.flush()
```

## URL

https://kafka.apache.org/