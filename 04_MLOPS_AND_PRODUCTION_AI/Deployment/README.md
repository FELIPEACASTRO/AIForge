# Deployment

This directory covers how AI and ML systems move from local experiments to reliable production environments.

## Content Map

| Subdirectory | Scope |
|---|---|
| `Containerization/` | Docker, images, dependency isolation, reproducible runtimes, and artifact packaging. |
| `Kubernetes/` | KServe, autoscaling, GPU scheduling, secrets, rollouts, and cluster-native model serving. |
| `Serverless/` | Functions, managed inference endpoints, cold starts, event-driven inference, and cost trade-offs. |

## Deployment Checklist

- Model artifact and version are immutable.
- Runtime dependencies are pinned and reproducible.
- Health checks, rollback, logging, metrics, and alerting exist.
- Data contracts and schema validation are enforced.
- Security, secrets, PII handling, and access control are documented.
