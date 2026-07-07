# Kubernetes Blue-Green Deployment

This directory covers blue-green release patterns for AI services running on Kubernetes.

## Scope

- Two-environment release, traffic switching, rollback, smoke tests, health checks, data compatibility, and model-version safety.
- Track old/new deployment, traffic switch method, validation gate, rollback trigger, and monitoring.

## Reference Links

- Kubernetes deployments: https://kubernetes.io/docs/concepts/workloads/controllers/deployment/
- Kubernetes services: https://kubernetes.io/docs/concepts/services-networking/service/
- KServe canary rollout docs: https://kserve.github.io/website/docs/model-serving/predictive-inference/rollout-strategies/canary

## Routing Rules

- Put canary-specific releases in the canary deployment folder.
- Put general Kubernetes deployment in the parent Kubernetes directory.
