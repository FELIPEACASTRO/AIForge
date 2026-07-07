# Canary Deployment

This directory covers canary release patterns for AI systems.

## Scope

- Partial traffic rollout, guardrail metrics, rollback triggers, cohort selection, shadow comparison, and model-version risk control.
- Track baseline, candidate, traffic percentage, success criteria, observation window, and automated rollback.

## Reference Links

- KServe canary rollout docs: https://kserve.github.io/website/docs/model-serving/predictive-inference/rollout-strategies/canary
- Kubernetes deployments: https://kubernetes.io/docs/concepts/workloads/controllers/deployment/
- Argo Rollouts: https://argo-rollouts.readthedocs.io/

## Routing Rules

- Put A/B experiment design in sibling `AB_Testing/`.
- Put Kubernetes blue-green releases in deployment Kubernetes folders.
