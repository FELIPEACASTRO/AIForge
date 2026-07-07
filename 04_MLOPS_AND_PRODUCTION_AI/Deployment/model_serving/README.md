# Deployment Model Serving

This directory covers deployment-specific patterns for model serving.

## Scope

- REST/gRPC serving, batch serving, autoscaling, model format, runtime selection, health checks, rollbacks, and operational ownership.
- Track endpoint contract, model version, runtime, dependency image, scaling policy, latency, throughput, and monitoring.

## Reference Links

- KServe: https://kserve.github.io/website/
- BentoML: https://docs.bentoml.com/
- MLflow model serving: https://mlflow.org/docs/latest/ml/deployment
- NVIDIA Triton: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html

## Routing Rules

- Put low-level performance in `../../Inference_Optimization/`.
- Put Kubernetes-specific deployment in sibling Kubernetes directories.
