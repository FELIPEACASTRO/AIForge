# Kubernetes Deployment

This directory covers Kubernetes deployment patterns for AI and ML systems.

## Scope

- Model services, batch jobs, GPU scheduling, autoscaling, canary/blue-green deployment, config, secrets, observability, and rollback.
- Track namespace, image, resources, GPU type, service account, scaling policy, ingress, and deployment strategy.

## Reference Links

- Kubernetes documentation: https://kubernetes.io/docs/home/
- Kubernetes GPU scheduling: https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/
- KServe: https://kserve.github.io/website/
- NVIDIA GPU Operator: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html

## Routing Rules

- Put blue-green examples in `BlueGreen/`.
- Put model-serving details in `../model_serving/`.
