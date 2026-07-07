# Model Serving

This directory covers production serving of ML, LLM, vision, audio, and multimodal models.

## Scope

- Online inference, batch serving, streaming inference, autoscaling, canary releases, shadow traffic, observability, model registry integration, and rollback.
- Track protocol, runtime, model format, scaling policy, latency, throughput, cost, safety checks, and monitoring.

## Reference Links

- KServe: https://kserve.github.io/website/
- BentoML: https://docs.bentoml.com/
- NVIDIA Triton: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html
- vLLM: https://docs.vllm.ai/
- Seldon Core: https://docs.seldon.ai/

## Routing Rules

- Put low-level performance work in `../Inference_Optimization/`.
- Put deployment environments in `../Deployment/`.
