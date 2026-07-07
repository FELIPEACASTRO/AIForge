# Inference Optimization

This directory covers optimization of model inference for latency, throughput, cost, reliability, memory, and hardware utilization.

## Scope

- Batching, caching, quantization, compilation, speculative decoding, KV-cache management, graph optimization, and runtime selection.
- CPU, GPU, NPU, edge, serverless, Kubernetes, and high-throughput serving.
- Measurements for p50/p95/p99 latency, tokens/sec, QPS, memory, cold start, quality regression, and cost per request.

## Reference Links

- vLLM documentation: https://docs.vllm.ai/
- vLLM supported models: https://docs.vllm.ai/en/latest/models/supported_models/
- NVIDIA Triton Inference Server: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html
- KServe serving frameworks: https://kserve.github.io/website/docs/model-serving/predictive-inference/frameworks/overview
- ONNX Runtime performance: https://onnxruntime.ai/docs/performance/
- TensorRT-LLM: https://nvidia.github.io/TensorRT-LLM/

## Routing Rules

- Put compression methods in `../Model_Optimization/`.
- Put model-serving architecture in `../Model_Serving/`.
- Put provider/cloud deployment in cloud-platform folders.
