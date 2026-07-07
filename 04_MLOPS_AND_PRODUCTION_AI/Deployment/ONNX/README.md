# ONNX Deployment

This directory covers deployment patterns using ONNX and ONNX Runtime.

## Scope

- Model export, runtime optimization, quantization, execution providers, compatibility testing, and cross-platform serving.
- Track source framework, opset, input/output schema, validation parity, execution provider, and latency.

## Reference Links

- ONNX: https://onnx.ai/
- ONNX Runtime: https://onnxruntime.ai/docs/
- ONNX Runtime performance: https://onnxruntime.ai/docs/performance/
- PyTorch ONNX export: https://pytorch.org/docs/stable/onnx.html

## Routing Rules

- Put model compression in model optimization.
- Put model-serving runtime details in deployment model-serving directories.
