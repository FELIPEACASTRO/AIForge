# Medical Imaging Edge Compression

This directory covers compression of medical-imaging models for edge and device-adjacent deployment.

## Scope

- Quantization, pruning, distillation, ONNX export, TensorRT, OpenVINO, LiteRT, and runtime benchmarking for imaging models.
- Track image modality, model architecture, target hardware, latency, memory, calibration, and clinically relevant metric changes.

## Reference Links

- ONNX Runtime: https://onnxruntime.ai/docs/
- TensorRT: https://docs.nvidia.com/deeplearning/tensorrt/
- LiteRT: https://developers.google.com/edge/litert
- OpenVINO: https://docs.openvino.ai/
- MONAI Deploy App SDK: https://github.com/Project-MONAI/monai-deploy-app-sdk

## Routing Rules

- Put general medical edge AI in the parent edge directory.
- Put segmentation-specific risk in medical image segmentation directories.
