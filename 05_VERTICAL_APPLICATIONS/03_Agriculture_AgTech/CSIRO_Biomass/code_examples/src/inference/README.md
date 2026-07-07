# Inference Modules

This directory stores inference code for biomass models.

## Scope

- Batch prediction, geospatial tiling, test-time augmentation, ensembling, calibration, export, and submission formatting.
- Inference modules should document input format, model checkpoint, output schema, device assumptions, and post-processing.

## Reference Links

- PyTorch grad modes and inference mode: https://docs.pytorch.org/docs/stable/notes/autograd.html#grad-modes
- ONNX Runtime: https://onnxruntime.ai/docs/
- TensorRT: https://docs.nvidia.com/deeplearning/tensorrt/

## Routing Rules

- Put training loops in `../training/`.
- Put deployment platform notes in the MLOps deployment directories.
- Put benchmark results beside the project or model report.
