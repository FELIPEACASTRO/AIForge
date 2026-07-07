# Model Optimization

This directory covers methods for making models cheaper, faster, smaller, more reliable, and easier to deploy without losing required quality.

## Content Map

| Subdirectory | Scope |
|---|---|
| `pruning_tools/` | Weight, channel, structured, unstructured, sparsity, lottery-ticket, and pruning automation resources. |

## Core Techniques

- Quantization: FP16, BF16, INT8, INT4, quantization-aware training, and post-training quantization.
- Pruning and sparsity: structured pruning for deployment, unstructured sparsity for compression, and accuracy recovery.
- Distillation: teacher-student training, task distillation, preference distillation, and small-model specialization.
- Compilation and runtime optimization: ONNX Runtime, TensorRT, OpenVINO, TVM, XLA, Torch Compile, and graph fusion.
- Serving optimization: batching, caching, speculative decoding, KV-cache management, and hardware-aware deployment.

## Source Families

- ONNX Runtime optimization and quantization documentation.
- PyTorch quantization, pruning, export, and compile documentation.
- TensorFlow Lite, TensorRT, OpenVINO, Hugging Face Optimum, and vendor accelerator guides.

## Reference Links

- ONNX Runtime documentation: https://onnxruntime.ai/docs/
- ONNX Runtime quantization: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- PyTorch quantization: https://docs.pytorch.org/docs/stable/quantization.html
- TensorFlow Lite optimization: https://www.tensorflow.org/lite/performance/model_optimization
- Hugging Face Optimum: https://huggingface.co/docs/optimum/index

## Routing Rules

- Put edge-healthcare compression work in `../../05_VERTICAL_APPLICATIONS/01_Healthcare_and_Medical_AI/Edge_AI/Model_Compression/`.
- Put deployment patterns in `../Deployment/`.
- Put experiment tracking in `../MLOps_Platforms/`.
- Put benchmark results next to the relevant model family or project showcase.
