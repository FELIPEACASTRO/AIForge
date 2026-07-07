# Healthcare Edge Model Compression

This directory covers compression of healthcare AI models for edge, embedded, mobile, hospital-device, and low-latency clinical settings.

## Content Map

| Subdirectory | Scope |
|---|---|
| `Medical_Imaging/` | Compression of imaging models for ultrasound, radiology, pathology, microscopy, surgical video, and portable screening devices. |

## Core Techniques

- Quantization, pruning, distillation, low-rank adaptation, sparse inference, and runtime compilation.
- Hardware-aware optimization for CPU, mobile GPU, NPU, FPGA, Jetson, medical workstations, and hospital edge servers.
- Clinical safety checks for calibration, sensitivity, specificity, subgroup performance, and boundary-case preservation.

## Source Families

- ONNX Runtime, TensorRT, TensorFlow Lite, OpenVINO, PyTorch quantization, Hugging Face Optimum, and vendor medical-device deployment guides.
- MONAI Deploy and DICOM/PACS deployment references when imaging workflows are involved.

## Reference Links

- ONNX Runtime: https://onnxruntime.ai/docs/
- TensorFlow Lite optimization: https://www.tensorflow.org/lite/performance/model_optimization
- PyTorch quantization: https://docs.pytorch.org/docs/stable/quantization.html
- OpenVINO optimization guide: https://docs.openvino.ai/
- Hugging Face Optimum: https://huggingface.co/docs/optimum/index
- MONAI Deploy App SDK: https://github.com/Project-MONAI/monai-deploy-app-sdk

## Routing Rules

- Put general compression methods in `../../../../04_MLOPS_AND_PRODUCTION_AI/Model_Optimization/`.
- Put segmentation details in `../../Medical_Imaging/Segmentation/`.
- Put edge deployment architecture in healthcare Edge AI or MLOps deployment folders.
