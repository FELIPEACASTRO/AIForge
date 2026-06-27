# Edge and On Device Deployment

> Edge/on-device deployment is the production discipline of running ML models directly on phones, embedded boards, and microcontrollers — covering on-device runtimes, model compilers, hardware accelerators, fleet orchestration, and OTA updates — instead of (or alongside) cloud inference.

## Why it matters

Running inference at the edge cuts latency to single-digit milliseconds, keeps user data on-device for privacy and regulatory compliance, removes per-call cloud cost, and keeps apps working offline. The catch is brutal resource budgets: microcontrollers have ~256 KB SRAM (≈50,000× less memory than a server GPU), phones have thermal/battery limits, and fleets span heterogeneous accelerators (Apple NPU, Qualcomm Hexagon, ARM Ethos, NVIDIA Jetson). This topic spans the full stack — quantization and compilation, the runtimes that execute the model, the silicon, and the MLOps to ship updates to thousands of devices safely.

## Taxonomy

| Layer | What it covers | Representative tech |
|---|---|---|
| Model optimization | Shrink models to fit edge budgets | Quantization (INT8/INT4), pruning, distillation, NAS (TinyNAS) |
| Compilers / IR | Lower & optimize graphs for target HW | Apache TVM, ExecuTorch export, Core ML Tools, XLA |
| On-device runtimes | Execute the model on-device | LiteRT/TFLite, ExecuTorch, ONNX Runtime, llama.cpp, MLC LLM |
| Hardware accelerators | Silicon for edge inference | Jetson/TensorRT, Coral Edge TPU, Hexagon NPU, ARM Ethos-U |
| On-device LLM | LLMs/VLMs on phones & laptops | MLC LLM, llama.cpp (GGUF), MediaPipe LLM, Gemini Nano |
| TinyML / MCU | Models on microcontrollers (<1 MB) | MCUNet/TinyEngine, TFLite Micro, Edge Impulse |
| Edge orchestration | Fleet, OTA, device mgmt at scale | KubeEdge, OpenYurt, Akri, NVIDIA Fleet Command |

## Key frameworks and runtimes

| Tool | Scope | Link |
|---|---|---|
| ExecuTorch (Meta, PyTorch-native) | Phones → MCUs, 50 KB runtime, 12+ backends | https://github.com/pytorch/executorch |
| LiteRT / TensorFlow Lite | Mobile & embedded inference | https://github.com/google-ai-edge/LiteRT |
| ONNX Runtime Mobile | Cross-platform on-device inference | https://onnxruntime.ai/docs/tutorials/mobile/ |
| Apache TVM | Edge compiler + runtime (autotuning) | https://github.com/apache/tvm |
| llama.cpp | C/C++ LLM inference, GGUF format | https://github.com/ggml-org/llama.cpp |
| MLC LLM | Compile LLMs once → iOS/Android/WebGPU | https://github.com/mlc-ai/mlc-llm |
| MediaPipe (LLM Inference API) | On-device vision/LLM pipelines | https://github.com/google-ai-edge/mediapipe |
| Apple Core ML / coremltools | NPU-accelerated iOS/macOS deployment | https://developer.apple.com/documentation/coreml |
| Qualcomm AI Hub | Compile/quantize for Snapdragon NPU | https://github.com/quic/ai-hub-models |
| MCUNet / TinyEngine (MIT) | TinyML NAS + MCU inference engine | https://github.com/mit-han-lab/mcunet |
| TensorFlow Lite Micro | TFLite for microcontrollers | https://github.com/tensorflow/tflite-micro |
| Edge Impulse | End-to-end TinyML dev platform | https://edgeimpulse.com/ |

## Hardware and orchestration

| Tool / platform | Role | Link |
|---|---|---|
| NVIDIA Jetson + TensorRT / DeepStream | Edge GPU inference & vision SDK | https://developer.nvidia.com/embedded/develop/software |
| Google Coral / Edge TPU | Low-power inference accelerator | https://coral.ai/ |
| ARM Ethos-U + CMSIS-NN | MCU/NPU kernels for embedded | https://github.com/ARM-software/CMSIS-NN |
| KubeEdge (CNCF, graduated) | K8s-native edge orchestration + OTA | https://github.com/kubeedge/kubeedge |
| OpenYurt | K8s edge autonomy / node fleet | https://github.com/openyurtio/openyurt |
| Akri (CNCF) | Expose leaf/IoT devices as K8s resources | https://github.com/project-akri/akri |
| tinyML Foundation | Community, benchmarks, events | https://www.tinyml.org/ |

## Benchmarks

| Benchmark | Focus | Link |
|---|---|---|
| MLPerf Tiny | TinyML inference on MCUs | https://mlcommons.org/benchmarks/inference-tiny/ |
| MLPerf Mobile / Inference Edge | Phone & edge accelerator inference | https://mlcommons.org/benchmarks/inference-edge/ |
| AI Benchmark (ETH Zürich) | Smartphone SoC AI performance | https://ai-benchmark.com/ |

## Key papers

| Paper | Year | Link |
|---|---|---|
| ExecuTorch — A Unified PyTorch Solution to Run AI Models On-Device (MLSys) | 2025 | https://arxiv.org/abs/2605.08195 |
| MCUNet: Tiny Deep Learning on IoT Devices (NeurIPS) | 2020 | https://arxiv.org/abs/2007.10319 |
| MCUNetV2: Memory-Efficient Patch-based Inference (NeurIPS) | 2021 | https://arxiv.org/abs/2110.15352 |
| On-Device Training Under 256KB Memory (MCUNetV3, NeurIPS) | 2022 | https://arxiv.org/abs/2206.15472 |
| TVM: An Automated End-to-End Optimizing Compiler for Deep Learning (OSDI) | 2018 | https://arxiv.org/abs/1802.04799 |
| From Tiny Machine Learning to Tiny Deep Learning: A Survey | 2025 | https://arxiv.org/abs/2506.18927 |
| Tiny Machine Learning: Progress and Futures | 2024 | https://arxiv.org/abs/2403.19076 |
| On-Device Language Models: A Comprehensive Review | 2024 | https://arxiv.org/abs/2409.00088 |
| A Systematic Evaluation of On-Device LLMs: Quantization, Performance, Resources | 2025 | https://arxiv.org/abs/2505.15030 |

## Cross-references in AIForge

- [Model Optimization](../Model_Optimization/) — quantization, pruning, distillation that make edge deployment possible
- [Inference Optimization](../Inference_Optimization/) — runtime/serving-level speedups upstream of edge runtimes
- [LLM Inference](../LLM_Inference/) — on-device LLM stacks (llama.cpp, MLC LLM) in depth
- [Deployment](../Deployment/) — cloud-side deployment patterns that complement edge fleets

## Sources

- ExecuTorch repo & paper: https://github.com/pytorch/executorch · https://arxiv.org/abs/2605.08195
- LiteRT / TFLite: https://github.com/google-ai-edge/LiteRT
- MLC LLM: https://github.com/mlc-ai/mlc-llm · llama.cpp: https://github.com/ggml-org/llama.cpp
- MCUNet / TinyEngine (MIT Han Lab): https://github.com/mit-han-lab/mcunet
- KubeEdge (CNCF): https://github.com/kubeedge/kubeedge · Akri: https://github.com/project-akri/akri
- Qualcomm AI Hub: https://github.com/quic/ai-hub-models · Apple Core ML: https://developer.apple.com/documentation/coreml
- TinyML survey: https://arxiv.org/abs/2506.18927 · On-device LLMs review: https://arxiv.org/abs/2409.00088
- MLPerf Tiny / Edge: https://mlcommons.org/benchmarks/inference-tiny/

_Seed section — curated from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
