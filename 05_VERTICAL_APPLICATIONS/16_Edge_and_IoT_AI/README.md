# 16 Edge and IoT AI

> Running AI inference (and increasingly training) directly on phones, browsers, microcontrollers, and edge accelerators — under tight memory, power, and latency budgets — instead of in the cloud.

## Why it matters

Edge and on-device AI removes round-trip latency, keeps user data local for privacy, cuts cloud cost, and works offline — making it the default deployment pattern for mobile assistants, wearables, cameras, and industrial IoT. The frontier spans three regimes: **on-device LLMs** (billions of params on phones/laptops via quantization), **TinyML** (kB-scale models on microcontrollers at mW power), and **edge accelerators** (NPUs/TPUs for vision and robotics). Squeezing models into these envelopes drives quantization, compilation, and hardware-aware NAS research.

## Taxonomy

| Sub-area | What it covers | Typical envelope |
|---|---|---|
| **On-device LLMs** | Quantized 1B–8B LLMs on phones, laptops, browsers | 2–16 GB RAM, 4-bit weights |
| **TinyML / MCU ML** | Vision, keyword spotting, anomaly detection on Cortex-M / RISC-V | 32 kB–1 MB RAM, mW power |
| **Mobile / embedded vision** | Detection, segmentation, classification on phones & SBCs | INT8, 1–15 W |
| **Edge accelerators** | NPU/TPU/GPU silicon for fixed-topology inference | 2–60 W, 4–275 TOPS |
| **Compilers & quantization** | Toolchains that lower & compress models per target | offline build step |
| **On-device training / personalization** | Local fine-tune, federated updates | <256 kB memory possible |

## Key on-device LLM runtimes

| Tool | Approach | Link |
|---|---|---|
| llama.cpp | CPU-first GGUF runtime; broadest quant + Android NDK support | https://github.com/ggml-org/llama.cpp |
| MLC LLM | TVM-compiled GPU-first engine (Metal/Vulkan/OpenCL/WebGPU) | https://github.com/mlc-ai/mlc-llm |
| WebLLM | In-browser LLM inference via WebGPU | https://github.com/mlc-ai/web-llm |
| ExecuTorch | PyTorch on-device runtime, ~50 kB core, 12+ HW backends (GA 1.0, 2025) | https://github.com/pytorch/executorch |
| MLX | Apple-silicon array framework for local inference/training | https://github.com/ml-explore/mlx |
| MNN | Alibaba lightweight DL inference engine (mobile/edge) | https://github.com/alibaba/MNN |
| ONNX Runtime + GenAI | Cross-platform inferencing; mobile/edge + LLM extensions | https://github.com/microsoft/onnxruntime-genai |

## Key TinyML & MCU frameworks

| Tool | Role | Link |
|---|---|---|
| LiteRT (ex-TensorFlow Lite) | Google on-device runtime for mobile/edge | https://github.com/google-ai-edge/LiteRT |
| TFLite Micro (TFLM) | TF inference for microcontrollers, no OS/malloc | https://github.com/tensorflow/tflite-micro |
| MCUNet / TinyEngine | Co-designed TinyNAS + lightweight MCU engine (MIT) | https://github.com/mit-han-lab/tinyengine |
| CMSIS-NN | Arm-optimized NN kernels for Cortex-M | https://github.com/ARM-software/CMSIS-NN |
| Edge Impulse | End-to-end TinyML MLOps platform (data→deploy) | https://www.edgeimpulse.com/ |
| Apache TVM | ML compiler stack lowering models to edge targets | https://github.com/apache/tvm |

## Edge accelerators & vendor toolkits

| Platform | Class / typical perf | Link |
|---|---|---|
| NVIDIA Jetson + TensorRT | GPU SoC, INT8/FP16, ~40 TOPS (Orin Nano), 7–60 W | https://developer.nvidia.com/embedded/jetson-modules |
| Google Coral Edge TPU | INT8-only ASIC, ~4 TOPS, 2–4 W | https://coral.ai/ |
| Hailo-8 | Dataflow NPU, ~26 TOPS, low power | https://hailo.ai/products/ai-accelerators/hailo-8-ai-accelerator/ |
| Qualcomm AI Hub | Pre-optimized models + tooling for Snapdragon NPUs | https://aihub.qualcomm.com/ |
| Syntiant NDP | Ultra-low-power always-on speech/sensing NPU | https://www.syntiant.com/ |

## Quantization & compression

| Method | Idea | Link |
|---|---|---|
| GPTQ | Layer-wise post-training quantization, ~3–4 bit | https://arxiv.org/abs/2210.17323 |
| AWQ | Activation-aware weight-only quantization | https://arxiv.org/abs/2306.00978 |
| SmoothQuant | W8A8 PTQ by migrating activation outliers to weights | https://arxiv.org/abs/2211.10438 |
| GGUF / k-quants | Single-file quant format (Q2_K–Q8_0, IQ) used by llama.cpp | https://github.com/ggml-org/ggml |

## Benchmarks & datasets

| Resource | Focus | Link |
|---|---|---|
| MLPerf Tiny | Standard suite for µW–mW MCU inference (accuracy/latency/energy) | https://github.com/mlcommons/tiny |
| MLPerf Mobile / Inference | Smartphone & edge inference benchmarks | https://mlcommons.org/benchmarks/inference-edge/ |
| Wake Vision | Large TinyML person-detection vision dataset | https://arxiv.org/abs/2405.00892 |
| PalmBench | Compressed-LLM benchmark on mobile platforms | https://arxiv.org/abs/2410.05315 |

## Key papers

| Year | Paper | Link |
|---|---|---|
| 2020 | MCUNet: Tiny Deep Learning on IoT Devices | https://arxiv.org/abs/2007.10319 |
| 2021 | MLPerf Tiny Benchmark | https://arxiv.org/abs/2106.07597 |
| 2022 | SmoothQuant: Accurate & Efficient PTQ for LLMs | https://arxiv.org/abs/2211.10438 |
| 2022 | On-Device Training Under 256KB Memory (MCUNetV3) | https://arxiv.org/abs/2206.15472 |
| 2022 | GPTQ: Accurate Post-Training Quantization for GPTs | https://arxiv.org/abs/2210.17323 |
| 2022 | Edge Impulse: An MLOps Platform for TinyML | https://arxiv.org/abs/2212.03332 |
| 2023 | AWQ: Activation-aware Weight Quantization | https://arxiv.org/abs/2306.00978 |
| 2023 | LLM in a Flash: Efficient Inference with Limited Memory (Apple) | https://arxiv.org/abs/2312.11514 |
| 2024 | On-Device Language Models: A Comprehensive Review | https://arxiv.org/abs/2409.00088 |

## Cross-references in AIForge

- [../../02_LLM — model architectures and the LLMs being compressed for the edge
- [../../03_TRAINING_AND_OPTIMIZATION — quantization, distillation, and efficiency techniques
- [../../04_INFRASTRUCTURE_AND_MLOPS — serving, runtimes, and deployment pipelines
- [../15_Robotics_and_Embodied_AI — on-board perception/control sharing edge accelerators

## Sources

- https://github.com/ggml-org/llama.cpp
- https://github.com/mlc-ai/mlc-llm
- https://github.com/pytorch/executorch
- https://github.com/google-ai-edge/LiteRT
- https://github.com/tensorflow/tflite-micro
- https://github.com/mit-han-lab/tinyengine
- https://github.com/mlcommons/tiny
- https://arxiv.org/abs/2007.10319
- https://arxiv.org/abs/2106.07597
- https://arxiv.org/abs/2312.11514
- https://aihub.qualcomm.com/
- https://coral.ai/
- https://hailo.ai/
- https://developer.nvidia.com/embedded/jetson-modules

_Seed section — curated from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
