# LLM Inference Frameworks

State-of-the-art runtimes and servers for **LLM inference at scale** — from datacenter GPUs to laptops and edge.

## Datacenter / GPU Inference Engines

| Framework | Strengths | Link |
|---|---|---|
| **vLLM** | PagedAttention, continuous batching, FP8, multi-LoRA, broad model support | https://github.com/vllm-project/vllm |
| **SGLang** | Structured generation, RadixAttention, fast prefix caching, JSON-mode | https://github.com/sgl-project/sglang |
| **TensorRT-LLM** | NVIDIA-optimized kernels, FP4/FP8, in-flight batching | https://github.com/NVIDIA/TensorRT-LLM |
| **TGI (HF Text Generation Inference)** | Production-grade, multi-tenant, broad model support | https://github.com/huggingface/text-generation-inference |
| **DeepSpeed-MII** | Multi-instance, ZeRO offload | https://github.com/microsoft/DeepSpeed-MII |
| **LMDeploy** (InternLM) | Turbomind kernels, KV-cache quantization | https://github.com/InternLM/lmdeploy |
| **RTP-LLM (Alibaba)** | High-throughput production inference | https://github.com/alibaba/rtp-llm |
| **NIM (NVIDIA Inference Microservices)** | Pre-packaged optimized containers | https://www.nvidia.com/en-us/ai/ |

## Local / Edge Runtimes

| Framework | Strengths | Link |
|---|---|---|
| **llama.cpp** | CPU/Metal/CUDA/Vulkan, GGUF, quantization 1-8 bit | https://github.com/ggerganov/llama.cpp |
| **Ollama** | Easy local serving over llama.cpp | https://ollama.com/ |
| **LM Studio** | Local desktop chat UI | https://lmstudio.ai/ |
| **MLX (Apple)** | Native Apple Silicon | https://github.com/ml-explore/mlx |
| **MLC LLM** | TVM-based, runs on phones / browsers (WebGPU) | https://github.com/mlc-ai/mlc-llm |
| **ExecuTorch (Meta)** | On-device PyTorch | https://github.com/pytorch/executorch |
| **TensorFlow Lite / LiteRT** | Mobile / embedded | https://www.tensorflow.org/lite |
| **WebLLM** | Run LLMs in the browser | https://webllm.mlc.ai/ |
| **Transformers.js (HF)** | Browser ONNX inference | https://huggingface.co/docs/transformers.js |

## Key Techniques

- **PagedAttention** — vLLM paper — https://arxiv.org/abs/2309.06180
- **Continuous Batching** — Orca paper (OSDI 2022)
- **Speculative Decoding** — https://arxiv.org/abs/2211.17192
- **Medusa / EAGLE / EAGLE-2 / EAGLE-3** — multi-token speculative heads
- **Prefix Caching / RadixAttention** — SGLang — https://arxiv.org/abs/2312.07104
- **FP8 / FP4 / INT4 Quantization** — AWQ, GPTQ, GGUF, ExLlamaV2, SmoothQuant
- **KV Cache Compression** — H2O, StreamingLLM, MQA/GQA
- **Disaggregated Prefill / Decode** — Splitwise, DistServe, Mooncake

## Quantization Tools

- **AutoGPTQ / AutoAWQ** — https://github.com/AutoGPTQ/AutoGPTQ
- **GPTQModel** — https://github.com/ModelCloud/GPTQModel
- **bitsandbytes** — 4-bit / 8-bit quantization — https://github.com/bitsandbytes-foundation/bitsandbytes
- **ExLlamaV2** — https://github.com/turboderp/exllamav2
- **llmcompressor (vLLM)** — https://github.com/vllm-project/llm-compressor
- **TorchAO** — https://github.com/pytorch/ao

## Serving Patterns

- **OpenAI-compatible APIs** — vLLM/SGLang/TGI all expose `/v1/chat/completions`
- **KServe / Knative Serving** — Kubernetes-native model serving
- **Triton Inference Server** — multi-framework model serving
- **Ray Serve** — distributed serving
- **BentoML** — model packaging and serving

## Benchmarking Inference

- **MLPerf Inference** — https://mlcommons.org/benchmarks/inference-datacenter/
- **InferBench** — https://github.com/InternLM/InferBench
- **GenAI-Perf (NVIDIA)** — https://github.com/triton-inference-server/perf_analyzer
- **LLMPerf (Ray)** — https://github.com/ray-project/llmperf
