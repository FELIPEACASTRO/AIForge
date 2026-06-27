# LLM Inference & Serving — 2026 Radar

The current (mid-2026) state of inference engines and where each wins.

## Landscape

Main frameworks in 2026: **vLLM, SGLang, TensorRT-LLM, LMDeploy, oMLX, Ollama, MLC LLM**.

| Engine | Sweet spot (2026) |
|---|---|
| **vLLM** | Cloud champion; best memory efficiency + hardware flexibility. HuggingFace Inference Endpoints now default to vLLM. |
| **SGLang** | Dominates **agent** scenarios via **RadixAttention** (eliminates redundant compute on shared prefixes) — best for RAG pipelines and multi-turn agent loops. ~16,200 tok/s vs vLLM ~12,500 tok/s on smaller models (≈29% throughput edge in that test). |
| **TensorRT-LLM** | Peak NVIDIA-optimized latency/throughput on H100-class hardware (FP8/FP4). |
| **oMLX** | Rising in the **Mac / Apple Silicon** ecosystem. |
| **Ollama / llama.cpp / MLC LLM** | Local & edge serving; GGUF quantization; phones/browsers (WebGPU). |
| **TGI (HuggingFace)** | As of Dec 2025: **bug-fixes only**, no new features — superseded by vLLM/SGLang as defaults. |

## Decision guide (2026)

- **Agents / RAG / repeated system prompts** → SGLang (RadixAttention prefix reuse).
- **General cloud serving / memory-constrained / multi-hardware** → vLLM.
- **Max NVIDIA performance** → TensorRT-LLM.
- **Apple Silicon** → oMLX / MLX.
- **Local desktop / edge** → Ollama, llama.cpp, MLC LLM.

## Techniques still central

PagedAttention, continuous batching, speculative decoding (Medusa/EAGLE-3), FP8/FP4 quantization (AWQ/GPTQ/GGUF), KV-cache compression (GQA/MQA), disaggregated prefill/decode (Mooncake/DistServe).

## Where to go deeper in AIForge

- [`04_MLOPS_AND_PRODUCTION_AI/LLM_Inference`](../04_MLOPS_AND_PRODUCTION_AI/LLM_Inference/)
- [`04_MLOPS_AND_PRODUCTION_AI/Inference_Optimization`](../04_MLOPS_AND_PRODUCTION_AI/Inference_Optimization/)

## Sources
- [Yotta Labs — Best LLM inference engines 2026](https://www.yottalabs.ai/post/best-llm-inference-engines-in-2026-vllm-tensorrt-llm-tgi-and-sglang-compared)
- [StableLearn — 2026 LLM inference framework guide](https://stable-learn.com/en/llm-inference-framework-guide-2026/)
- [Spheron — vLLM vs TensorRT-LLM vs SGLang H100 benchmarks](https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/)
- [Sesame Disk — llama.cpp vs vLLM vs SGLang vs Ollama 2026](https://sesamedisk.com/llamacpp-vs-vllm-vs-sglang-vs-ollama-2026/)
