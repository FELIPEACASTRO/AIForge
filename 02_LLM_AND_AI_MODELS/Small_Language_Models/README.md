# Small Language Models

> Small Language Models (SLMs) are compact transformer LMs — typically 100M–4B parameters — engineered for on-device, edge, and cost-sensitive deployment through data-quality curation, distillation, and quantization, while approaching the quality of models many times their size.

## Why it matters

SLMs are one of the defining 2024–2026 trends in applied AI: they run locally on phones, laptops, and edge hardware, keeping data private and inference cheap with low latency. Architectural and data-recipe advances (deep-thin designs, grouped-query attention, embedding sharing, synthetic/distilled data) now let sub-4B models match older 7B–70B models on math, coding, and instruction-following. They are increasingly the right default for agentic workflows, RAG retrievers, function-calling routers, and any deployment where cloud cost, latency, or privacy dominate.

## Taxonomy

| Sub-area | What it covers | Representative work |
|---|---|---|
| Edge / on-device families | Models tuned to fit phone/ARM memory budgets | Llama 3.2 1B/3B, MobileLLM, Gemma 3 270M/1B |
| Data-quality SLMs | "Textbook-quality" + synthetic data recipes | Phi-3/Phi-4-Mini, SmolLM2/3 |
| Distillation-trained SLMs | Small students distilled from large teachers | Gemma 2/3 small, Qwen distilled variants |
| Quantized SLMs | INT4/INT8/QAT for reduced memory footprint | Quantized Llama 3.2, Gemma 3 QAT |
| Small reasoning models | RL/SFT-tuned compact reasoners | Phi-4-Mini-Reasoning, SmolLM3 (think mode) |
| SLM ↔ LLM collaboration | Routing, cascades, cloud-edge hybrids | See surveys below |

## Key models

| Model | Sizes | Org | Highlights | Link |
|---|---|---|---|---|
| Phi-4-Mini | 3.8B | Microsoft | 200K vocab, GQA, strong math/code; multimodal & reasoning variants | https://arxiv.org/abs/2503.01743 |
| Phi-3-Mini | 3.8B | Microsoft | "Textbook" data recipe, runs on phone | https://arxiv.org/abs/2404.14219 |
| Gemma 3 | 1B / 4B (+270M) | Google DeepMind | 128K context, multimodal (4B+), distillation, QAT | https://arxiv.org/abs/2503.19786 |
| Gemma 2 | 2B / 9B | Google DeepMind | Distillation-trained small models | https://arxiv.org/abs/2408.00118 |
| SmolLM2 / SmolLM3 | 135M–3B | Hugging Face | Fully open; SmolLM3 dual-mode reasoning, 128K, 6 langs | https://github.com/huggingface/smollm |
| Qwen2.5 (small) | 0.5B / 1.5B / 3B | Alibaba Qwen | 18T-token data, quantized variants, coder series | https://arxiv.org/abs/2412.15115 |
| Llama 3.2 | 1B / 3B | Meta | Edge-optimized, 128K, ARM/Qualcomm/MediaTek day-one | https://huggingface.co/meta-llama/Llama-3.2-1B |
| MobileLLM | 125M / 350M | Meta | Deep-thin arch, embedding sharing, GQA for phones | https://arxiv.org/abs/2402.14905 |
| TinyLlama | 1.1B | Open community | 3T-token pretraining on Llama 2 arch | https://arxiv.org/abs/2401.02385 |

## Design techniques

| Technique | Purpose | Reference |
|---|---|---|
| Synthetic / "textbook-quality" data | Lift small-model quality via curated training mix | Phi-3 report https://arxiv.org/abs/2404.14219 |
| Knowledge distillation | Transfer large-teacher behavior into small student | Gemma 3 report https://arxiv.org/abs/2503.19786 |
| Grouped-query attention (GQA) | Cut KV-cache / speed long-context inference | MobileLLM https://arxiv.org/abs/2402.14905 |
| Embedding sharing + deep-thin nets | Maximize accuracy per parameter at sub-1B scale | MobileLLM https://arxiv.org/abs/2402.14905 |
| Quantization (INT4/INT8, QAT) | Shrink memory & enable on-device inference | Quantized Llama https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/ |
| NoPE + staged curriculum | Long-context + reasoning on small budgets | SmolLM3 https://huggingface.co/blog/smollm3 |

## On-device runtimes & tooling

| Tool | Use | Link |
|---|---|---|
| llama.cpp | C/C++ GGUF inference on CPU/edge | https://github.com/ggml-org/llama.cpp |
| Ollama | Local model runner / serving | https://github.com/ollama/ollama |
| MLC LLM | Compile LLMs for mobile/web/edge | https://github.com/mlc-ai/mlc-llm |
| ExecuTorch | PyTorch on-device runtime (Llama edge) | https://github.com/pytorch/executorch |
| MLX | Apple-silicon array framework for local LMs | https://github.com/ml-explore/mlx |

## Benchmarks

SLMs are typically reported on the same suites as larger LLMs, scaled for their class:

| Benchmark | Measures | Link |
|---|---|---|
| MMLU | Broad knowledge (57 subjects) | https://arxiv.org/abs/2009.03300 |
| GSM8K | Grade-school math reasoning | https://arxiv.org/abs/2110.14168 |
| HumanEval | Code generation pass@k | https://arxiv.org/abs/2107.03374 |
| IFEval | Instruction-following | https://arxiv.org/abs/2311.07911 |
| Open LLM Leaderboard (HF) | Aggregated community eval | https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard |

## Key papers

| Paper | Year | Link |
|---|---|---|
| Phi-4-Mini Technical Report (Mixture-of-LoRAs) | 2025 | https://arxiv.org/abs/2503.01743 |
| Phi-3 Technical Report | 2024 | https://arxiv.org/abs/2404.14219 |
| Gemma 3 Technical Report | 2025 | https://arxiv.org/abs/2503.19786 |
| Gemma 2: Improving Open LMs at a Practical Size | 2024 | https://arxiv.org/abs/2408.00118 |
| Qwen2.5 Technical Report | 2024 | https://arxiv.org/abs/2412.15115 |
| MobileLLM: Sub-billion Parameter LMs for On-Device | 2024 | https://arxiv.org/abs/2402.14905 |
| TinyLlama: An Open-Source Small Language Model | 2024 | https://arxiv.org/abs/2401.02385 |
| Small Language Models: Survey, Measurements, and Insights | 2024 | https://arxiv.org/abs/2409.15790 |
| A Survey of Small Language Models | 2024 | https://arxiv.org/abs/2410.20011 |
| Small Language Models are the Future of Agentic AI | 2025 | https://arxiv.org/abs/2506.02153 |

## Cross-references in AIForge

- [Text LLMs](../Text_LLMs/) — general-purpose large language models
- [MoE Models](../MoE_Models/) — sparse models as an alternative efficiency axis
- [Reasoning Models](../Reasoning_Models/) — small reasoning variants (Phi-4-Mini-Reasoning, SmolLM3)
- [Frameworks](../Frameworks/) — serving and inference stacks for SLMs

## Sources

- https://arxiv.org/abs/2503.01743 — Phi-4-Mini Technical Report
- https://arxiv.org/abs/2404.14219 — Phi-3 Technical Report
- https://arxiv.org/abs/2503.19786 — Gemma 3 Technical Report
- https://arxiv.org/abs/2412.15115 — Qwen2.5 Technical Report
- https://arxiv.org/abs/2402.14905 — MobileLLM
- https://arxiv.org/abs/2401.02385 — TinyLlama
- https://arxiv.org/abs/2409.15790 — SLM Survey, Measurements, and Insights
- https://huggingface.co/blog/smollm3 — SmolLM3 release
- https://huggingface.co/HuggingFaceTB/SmolLM3-3B — SmolLM3-3B model card
- https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/ — Llama 3.2 edge release
- https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/ — Quantized Llama 3.2

_Seed section expanded from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
