# 🛰️ Open-Weight LLMs 2026 — Radar

> The fastest-moving front in AI: open-weight models that, by late 2026, reach striking distance of closed frontier on coding, reasoning, and agentic work.

**Refreshed:** 2026-10-09 · **Scope:** releases through October 2026

> ⚠️ Fast-moving field: entries reflect public reporting at refresh time. Confirm versions, licenses, and benchmark figures at the source links before relying on them.

## 🆕 New since the last refresh (June → October 2026)

Newest first. All figures are as reported by the linked source or model card, not independently verified here.

| Item | Date | What's new |
|---|---|---|
| [Mistral Large 4](https://mistral.ai/news/mistral-large-4/) | 2026-10-06 | Announced open-weight: 1T total / 52B active natively-multimodal MoE. API preview is live now; per Mistral's announcement, weights are due end of Oct 2026 (the `Mistral-Large-4-1T-A52B` HF repo is marked upcoming, custom license expected). |
| [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 2026-09 | Newest DeepSeek open release: MIT-licensed multimodal MoE, 552B backbone (~8B active prefill / 16B decode), 1M context. Per the model card: GPQA Diamond 90.9, Terminal-Bench 2.1 90.6, Codeforces 3471. |
| [Zhipu / Z.ai GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) | 2026-09 | 753B flagship with large agentic/coding gains over GLM-5.2. As reported: Terminal-Bench 2.1 88.2, FrontierSWE 78.1, HLE-with-tools 62.5. Open weights (GLM-5.2 on Sep 1 and GLM-5 on Aug 11 were also open). |
| [Zhipu / Z.ai GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | 2026-09 | 321B multimodal "Flash" variant of GLM-5.3, open weights on Hugging Face (a BF16 variant is also published). |
| [DeepSeek-V4-Flash-Vision-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp) | 2026-09-01 | Experimental vision-enabled V4-Flash checkpoint, open weights (MIT) on Hugging Face. |
| [Tencent Hunyuan Hy4-preview](https://huggingface.co/tencent/Hy4-preview) | 2026-08-28 | Newest Hunyuan open checkpoint: 780B preview MoE on Hugging Face (an FP8 variant is also posted); check the model card for license. |
| [IBM Granite 4.2](https://huggingface.co/ibm-granite/granite-4.2-30b) | 2026-08-25 | Open enterprise reasoning family — 3B / 8B / 30B dense (Mamba-2 hybrid), 128K context, Apache 2.0, with toggleable chain-of-thought plus agentic RL for code/terminal. |
| [Qwen3.8-2.4T-A95B (Qwen3.8-Max open weights)](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) | 2026-08 | Alibaba's first Max-class open checkpoint: 2.4T total / 95B active MoE, 262K native context (extensible to ~1M), custom `qwen3.8-max` license, text-only. |
| [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | 2026-08 | Efficient Qwen3.8 MoE: 125B total / 6B active (~180B with n-gram + MTP), 262K→1M context, `qwen-community-1.0`, multimodal. Per the model card: GPQA 91.7, LiveCodeBench v6 91.9. |
| [DeepSeek-V4-Pro-0813](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813) | 2026-08-13 | GA flagship V4-Pro MoE (reported ~1.6T total / ~49B active), 1M context, MIT open weights; reported frontier reasoning at roughly 1/6 the price of Western APIs. |
| [NVIDIA Nemotron 3.5 Lightning (30B-A3B)](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) | 2026-08-11 | Open 30B / 3B-active hybrid Mamba-2 + Attention + MoE for fast long-running agents, up to 1M context (256K on a single H100); weights + data + recipes under OpenMDW-1.1, ~670 tok/s as reported. |
| [Meta Muse Glimmer 30B](https://www.infoq.com/news/2026/08/meta-muse-glimmer/) | 2026-08-10 | Meta's first open-weight model since Llama 4: ~30B dense multimodal agentic model under Apache 2.0, runs on one 24GB consumer GPU. As reported, it leads MCP Atlas (75.5); training data was not released. |
| [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | 2026-08-01 | Open-weight V4-Flash MoE (reported 284B total / ~13B active), MIT, 1M context; the low-cost fast variant of the V4 line. |
| [MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) | 2026-07-23 | Newest MiniMax open LLM: 427B multimodal MoE (succeeding the 229B M2.7 line), published on HF/GitHub for private deployment and fine-tuning. |
| [Moonshot Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) | 2026-07 | Reported as the largest open-weight model to date: 2.8T total / 104B active MoE, 1M context, custom Kimi K3 license. Per the model card: GPQA 93.5, BrowseComp 91.2, FrontierSWE 81.2. |
| [Tencent Hunyuan Hy3](https://huggingface.co/tencent/Hy3) | 2026-07 | 299B MoE (reported ~295B total / 21B active), Apache 2.0 final release that lifted the preview's EU/UK/Korea restrictions; open weights on HF/ModelScope. |
| [Moonshot Kimi-K2.7-Code](https://huggingface.co/moonshotai/Kimi-K2.7-Code) | 2026-06-15 | Agentic coding MoE (~1T total / ~32B active) built on K2.6, 256K context, Modified MIT; reported ~30% fewer reasoning tokens. |

## Still best-in-class (earlier 2026 and 2025)

Older than the refresh window, but still the leading open option in their niche. Newest first.

| Item | Date | What's new |
|---|---|---|
| [NVIDIA Nemotron 3 Ultra (550B)](https://nvidianews.nvidia.com/news/nvidia-debuts-nemotron-3-family-of-open-models) | 2026-06-04 | 550B / 55B-active hybrid Mamba-Transformer MoE open reasoning model; weights, recipes, and datasets under OpenMDW-1.1. Reported as the top-scoring US open model at launch. |
| [Cohere Command A+](https://cohere.com/blog/command-a-plus) | 2026-05-20 | Cohere's first fully Apache-2.0 model: 218B total / 25B active MoE, 48 languages plus multimodal docs, native citations; per Cohere, runs on 2x H100 or a single Blackwell. |
| [Google Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) | 2026-04-02 | Apache-2.0 open family (E2B, E4B, 26B MoE, 31B dense), up to 256K context, multimodal, 140+ languages; the current best Gemma and a strong edge/open option. |
| [Microsoft Phi-4 family (reasoning / vision)](https://azure.microsoft.com/en-us/products/phi) | 2026-03 | MIT-licensed small open models; Phi-4-reasoning-vision-15B (Mar 2026) is the latest. No official Phi-5 as of Oct 2026 — Microsoft's newer MAI foundation models are closed. |
| [OpenAI gpt-oss-120b / gpt-oss-20b](https://openai.com/index/introducing-gpt-oss/) | 2025-08 | Apache-2.0 open reasoning MoE (117B / 5.1B active and 21B / 3.6B active); still OpenAI's top open-weight line as of Oct 2026 — no confirmed 2026 successor despite SEO claims. |

## Reading the radar

- **Chinese labs set the open pace.** DeepSeek (V4 line), Qwen (3.8 Max/Flash), Moonshot (Kimi K3), Zhipu/Z.ai (GLM-5.3), Tencent (Hunyuan), and MiniMax (M3) account for most of the headline open releases this window.
- **MoE dominates.** Almost every new flagship is a sparse mixture-of-experts; several (Granite, Nemotron) pair this with Mamba-2 hybrid attention for efficiency.
- **Licensing splits.** MIT (DeepSeek), Apache 2.0 (Granite, Cohere, Gemma, Meta Muse, gpt-oss) sit alongside custom community licenses (Qwen Max, Kimi K3) — confirm terms per model card.
- **Western open weights return.** Meta's Muse Glimmer (first open model since Llama 4) and Mistral Large 4's announced open-weight drop are the notable Western moves.

## Related

- Radar index: [`00_FRONTIER_AI_2026/README.md`](./README.md) · sibling pages: [Frontier_LLMs_2026.md](./Frontier_LLMs_2026.md) · [Reasoning_and_Agents_2026.md](./Reasoning_and_Agents_2026.md) · [Inference_and_Serving_2026.md](./Inference_and_Serving_2026.md)
- Stable pillars: [`02_LLM_AND_AI_MODELS/Text_LLMs/Open_Source_LLMs`](../02_LLM_AND_AI_MODELS/Text_LLMs/Open_Source_LLMs/) · [`02_LLM_AND_AI_MODELS/MoE_Models`](../02_LLM_AND_AI_MODELS/MoE_Models/) · [`02_LLM_AND_AI_MODELS/Small_Language_Models`](../02_LLM_AND_AI_MODELS/Small_Language_Models/)

**Sources:** per-row links above — Hugging Face model cards (deepseek-ai, Qwen, moonshotai, zai-org, MiniMaxAI, tencent, nvidia, ibm-granite), [Mistral AI news](https://mistral.ai/news/mistral-large-4/), [NVIDIA news](https://nvidianews.nvidia.com/news/nvidia-debuts-nemotron-3-family-of-open-models), [Cohere blog](https://cohere.com/blog/command-a-plus), [Google blog (Gemma 4)](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/), [InfoQ (Meta Muse Glimmer)](https://www.infoq.com/news/2026/08/meta-muse-glimmer/), [Microsoft Azure (Phi)](https://azure.microsoft.com/en-us/products/phi), [OpenAI (gpt-oss)](https://openai.com/index/introducing-gpt-oss/).

**Keywords:** open-weight LLMs 2026, open-source LLMs, DeepSeek V4, Qwen3.8 Max, Moonshot Kimi K3, GLM-5.3, MiniMax M3, Tencent Hunyuan, Mistral Large 4, Meta Muse Glimmer, NVIDIA Nemotron, IBM Granite 4.2, Cohere Command A+, Google Gemma 4, Microsoft Phi-4, OpenAI gpt-oss, MoE, Mamba hybrid, Apache 2.0, MIT license, 1M context, agentic coding, October 2026.
