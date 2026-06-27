# Reasoning Models

> Reasoning models (a.k.a. Large Reasoning Models / LRMs) are LLMs trained to "think before answering" by emitting long internal chain-of-thought, typically via reinforcement learning with verifiable rewards (RLVR) plus inference-time (test-time) compute scaling. Examples: OpenAI o1/o3/o4-mini, DeepSeek-R1, Gemini 2.5 Thinking, Claude extended thinking, Qwen QwQ.

## Why it matters

Reasoning models are the defining model class of 2025-2026: they trade extra inference compute for large accuracy gains on math, code, and science, breaking ceilings that plain next-token prediction plateaued at. The key shift is from RL-for-alignment (RLHF/DPO) to RL-for-correctness (RLVR), where rewards come from automatically checkable outcomes (a math answer, a unit test) rather than human preference. DeepSeek-R1 showed that "zero-RL" on a base model can induce reflective, self-correcting reasoning without supervised CoT data, and that this capability distills cheaply into small dense models. This opened a fully open reproduction race (Open-R1, OpenThoughts, Tülu 3, s1) that put frontier-level reasoning within reach of academic budgets.

## Taxonomy

| Approach | Idea | Representative work |
|---|---|---|
| **CoT prompting** | Elicit step-by-step reasoning via few-shot exemplars (no training) | Wei et al. 2022 |
| **Outcome RL / RLVR** | RL against an automatic verifier (answer match, unit tests) | DeepSeek-R1, Tülu 3, DAPO |
| **Process supervision (PRM)** | Reward/verify each intermediate step, not just the final answer | Let's Verify Step by Step, PRM800K |
| **SFT distillation of long-CoT** | Fine-tune small models on traces from a strong teacher (R1/QwQ) | OpenThoughts, Sky-T1, R1-Distill |
| **Data-efficient SFT** | Tiny, curated, hard reasoning sets unlock latent ability | LIMO, s1 (1k samples) |
| **Test-time scaling** | Spend more compute at inference: longer CoT, sampling, search, budget forcing | s1, self-consistency, best-of-N |
| **RL algorithms** | Policy-optimization variants tuned for long-CoT stability | PPO, GRPO, Dr.GRPO, DAPO, VAPO |

## Key models

| Model | Lab | Open weights | Link |
|---|---|---|---|
| o1 / o3 / o4-mini | OpenAI | No | https://openai.com/index/learning-to-reason-with-llms/ |
| DeepSeek-R1 (+ distills) | DeepSeek | Yes (MIT) | https://github.com/deepseek-ai/DeepSeek-R1 |
| Gemini 2.5 (Thinking) | Google DeepMind | No | https://deepmind.google/technologies/gemini/ |
| Claude (extended thinking) | Anthropic | No | https://www.anthropic.com/news/visible-extended-thinking |
| QwQ-32B | Alibaba Qwen | Yes (Apache-2.0) | https://huggingface.co/Qwen/QwQ-32B |
| Kimi k1.5 | Moonshot AI | Partial | https://arxiv.org/abs/2501.12599 |
| Tülu 3 (RLVR) | Allen Institute (Ai2) | Yes | https://allenai.org/blog/tulu-3 |
| Nemotron / OpenReasoning | NVIDIA | Yes | https://huggingface.co/nvidia |

## Key open methods & datasets

| Resource | What it is | Link |
|---|---|---|
| Open-R1 | HF's open reproduction of the R1 pipeline | https://github.com/huggingface/open-r1 |
| OpenThoughts | Open data recipes; OpenThinker3-7B SOTA | https://github.com/open-thoughts/open-thoughts |
| Sky-T1 (NovaSky) | Train an o1-preview-class model for ~$450 | https://github.com/NovaSky-AI/SkyThought |
| s1 / s1K | 1k-sample SFT + budget forcing | https://github.com/simplescaling/s1 |
| LIMO | "Less is More" reasoning SFT | https://github.com/GAIR-NLP/LIMO |
| PRM800K | 800k step-level human labels (process supervision) | https://github.com/openai/prm800k |
| open-instruct | Ai2 post-training codebase (RLVR) | https://github.com/allenai/open-instruct |
| OpenR1-Math-220K | Open math reasoning trace dataset | https://huggingface.co/datasets/open-r1/OpenR1-Math-220k |

## Benchmarks

| Benchmark | Domain | Link |
|---|---|---|
| AIME 2024 / 2025 | Competition math (hard) | https://huggingface.co/datasets/HuggingFaceH4/aime_2024 |
| MATH-500 | Competition math | https://github.com/openai/prm800k |
| GPQA Diamond | Graduate-level science (Google-proof) | https://arxiv.org/abs/2311.12022 |
| LiveCodeBench | Contamination-free coding | https://livecodebench.github.io/ |
| ARC-AGI | Abstraction & reasoning corpus | https://arcprize.org/ |
| FrontierMath | Research-level math (very hard) | https://epoch.ai/frontiermath |
| Humanity's Last Exam | Broad expert frontier | https://agi.safe.ai/ |

Indicative figures (vendor-reported, shift quickly): o3 ≈83% AIME 2024, ≈87.7% GPQA Diamond, ≈96.7% ARC-AGI; DeepSeek-R1 ≈79.8% AIME 2024. Treat all as approximate and verify against the latest sources.

## Key papers

| Paper | arXiv / link |
|---|---|
| Chain-of-Thought Prompting Elicits Reasoning (Wei et al., 2022) | https://arxiv.org/abs/2201.11903 |
| Let's Verify Step by Step (Lightman et al., 2023) — PRMs | https://arxiv.org/abs/2305.20050 |
| DeepSeekMath — introduces GRPO (Shao et al., 2024) | https://arxiv.org/abs/2402.03300 |
| Tülu 3: Open Post-Training + RLVR (Lambert et al., 2024) | https://arxiv.org/abs/2411.15124 |
| DeepSeek-R1: Incentivizing Reasoning via RL (Guo et al., 2025) | https://arxiv.org/abs/2501.12948 |
| Kimi k1.5: Scaling RL with LLMs (Team Kimi, 2025) | https://arxiv.org/abs/2501.12599 |
| s1: Simple Test-Time Scaling (Muennighoff et al., 2025) | https://arxiv.org/abs/2501.19393 |
| DAPO: Open-Source LLM RL System at Scale (Yu et al., 2025) | https://arxiv.org/abs/2503.14476 |
| OpenThoughts: Data Recipes for Reasoning Models (2025) | https://arxiv.org/abs/2506.04178 |
| A Survey of Process Reward Models (2025) | https://arxiv.org/abs/2510.08049 |

## Cross-references in AIForge

- [Text LLMs](../Text_LLMs/) — base general-purpose LLMs that reasoning models are post-trained from
- [Small Language Models](../Small_Language_Models/) — targets for cheap long-CoT distillation (R1-Distill, OpenThinker)
- [Code LLMs](../Code_LLMs/) — code reasoning + verifiable (unit-test) reward signals
- [Research Labs](../Research_Labs/) — OpenAI, DeepSeek, Google DeepMind, Ai2, Qwen, Moonshot

## Sources

- https://arxiv.org/abs/2501.12948 — DeepSeek-R1
- https://openai.com/index/learning-to-reason-with-llms/ — OpenAI o1
- https://arxiv.org/abs/2402.03300 — DeepSeekMath / GRPO
- https://arxiv.org/abs/2503.14476 — DAPO
- https://arxiv.org/abs/2411.15124 — Tülu 3 / RLVR
- https://arxiv.org/abs/2501.19393 — s1
- https://arxiv.org/abs/2506.04178 — OpenThoughts
- https://arxiv.org/abs/2305.20050 — Let's Verify Step by Step
- https://github.com/huggingface/open-r1 — Open-R1
- https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training — survey overview

_Seed section — curated from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
