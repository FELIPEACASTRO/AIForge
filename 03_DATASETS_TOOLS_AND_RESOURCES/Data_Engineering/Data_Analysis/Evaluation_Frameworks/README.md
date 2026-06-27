# Evaluation Frameworks

> Software libraries and harnesses that *run* LLM/foundation-model evaluations — defining tasks, sampling models, scoring outputs, and reporting reproducible results — as distinct from the benchmark *datasets* and leaderboards they execute.

## Why it matters

A benchmark dataset is inert; the framework is what turns it into a number you can trust and reproduce. Differences in prompting, few-shot ordering, answer parsing, and aggregation routinely swing reported scores by several points, so the *harness* is part of the result. Standardized frameworks make evaluations comparable across models, support LLM-as-a-judge and RAG-specific metrics that go beyond exact match, and integrate eval into CI/CD so regressions are caught before deployment.

> Scope note: pillar 01 hosts `AI_Evaluation` (prompt-/methodology-level concepts). This page is scoped to **tooling you run**. For benchmark datasets and leaderboards, see `HuggingFace_Hub/Leaderboards_and_Evaluation.md`.

## Taxonomy

| Sub-area | What it covers | Representative tools |
|---|---|---|
| Academic harnesses | Standardized task/model interfaces, reproducible accuracy benchmarks | lm-evaluation-harness, HELM, lighteval, OpenAI Evals |
| Application / unit-test eval | Pytest-style assertions, CI gating, metric library | DeepEval, promptfoo |
| RAG-specific eval | Faithfulness, context relevance/recall, answer relevance | Ragas, TruLens |
| Agent / safety eval | Tool-call traces, scripted eval plans, model-graded rubrics | Inspect AI, OpenAI Evals |
| Observability + eval platforms | Tracing, online eval, LLM-as-judge scoring in production | Arize Phoenix, Langfuse |
| Judge methods | Algorithms scoring outputs (LLM-as-judge, G-Eval, pairwise) | MT-Bench/Arena, G-Eval, RAGAS metrics |

## Key frameworks & tools

| Tool | Maintainer | Focus | Link |
|---|---|---|---|
| lm-evaluation-harness | EleutherAI | De-facto academic standard; backend for the Open LLM Leaderboard | https://github.com/EleutherAI/lm-evaluation-harness |
| HELM | Stanford CRFM | Holistic multi-metric eval (accuracy, calibration, robustness, bias, toxicity, efficiency) | https://github.com/stanford-crfm/helm |
| lighteval | Hugging Face | Lightweight, backend-agnostic pipeline; leaderboard-style runs | https://github.com/huggingface/lighteval |
| OpenAI Evals | OpenAI | Eval framework + open registry of benchmarks | https://github.com/openai/evals |
| simple-evals | OpenAI | Minimal, transparent reference implementations of common benchmarks | https://github.com/openai/simple-evals |
| Inspect AI | UK AI Safety Institute (AISI) | Scripted eval plans, tool use, model-graded rubrics; safety/agent evals | https://github.com/UKGovernmentBEIS/inspect_ai |
| DeepEval | Confident AI | "Pytest for LLMs"; metric library + CI gating | https://github.com/confident-ai/deepeval |
| promptfoo | promptfoo | Local-first CLI/dashboard; prompt A/B, RAG/agent eval, red-team | https://github.com/promptfoo/promptfoo |
| Ragas | Exploding Gradients | RAG + agentic eval metrics (faithfulness, context recall) | https://github.com/explodinggradients/ragas |
| TruLens | TruEra / Snowflake | RAG triad feedback functions, composable evals | https://github.com/truera/trulens |
| Phoenix | Arize AI | Open-source LLM observability + online/offline eval | https://github.com/Arize-ai/phoenix |
| Langfuse | Langfuse | Tracing + eval + prompt management platform | https://github.com/langfuse/langfuse |

## Judge methods & metric definitions

| Method | Idea | Reference |
|---|---|---|
| LLM-as-a-Judge (MT-Bench / Arena) | Strong LLM grades open-ended answers; ~80% agreement with humans | https://arxiv.org/abs/2306.05685 |
| G-Eval | CoT + form-filling LLM scoring of NLG quality | https://arxiv.org/abs/2303.16634 |
| RAGAS metrics | Reference-free faithfulness / context / answer relevance | https://arxiv.org/abs/2309.15217 |
| TruLens RAG triad | Groundedness, context relevance, answer relevance | https://www.trulens.org/ |

## Benchmarks commonly run through these frameworks

| Benchmark | Scope | Link |
|---|---|---|
| MMLU | 57-subject multitask knowledge | https://arxiv.org/abs/2009.03300 |
| BIG-bench | 200+ diverse tasks, 450 authors | https://arxiv.org/abs/2206.04615 |
| HELM scenarios | 16 core scenarios × 7 metrics | https://crfm.stanford.edu/helm/ |
| MT-Bench | Multi-turn instruction following | https://arxiv.org/abs/2306.05685 |
| GPQA | Graduate-level Google-proof QA | https://arxiv.org/abs/2311.12022 |
| IFEval | Instruction-following verifiability | https://arxiv.org/abs/2311.07911 |

## Key papers

| Paper | Year | Link |
|---|---|---|
| Holistic Evaluation of Language Models (HELM) | 2022 | https://arxiv.org/abs/2211.09110 |
| Beyond the Imitation Game (BIG-bench) | 2022 | https://arxiv.org/abs/2206.04615 |
| Measuring Massive Multitask Language Understanding (MMLU) | 2020 | https://arxiv.org/abs/2009.03300 |
| Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena | 2023 | https://arxiv.org/abs/2306.05685 |
| G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment | 2023 | https://arxiv.org/abs/2303.16634 |
| Ragas: Automated Evaluation of Retrieval Augmented Generation | 2023 | https://arxiv.org/abs/2309.15217 |
| GPQA: A Graduate-Level Google-Proof Q&A Benchmark | 2023 | https://arxiv.org/abs/2311.12022 |
| Instruction-Following Evaluation for LLMs (IFEval) | 2023 | https://arxiv.org/abs/2311.07911 |

## Cross-references in AIForge

- [HuggingFace Hub — Leaderboards and Evaluation](../../../HuggingFace_Hub/Leaderboards_and_Evaluation.md) — benchmark datasets and public leaderboards
- [Data Analysis (parent section)](../) — sibling data-quality and analysis tooling
- [Data Engineering (pillar root)](../../) — pipelines that feed eval datasets

## Sources

- https://github.com/EleutherAI/lm-evaluation-harness
- https://github.com/stanford-crfm/helm
- https://github.com/huggingface/lighteval
- https://github.com/openai/evals
- https://github.com/openai/simple-evals
- https://github.com/UKGovernmentBEIS/inspect_ai
- https://github.com/confident-ai/deepeval
- https://github.com/promptfoo/promptfoo
- https://github.com/explodinggradients/ragas
- https://github.com/truera/trulens
- https://github.com/Arize-ai/phoenix
- https://github.com/langfuse/langfuse
- https://arxiv.org/abs/2211.09110
- https://arxiv.org/abs/2306.05685
- https://arxiv.org/abs/2303.16634
- https://arxiv.org/abs/2309.15217
- https://arxiv.org/abs/2206.04615
