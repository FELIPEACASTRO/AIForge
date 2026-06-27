# Test Time Compute

> Spending additional compute at inference time — via repeated sampling, search, verifiers, or extended chain-of-thought — to improve answer quality without changing model weights. The 2024–2026 paradigm shift behind o1/o3, DeepSeek-R1, Gemini-Thinking, and QwQ.

## Why it matters

Scaling test-time compute (TTC) can be more parameter-efficient than scaling pre-training: a small model with a good verifier and search budget can match or beat a much larger model on hard, verifiable tasks (math, code, logic). It reframes inference from a single forward pass into a *search-and-verify* process, opening a second scaling axis (FLOPs-at-inference) orthogonal to model size. This is the core mechanism behind frontier "reasoning models" and a central, fast-moving research area in 2024–2026.

## Taxonomy

| Axis | Approach | Idea |
|---|---|---|
| **Sequential** | Long chain-of-thought / budget forcing | One long trace; force the model to "think" longer (e.g. append "Wait") before answering |
| **Sequential** | Self-refinement / self-correction | Iteratively revise an answer using self- or verifier feedback |
| **Parallel** | Best-of-N (BoN) sampling | Sample N candidates, pick the best via verifier or majority vote |
| **Parallel** | Self-consistency | Sample many CoT paths, marginalize via majority vote on the final answer |
| **Search** | Beam search / lookahead | Expand multiple partial steps, score with a PRM, keep top beams |
| **Search** | MCTS / tree search (incl. DVTS) | Tree-structured exploration of reasoning steps with value backup |
| **Verification** | Outcome Reward Model (ORM) | Scores only the final answer; pairs naturally with BoN |
| **Verification** | Process Reward Model (PRM) | Scores each intermediate step; enables step-level pruning and search |
| **Training-side** | RL for reasoning (RLVR) | Train the model to emit useful long CoT (DeepSeek-R1, o1-style) |

## Key methods and frameworks

| Name | What it is | Link |
|---|---|---|
| Self-Consistency | Majority vote over sampled CoT paths | https://arxiv.org/abs/2203.11171 |
| Tree of Thoughts (ToT) | Deliberate tree search over thought steps | https://arxiv.org/abs/2305.10601 |
| Best-of-N + verifier | Sample N, rerank with ORM/PRM | https://arxiv.org/abs/2407.21787 |
| Compute-optimal TTC | Pick BoN vs. search vs. revision by prompt difficulty | https://arxiv.org/abs/2408.03314 |
| s1 budget forcing | Force longer/shorter thinking via control tokens | https://arxiv.org/abs/2501.19393 |
| Process Reward Models (PRM800K) | Step-level verifier training | https://arxiv.org/abs/2305.20050 |
| ThinkPRM (generative PRM) | PRM that reasons before scoring steps | https://arxiv.org/abs/2504.16828 |
| HF `search-and-learn` | Recipes: BoN, beam search, DVTS with PRMs | https://github.com/huggingface/search-and-learn |

## Key reasoning models (TTC in production)

| Model | Org | Notes | Link |
|---|---|---|---|
| o1 / o3 | OpenAI | RL-trained long-CoT; inference-time reasoning | https://openai.com/index/learning-to-reason-with-llms/ |
| DeepSeek-R1 / R1-Zero | DeepSeek | Open-weight; reasoning emerges from pure RL | https://arxiv.org/abs/2501.12948 |
| QwQ-32B-Preview | Qwen | First open-weight o1-style reasoning model | https://qwenlm.github.io/blog/qwq-32b-preview/ |
| Qwen3 (thinking mode) | Qwen | Unified thinking / non-thinking inference | https://arxiv.org/abs/2505.09388 |
| Gemini 2.0/2.5 Flash Thinking | Google DeepMind | Exposed reasoning traces | https://deepmind.google/technologies/gemini/ |

## Benchmarks

| Benchmark | Domain | Link |
|---|---|---|
| MATH-500 | Competition math (verifiable) | https://huggingface.co/datasets/HuggingFaceH4/MATH-500 |
| AIME 2024 | Olympiad math, hard | https://huggingface.co/datasets/HuggingFaceH4/aime_2024 |
| GPQA Diamond | Graduate-level science Q&A | https://arxiv.org/abs/2311.12022 |
| ProcessBench | Locating step-level errors (PRM eval) | https://arxiv.org/abs/2412.06559 |
| PRM800K | 800K step-level human labels | https://github.com/openai/prm800k |

## Key papers

| Paper | Year | Link |
|---|---|---|
| Self-Consistency Improves CoT Reasoning (Wang et al.) | 2022 | https://arxiv.org/abs/2203.11171 |
| Tree of Thoughts (Yao et al.) | 2023 | https://arxiv.org/abs/2305.10601 |
| Let's Verify Step by Step / PRM800K (Lightman et al.) | 2023 | https://arxiv.org/abs/2305.20050 |
| Large Language Monkeys: Repeated Sampling (Brown et al.) | 2024 | https://arxiv.org/abs/2407.21787 |
| Scaling LLM Test-Time Compute Optimally (Snell et al.) | 2024 | https://arxiv.org/abs/2408.03314 |
| Empirical Analysis of Compute-Optimal Inference (Wu et al.) | 2024 | https://arxiv.org/abs/2408.00724 |
| s1: Simple Test-Time Scaling (Muennighoff et al.) | 2025 | https://arxiv.org/abs/2501.19393 |
| DeepSeek-R1 (DeepSeek-AI) | 2025 | https://arxiv.org/abs/2501.12948 |
| Process Reward Models That Think (Khalifa et al.) | 2025 | https://arxiv.org/abs/2504.16828 |
| A Survey of Frontiers in LLM Reasoning (inference scaling) | 2025 | https://arxiv.org/abs/2504.09037 |
| Trust but Verify: Survey on Verification for Test-Time Scaling | 2025 | https://arxiv.org/abs/2508.16665 |

## Cross-references in AIForge

- [Reinforcement Learning](../Reinforcement_Learning/) — RLVR / RL that trains long-CoT reasoning models
- [Speculative Decoding](../Speculative_Decoding/) — complementary inference-efficiency axis
- [Prompt Engineering](../Prompt_Engineering/) — chain-of-thought and prompting foundations
- [AI Evaluation](../AI_Evaluation/) — verifiers, reward models, and reasoning benchmarks

## Sources

- https://arxiv.org/abs/2408.03314 — Scaling LLM Test-Time Compute Optimally (DeepMind)
- https://arxiv.org/abs/2407.21787 — Large Language Monkeys
- https://arxiv.org/abs/2305.20050 — Let's Verify Step by Step
- https://arxiv.org/abs/2501.12948 — DeepSeek-R1
- https://arxiv.org/abs/2504.16828 — Process Reward Models That Think
- https://arxiv.org/abs/2501.19393 — s1: Simple Test-Time Scaling
- https://arxiv.org/abs/2504.09037 — Survey of Frontiers in LLM Reasoning
- https://arxiv.org/abs/2508.16665 — Survey on Verification for Test-Time Scaling
- https://github.com/huggingface/search-and-learn — HF recipes (BoN/beam/DVTS)
- https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute — HF blog
- https://openai.com/index/learning-to-reason-with-llms/ — OpenAI o1

_Seed section — curated from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
