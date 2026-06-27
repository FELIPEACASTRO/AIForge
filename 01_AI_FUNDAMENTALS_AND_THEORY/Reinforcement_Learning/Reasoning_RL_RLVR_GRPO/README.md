# Reasoning RL RLVR GRPO

> Reinforcement Learning with Verifiable Rewards (RLVR) and Group Relative Policy Optimization (GRPO) — the training recipe behind DeepSeek-R1, Kimi k1.5 and the 2025–2026 wave of large reasoning models, where binary correctness signals replace learned reward models and a critic-free policy-gradient estimator scales chain-of-thought reasoning.

## Why it matters

RLVR is the post-training paradigm that turned base LLMs into reasoning models: instead of a learned reward model (RLHF), it uses deterministic verifiers (math answer-checkers, unit-test pass/fail) to emit a binary reward, sidestepping much of the reward-hacking that plagues preference-based RL. GRPO removes the value/critic network entirely by computing advantages from the relative reward of a group of sampled rollouts, making large-scale reasoning RL tractable on commodity GPU clusters. Together they are the most consequential RL development for LLMs since RLHF, and the open recipes (Open-R1, DAPO, verl) make the method reproducible outside frontier labs.

## Taxonomy

| Sub-area | What it covers | Representative work |
|---|---|---|
| **Verifiable rewards (RLVR)** | Rule/verifier-based binary rewards for math, code, logic | Tulu 3 RLVR, DeepSeek-R1 |
| **Critic-free policy optimization** | GRPO and group-baseline advantage estimation | DeepSeekMath, GRPO |
| **GRPO variants / fixes** | Length-bias, normalization, clip and sampling fixes | Dr.GRPO, DAPO, REINFORCE++ |
| **Zero-RL ("R1-Zero")** | RL directly on a base model, no SFT cold-start | DeepSeek-R1-Zero, Open-Reasoner-Zero |
| **Reward hacking / Goodhart** | Models gaming verifiers/formatting instead of reasoning | RLVR robustness studies |
| **Open training stacks** | Scalable RL infra (Ray/vLLM rollout + FSDP/Megatron train) | verl, OpenRLHF, TRL, NeMo-RL |

## Key methods

| Method | Core idea | Link |
|---|---|---|
| **GRPO** | Sample a group of G outputs per prompt; advantage = (reward − group mean) / group std; no critic network | https://arxiv.org/abs/2402.03300 |
| **RLVR** | Replace reward model with deterministic verifier giving binary correct/incorrect reward | https://arxiv.org/abs/2411.15124 |
| **DAPO** | Decoupled Clip + Dynamic Sampling: clip-higher (anti entropy-collapse), dynamic sampling, token-level loss, overlong reward shaping | https://arxiv.org/abs/2503.14476 |
| **Dr.GRPO** | Removes length and std normalization biases in GRPO that inflate response length | https://arxiv.org/abs/2503.20783 |
| **REINFORCE++** | Lightweight global-baseline REINFORCE variant, stable without group sampling | https://arxiv.org/abs/2501.03262 |
| **R1-Zero training** | Pure RL on a base model (no SFT), produces emergent long CoT and "aha" behavior | https://arxiv.org/abs/2501.12948 |

## Key frameworks & tools

| Tool | Origin | Notes | Link |
|---|---|---|---|
| **verl** | ByteDance Seed | Most mature high-performance RL stack; GRPO/PPO/DAPO at scale (Ray + vLLM + FSDP/Megatron) | https://github.com/volcengine/verl |
| **OpenRLHF** | Open community | Ray-based, easy-to-use; PPO, DAPO, REINFORCE++, async RL, VLM | https://github.com/OpenRLHF/OpenRLHF |
| **TRL (GRPOTrainer)** | Hugging Face | Tight HF ecosystem integration; GRPO + reward functions | https://huggingface.co/docs/trl/main/en/grpo_trainer |
| **Open-R1** | Hugging Face | Open reproduction of the DeepSeek-R1 pipeline | https://github.com/huggingface/open-r1 |
| **NeMo-RL** | NVIDIA | Scalable post-training (GRPO/PPO/DPO) with Megatron backend | https://github.com/NVIDIA/NeMo-RL |
| **Open-Reasoner-Zero** | OpenReasonerZero | Minimal, open R1-Zero-style training recipe | https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero |
| **DAPO (code)** | ByteDance/Tsinghua | Official DAPO recipe built on verl | https://github.com/BytedTsinghua-SIA/DAPO |
| **awesome-RLVR** | OpenDILab | Curated continually-updated RLVR reading list | https://github.com/opendilab/awesome-RLVR |

## Benchmarks

| Benchmark | Domain | Use in RLVR | Link |
|---|---|---|---|
| **AIME 2024/2025** | Competition math | Headline metric for reasoning RL (DAPO, Dr.GRPO, R1) | https://huggingface.co/datasets/Maxwell-Jia/AIME_2024 |
| **MATH-500** | Competition math | Standard math reasoning eval subset | https://huggingface.co/datasets/HuggingFaceH4/MATH-500 |
| **GPQA Diamond** | Graduate science QA | Out-of-domain generalization check | https://arxiv.org/abs/2311.12022 |
| **LiveCodeBench** | Code reasoning | Contamination-resistant coding eval | https://arxiv.org/abs/2403.07974 |
| **Codeforces / SWE-bench** | Code / SWE | Verifiable code reward domains | https://arxiv.org/abs/2310.06770 |

## Key papers

| Paper | Year | Link |
|---|---|---|
| DeepSeekMath — introduces GRPO | 2024 | https://arxiv.org/abs/2402.03300 |
| Tulu 3 — RLVR at scale (AllenAI) | 2024 | https://arxiv.org/abs/2411.15124 |
| DeepSeek-R1 — incentivizing reasoning via RL | 2025 | https://arxiv.org/abs/2501.12948 |
| Kimi k1.5 — scaling RL with LLMs | 2025 | https://arxiv.org/abs/2501.12599 |
| REINFORCE++ — simple & stable RLHF baseline | 2025 | https://arxiv.org/abs/2501.03262 |
| DAPO — open-source RL system at scale | 2025 | https://arxiv.org/abs/2503.14476 |
| Understanding R1-Zero-Like Training (Dr.GRPO) | 2025 | https://arxiv.org/abs/2503.20783 |

## Cross-references in AIForge

- [Reinforcement Learning — OpenRLHF](../OpenRLHF_2025.md)
- [Test-Time Compute](../../Test_Time_Compute/)
- [Modern Fine-Tuning](../../Modern_Fine_Tuning/)
- [AI Safety and Alignment](../../AI_Safety_and_Alignment/)

## Sources

- https://arxiv.org/abs/2402.03300 — DeepSeekMath / GRPO
- https://arxiv.org/abs/2501.12948 — DeepSeek-R1
- https://arxiv.org/abs/2411.15124 — Tulu 3 / RLVR
- https://arxiv.org/abs/2503.14476 — DAPO
- https://arxiv.org/abs/2503.20783 — Dr.GRPO / Understanding R1-Zero
- https://arxiv.org/abs/2501.12599 — Kimi k1.5
- https://github.com/volcengine/verl — verl
- https://github.com/OpenRLHF/OpenRLHF — OpenRLHF
- https://huggingface.co/docs/trl/main/en/grpo_trainer — TRL GRPOTrainer
- https://github.com/huggingface/open-r1 — Open-R1
- https://github.com/opendilab/awesome-RLVR — awesome-RLVR

_Expanded from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
