# Adaptive Compute Mixture of Depths

> Architectures and routing mechanisms that let transformers (and earlier nets) spend **variable compute per token, per layer, or per sample** — a distinct axis from Mixture-of-Experts (variable *width*), covering Mixture-of-Depths, early exit, layer skipping, and adaptive halting.

## Why it matters

Standard transformers spread FLOPs uniformly across every token at every layer, even though "the" and a pivotal entity rarely need equal compute. Adaptive-compute methods learn to **route, halt, or skip**, cutting inference cost (often 30-50%+) while matching dense-model quality, and they enable test-time compute scaling on hard inputs. This is a genuinely distinct architectural lever from MoE: MoE varies *which parameters* fire (width), while Mixture-of-Depths and friends vary *how much depth/compute* each token receives.

## Taxonomy

| Sub-area | Granularity | Decision mechanism | Representative methods |
|---|---|---|---|
| **Mixture-of-Depths (MoD)** | Per token, per layer | Top-k router selects tokens to receive full block; rest take residual bypass (static graph) | MoD, MoDification, MoDE |
| **Early exit** | Per token / per sample | Confidence/entropy threshold at intermediate layers stops computation | CALM, DeeBERT, Depth-Adaptive Transformer |
| **Adaptive halting (ACT)** | Per position, recurrent depth | Cumulative halting probability triggers stop | ACT, Universal Transformer, PonderNet |
| **Dynamic layer/block skipping** | Per sample (vision) | Gating/policy net (often RL) drops blocks | SkipNet, BlockDrop |
| **Conditional branch routing** | Per token | Light vs. heavy branch per token in FFN/attention | CoLT5, Adaptive Computation Modules |
| **Recursive depth + sharing** | Per token | Router assigns recursion depth over a shared layer stack | Mixture-of-Recursions (MoR), MoEUT |

## Key methods

| Method | Year | Core idea | Link |
|---|---|---|---|
| Mixture-of-Depths (MoD) | 2024 | Top-k token routing per layer under a fixed compute budget; static computation graph | https://arxiv.org/abs/2404.02258 |
| Mixture-of-Recursions (MoR) | 2025 | Shared recursive layer stack + per-token recursion-depth router (NeurIPS 2025) | https://arxiv.org/abs/2507.10524 |
| MoDification | 2024 | Practical recipe making MoD easy to add to pretrained LLMs | https://arxiv.org/abs/2410.14268 |
| CALM (Confident Adaptive LM) | 2022 | Per-token early exit with calibrated confidence, decoder LMs | https://arxiv.org/abs/2207.07061 |
| Depth-Adaptive Transformer | 2019 | Learns per-token output depth; anytime prediction | https://arxiv.org/abs/1910.10073 |
| CoLT5 | 2023 | Conditional light/heavy branches per token for long-input transformers | https://arxiv.org/abs/2303.09752 |
| Universal Transformer | 2018 | Weight-shared recurrent block + Adaptive Computation Time halting | https://arxiv.org/abs/1807.03819 |
| PonderNet | 2021 | Probabilistic, differentiable learned halting (latent-variable ACT) | https://arxiv.org/abs/2107.05407 |
| MoEUT | 2024 | Shared-layer (Universal) transformer made competitive via MoE FFN + attention | https://arxiv.org/abs/2405.16039 |
| Adaptive Computation Modules | 2023 | Granular per-token conditional compute inside each module | https://arxiv.org/abs/2312.10193 |

## Foundational / vision precursors

| Method | Year | Idea | Link |
|---|---|---|---|
| Adaptive Computation Time (ACT) | 2016 | Original learned halting for RNNs (Graves) | https://arxiv.org/abs/1603.08983 |
| SkipNet | 2017 | Gating net skips conv blocks (supervised + RL) | https://arxiv.org/abs/1711.09485 |
| BlockDrop | 2017 | Policy net drops residual blocks per image | https://arxiv.org/abs/1711.08393 |
| DeeBERT | 2020 | Early-exit off-ramps for BERT inference | https://arxiv.org/abs/2004.12993 |

## Implementations & resources

| Resource | What | Link |
|---|---|---|
| `raymin0223/mixture_of_recursions` | Official MoR code (NeurIPS 2025) | https://github.com/raymin0223/mixture_of_recursions |
| MoR paper page (HF) | Discussion + artifacts | https://huggingface.co/papers/2507.10524 |
| MoEUT paper page (HF) | Code + reviews | https://huggingface.co/papers/2405.16039 |
| Conditional computation survey (HTML) | Principles & research trends | https://arxiv.org/html/2403.07965v1 |

## Key papers

- **Mixture-of-Depths: Dynamically allocating compute in transformer-based language models** — Raposo, Ritter, Richards, Lillicrap, Humphreys, Santoro (2024). https://arxiv.org/abs/2404.02258
- **Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation** — Bae et al. (NeurIPS 2025). https://arxiv.org/abs/2507.10524
- **Confident Adaptive Language Modeling (CALM)** — Schuster et al. (NeurIPS 2022). https://arxiv.org/abs/2207.07061
- **Depth-Adaptive Transformer** — Elbayad, Gu, Grave, Auli (ICLR 2020). https://arxiv.org/abs/1910.10073
- **CoLT5: Faster Long-Range Transformers with Conditional Computation** — Ainslie et al. (2023). https://arxiv.org/abs/2303.09752
- **Universal Transformers** — Dehghani et al. (ICLR 2019). https://arxiv.org/abs/1807.03819
- **PonderNet: Learning to Ponder** — Banino, Balaguer, Blundell (2021). https://arxiv.org/abs/2107.05407
- **Adaptive Computation Time for Recurrent Neural Networks** — Graves (2016). https://arxiv.org/abs/1603.08983
- **Conditional computation in neural networks: principles and research trends** — survey (2024). https://arxiv.org/abs/2403.07965

## Cross-references in AIForge

- [MoE Models](../../../02_LLM_AND_AI_MODELS/MoE_Models/) — sparse width-routing; the complementary axis to depth-routing.
- [Attention Mechanisms](../attention_mechanisms/) — the block whose cost MoD/CoLT5 conditionally reduce.
- [Neural Architecture Search](../Neural_Architecture_Search/) — automating compute-allocation design choices.
- [Reasoning Models](../../../02_LLM_AND_AI_MODELS/Reasoning_Models/) — test-time / adaptive compute scaling at the system level.

## Sources

- https://arxiv.org/abs/2404.02258 — Mixture-of-Depths
- https://arxiv.org/abs/2507.10524 — Mixture-of-Recursions
- https://arxiv.org/abs/2207.07061 — CALM
- https://arxiv.org/abs/1910.10073 — Depth-Adaptive Transformer
- https://arxiv.org/abs/2303.09752 — CoLT5
- https://arxiv.org/abs/1807.03819 — Universal Transformers
- https://arxiv.org/abs/2107.05407 — PonderNet
- https://arxiv.org/abs/1603.08983 — Adaptive Computation Time
- https://arxiv.org/abs/2405.16039 — MoEUT
- https://arxiv.org/abs/2403.07965 — Conditional computation survey
- https://github.com/raymin0223/mixture_of_recursions — MoR code

_Seed section expanded from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
