# Tabular Deep Learning

> Deep learning applied to structured/tabular data (rows of heterogeneous numerical and categorical features) — the most common real-world ML modality — and the long-running debate over whether neural nets can match gradient-boosted decision trees (GBDTs).

## Why it matters

Tabular data dominates industry ML (finance, healthcare, churn, fraud, scientific tables), yet for years tuned GBDTs (XGBoost, LightGBM, CatBoost) beat neural nets on it. Since ~2021 a wave of architectures (FT-Transformer, SAINT, NODE, TabM) and **tabular foundation models** (TabPFN, TabICL, TabDPT) has closed much of the gap, with TabPFN v2 published in *Nature* (2025). Understanding when to reach for trees vs. deep nets vs. in-context foundation models is now a core practical skill, and benchmarks like TabArena make the comparison reproducible.

## Taxonomy

| Family | Idea | Representative models |
|---|---|---|
| **Gradient-boosted trees (baseline)** | Ensembles of decision trees; still the default | XGBoost, LightGBM, CatBoost |
| **MLP / regularized MLP** | Plain feed-forward nets with strong tuning tricks | MLP, RealMLP, TabM |
| **Differentiable trees** | Neural nets mimicking oblivious decision tree ensembles | NODE, GANDALF |
| **Attention / Transformer** | Feature tokenization + self-attention | TabTransformer, FT-Transformer, SAINT, AutoInt |
| **Retrieval / neighbor-based** | Predict by attending to retrieved training rows | TabR, ModernNCA |
| **In-context foundation models** | Pre-trained on synthetic priors; no per-task training | TabPFN, TabPFN v2, TabICL, TabDPT |
| **Interpretable / attentive** | Built-in feature selection & sparsity | TabNet, GANDALF |

## Key models

| Model | Year | Type | Link |
|---|---|---|---|
| TabNet | 2019 | Attentive, interpretable, sequential feature selection | https://arxiv.org/abs/1908.07442 |
| NODE | 2019 | Neural Oblivious Decision Ensembles | https://arxiv.org/abs/1909.06312 |
| TabTransformer | 2020 | Transformer over categorical embeddings | https://arxiv.org/abs/2012.06678 |
| FT-Transformer | 2021 | Feature Tokenizer + Transformer (strong DL baseline) | https://arxiv.org/abs/2106.11959 |
| SAINT | 2021 | Self-attention + intersample attention + contrastive pretrain | https://arxiv.org/abs/2106.01342 |
| GANDALF | 2022 | Gated feature-learning units, efficient & interpretable | https://arxiv.org/abs/2207.08548 |
| TabPFN | 2022 | In-context Prior-Data Fitted Network for small data | https://arxiv.org/abs/2207.01848 |
| TabR | 2023 | Retrieval-augmented deep tabular model | https://arxiv.org/abs/2307.14338 |
| RealMLP | 2024 | Tuned MLP that rivals GBDTs | https://arxiv.org/abs/2407.04491 |
| TabM | 2024 | Parameter-efficient MLP ensemble (BatchEnsemble) | https://arxiv.org/abs/2410.24210 |
| TabDPT | 2024 | Foundation model scaled on real data | https://arxiv.org/abs/2410.18164 |
| TabICL | 2025 | In-context foundation model for large data | https://arxiv.org/abs/2502.05564 |
| TabPFN v2 | 2025 | *Nature* tabular foundation model (clf + reg) | https://www.nature.com/articles/s41586-024-08328-6 |
| TabPFN-2.5 | 2025 | Scaled successor, broader data regimes | https://arxiv.org/abs/2511.08667 |

## Key tools & frameworks

| Tool | What it is | Link |
|---|---|---|
| pytorch-tabular | High-level library (FT-Transformer, TabNet, GANDALF, NODE…) | https://github.com/manujosephv/pytorch_tabular |
| RTDL (`rtdl_revisiting_models`) | Reference impls of FT-Transformer / ResNet / MLP | https://github.com/yandex-research/rtdl |
| TabPFN | Official package + HF weights (Prior Labs) | https://github.com/PriorLabs/TabPFN |
| TabM (official) | Authors' reference implementation | https://github.com/yandex-research/tabm |
| AutoGluon (Tabular) | AutoML stacking GBDTs + DL + foundation models | https://github.com/autogluon/autogluon |
| TabArena | Living benchmark harness + leaderboard | https://github.com/autogluon/tabarena |
| pytorch-frame | Modular deep tabular framework (PyG team) | https://github.com/pyg-team/pytorch-frame |

## Benchmarks & datasets

| Benchmark | Scope | Link |
|---|---|---|
| TabArena | Living, curated 51-dataset benchmark + public leaderboard (NeurIPS 2025) | https://tabarena.ai |
| Grinsztajn et al. benchmark | The "why trees still win" suite (numerical/categorical, mid-size) | https://arxiv.org/abs/2207.08815 |
| OpenML-CC18 / AMLB | Classic curated OpenML classification suites | https://www.openml.org/search?type=benchmark |
| WhyTrees datasets | Companion data for Grinsztajn et al. on HF | https://huggingface.co/datasets/inria-soda/tabular-benchmark |

**Rule of thumb (from TabArena & surveys):** tuned GBDTs remain strong defaults; deep nets (TabM, RealMLP) catch up under larger time budgets with ensembling; foundation models (TabPFN v2) lead on **small** datasets (≲10k rows, ≲500 features).

## Key papers

| Paper | Venue / Year | Link |
|---|---|---|
| TabNet: Attentive Interpretable Tabular Learning | AAAI 2021 | https://arxiv.org/abs/1908.07442 |
| Revisiting Deep Learning Models for Tabular Data (FT-Transformer) | NeurIPS 2021 | https://arxiv.org/abs/2106.11959 |
| Why do tree-based models still outperform DL on tabular data? | NeurIPS 2022 | https://arxiv.org/abs/2207.08815 |
| TabPFN: A Transformer That Solves Small Tabular Problems in a Second | ICLR 2023 | https://arxiv.org/abs/2207.01848 |
| Accurate predictions on small data with a tabular foundation model (TabPFN v2) | Nature 2025 | https://www.nature.com/articles/s41586-024-08328-6 |
| TabR: Tabular Deep Learning Meets Nearest Neighbors | ICLR 2024 | https://arxiv.org/abs/2307.14338 |
| TabM: Advancing Tabular DL with Parameter-Efficient Ensembling | ICLR 2025 | https://arxiv.org/abs/2410.24210 |
| TabICL: A Tabular Foundation Model for In-Context Learning on Large Data | 2025 | https://arxiv.org/abs/2502.05564 |
| A Survey on Deep Tabular Learning | 2024 | https://arxiv.org/abs/2410.12034 |
| TabArena: A Living Benchmark for ML on Tabular Data | NeurIPS 2025 | https://arxiv.org/abs/2506.16791 |

## Cross-references in AIForge

- [Deep Learning](../Deep_Learning/) — foundational architectures and training.
- [Machine Learning](../Machine_Learning/) — gradient boosting and classical baselines.
- [Vision Transformers](../Vision_Transformers/) — Transformer mechanics reused in FT-Transformer/SAINT.
- [Uncertainty Quantification](../Uncertainty_Quantification/) — calibration of tabular predictions.
- [Recommender Systems](../Recommender_Systems/) — a major consumer of tabular feature models.

## Sources

- https://arxiv.org/abs/2410.12034 — A Survey on Deep Tabular Learning
- https://github.com/LAMDA-Tabular/Tabular-Survey — Representation Learning for Tabular Data survey + code
- https://arxiv.org/abs/2506.16791 — TabArena
- https://github.com/autogluon/tabarena — TabArena repo
- https://www.nature.com/articles/s41586-024-08328-6 — TabPFN v2 (Nature)
- https://en.wikipedia.org/wiki/TabPFN
- https://arxiv.org/abs/2207.08815 — Grinsztajn et al. (trees vs DL)
- https://huggingface.co/datasets/inria-soda/tabular-benchmark

_Seed section — expanded from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
