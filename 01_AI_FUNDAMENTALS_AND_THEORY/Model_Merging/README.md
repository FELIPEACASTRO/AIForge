# Model Merging

> Model merging combines the weights of multiple separately-trained checkpoints (usually fine-tunes of a shared base model) into a single model — without any gradient training, extra inference cost, or access to the original training data.

## Why it matters

Merging is a training-free way to fuse capabilities: you can blend a math fine-tune, a coding fine-tune, and a chat fine-tune into one checkpoint that keeps much of each skill. It powers a large fraction of top Open LLM Leaderboard models and is a standard production technique for building multi-task or domain-specialized models cheaply. Because it operates only on weights, it costs minutes on a single machine instead of GPU-days, and the merged model has the same size and latency as its parents.

## Taxonomy

| Family | Idea | Representative methods |
|---|---|---|
| **Weight averaging** | Interpolate parameters of models from the same base | Model Soups, Linear, SLERP, Model Stock |
| **Task-vector arithmetic** | Treat `fine-tuned − base` as a "task vector"; add/subtract/scale | Task Arithmetic, TIES, DARE, Model Breadcrumbs |
| **Statistics / activation-aware** | Weight the merge using Fisher info or input covariances | Fisher Merging, RegMean |
| **Layer stacking (frankenmerge)** | Concatenate layers from different models into a larger net | Passthrough / FrankenMerging |
| **Learned / optimized merges** | Search or train merge coefficients | Evolutionary Merging, SCE, Arcee Fusion, WEMoE |
| **Routed / MoE merges** | Keep critical modules as a mixture of experts | WEMoE / E-WEMoE |

## Key methods

| Method | One-liner | Link |
|---|---|---|
| Model Soups | Average weights of many fine-tunes for robustness/accuracy gains | https://arxiv.org/abs/2203.05482 |
| Task Arithmetic | Edit models by adding/negating task vectors (`θ_ft − θ_base`) | https://arxiv.org/abs/2212.04089 |
| TIES-Merging | Trim, elect sign, disjoint-merge to resolve parameter interference | https://arxiv.org/abs/2306.01708 |
| DARE | Drop-and-rescale task deltas before merging (Super Mario) | https://arxiv.org/abs/2311.03099 |
| Model Breadcrumbs | Sparse masks removing outliers + small noise from task vectors | https://arxiv.org/abs/2312.06795 |
| Model Stock | Layer-wise averaging that beats soups with just 2 fine-tunes | https://arxiv.org/abs/2403.19522 |
| Fisher Merging | Weight params by diagonal Fisher information | https://arxiv.org/abs/2111.09832 |
| RegMean | Closed-form regression mean using input activation covariances | https://arxiv.org/abs/2212.09849 |
| Evolutionary Merging | CMA-ES search over merge recipes & layer permutations | https://arxiv.org/abs/2403.13187 |
| WEMoE / E-WEMoE | Static-merge non-critical modules, MoE-route critical ones | https://arxiv.org/abs/2410.21804 |
| SLERP | Spherical interpolation preserving weight norm (two models) | https://github.com/arcee-ai/mergekit/blob/main/docs/merge_methods.md |

## Key tools & frameworks

| Tool | What it does | Link |
|---|---|---|
| mergekit (Arcee) | De-facto toolkit; supports Linear, SLERP, TIES, DARE, SCE, Passthrough, Arcee Fusion, MoE | https://github.com/arcee-ai/mergekit |
| mergekit-evolve | Evolutionary optimization of merge recipes (Sakana-style) | https://github.com/arcee-ai/mergekit/blob/main/docs/evolve.md |
| Arcee Fusion | Importance-aware merge method in mergekit v0.1+ | https://www.arcee.ai/blog/meet-mergekit-v0-1-arcee-fusion-expanded-model-support-multi-gpu-acceleration |
| FusionBench | Benchmark suite for deep model fusion / merging | https://github.com/tanganke/fusion_bench |
| Efficient-WEMoE | Reference impl of weight-ensembling MoE merging | https://github.com/EnnengYang/Efficient-WEMoE |
| Awesome-Model-Merging | Curated paper/code list tracking the field | https://github.com/EnnengYang/Awesome-Model-Merging-Methods-Theories-Applications |

## Benchmarks & evaluation

- **FusionBench** — standardized tasks/datasets for comparing fusion and merging algorithms: https://github.com/tanganke/fusion_bench
- **Open LLM Leaderboard** — merged models frequently top general LLM evals (MMLU, GSM8K, etc.): https://huggingface.co/open-llm-leaderboard
- Merging is commonly evaluated on the parents' own task suites (vision: 8-task ViT benchmark from Task Arithmetic; LLMs: math/code/chat benchmarks) to measure retained per-task accuracy vs. interference.

## Key papers

| Year | Paper | Link |
|---|---|---|
| 2022 | Model Soups: averaging weights of multiple fine-tuned models | https://arxiv.org/abs/2203.05482 |
| 2022 | Editing Models with Task Arithmetic | https://arxiv.org/abs/2212.04089 |
| 2022 | Dataless Knowledge Fusion by Merging Weights (RegMean) | https://arxiv.org/abs/2212.09849 |
| 2023 | TIES-Merging: Resolving Interference When Merging Models | https://arxiv.org/abs/2306.01708 |
| 2023 | Language Models are Super Mario (DARE) | https://arxiv.org/abs/2311.03099 |
| 2023 | Model Breadcrumbs: Scaling Multi-Task Merging with Sparse Masks | https://arxiv.org/abs/2312.06795 |
| 2024 | Arcee's MergeKit: A Toolkit for Merging LLMs (EMNLP industry) | https://arxiv.org/abs/2403.13257 |
| 2024 | Evolutionary Optimization of Model Merging Recipes (Sakana AI) | https://arxiv.org/abs/2403.13187 |
| 2024 | Model Stock: All We Need Is Just a Few Fine-Tuned Models (ECCV) | https://arxiv.org/abs/2403.19522 |
| 2024 | Model Merging in LLMs, MLLMs, and Beyond: a comprehensive survey | https://arxiv.org/abs/2408.07666 |

## Cross-references in AIForge

- [Modern Fine-Tuning](../Modern_Fine_Tuning/README.md) — merging consumes fine-tuned checkpoints; LoRA adapters can also be merged
- [Transfer Learning — task vectors generalize the transfer-learning view of fine-tuning
- [Domain Adaptation — merging is a training-free path to multi-domain models
- [Deep Learning — loss-landscape / linear-mode-connectivity intuitions underpin why averaging works

## Sources

- mergekit (Arcee) — https://github.com/arcee-ai/mergekit
- mergekit merge methods docs — https://github.com/arcee-ai/mergekit/blob/main/docs/merge_methods.md
- Arcee's MergeKit paper — https://arxiv.org/abs/2403.13257
- Model Merging survey (LLMs/MLLMs and Beyond) — https://arxiv.org/abs/2408.07666
- A Review of Model Merging Approaches — https://arxiv.org/html/2503.08998v1
- Awesome-Model-Merging list — https://github.com/EnnengYang/Awesome-Model-Merging-Methods-Theories-Applications
- FusionBench — https://github.com/tanganke/fusion_bench

_Seed section expanded from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
