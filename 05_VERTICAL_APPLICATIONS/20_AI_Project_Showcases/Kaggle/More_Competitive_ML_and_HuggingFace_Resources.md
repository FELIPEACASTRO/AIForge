# More Competitive ML & HuggingFace Resources

> Additional verified resources on competitive/applied ML technique and the HuggingFace ecosystem — not previously listed.

_46 sources, each WebFetch-verified and de-duplicated against the existing section (multi-agent double-check)._

## Papers & Surveys (14)

| Resource | What it is |
|---|---|
| [TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling (arXiv 2410.24210, ICLR 2025)](https://arxiv.org/abs/2410.24210) | Introduces TabM, an MLP that imitates an implicit ensemble via parameter sharing — directly relevant to the 'cheap diverse ensemble' techniques that win tabular Kaggle competitions. |
| [TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models (arXiv 2511.08667)](https://arxiv.org/abs/2511.08667) | Latest TabPFN foundation model handling up to 50k rows / 2k features with a distillation path — an increasingly common drop-in base learner in modern tabular Kaggle ensembles. |
| [Benchmarking State-of-the-Art Gradient Boosting Algorithms for Classification (arXiv 2305.17094)](https://arxiv.org/abs/2305.17094) | Head-to-head benchmark of GBM/XGBoost/LightGBM/CatBoost with explicit hyperparameter optimization comparing randomized search vs Bayesian/TPE tuning — directly on the GBDT-tuning angle. |
| [A Review of Pseudo Labeling for Semi-Supervised Learning (arXiv 2408.07221)](https://arxiv.org/abs/2408.07221) | Focused review of pseudo-labeling for SSL — confidence thresholding, confirmation bias, and label refinement — the theory behind a staple Kaggle technique for unlabeled/test-time data. |
| [XStacking: Explanation-Guided Stacked Ensemble Learning (arXiv 2507.17650, ECML-PKDD 2025)](https://arxiv.org/abs/2507.17650) | Recent stacked-ensemble framework adding dynamic feature transformation plus Shapley explanations, evaluated on 29 datasets — a fresh research angle on the stacking technique central to Kaggle. |
| [Diverse Models, United Goal: A Comprehensive Survey of Ensemble Learning (CAAI Trans. Intelligence Technology)](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cit2.70030) | 2025 survey covering bagging, boosting, stacking and ensemble diversity — a current reference tying together the ensembling methods used across competitive ML. |
| [TabICL: A Tabular Foundation Model for In-Context Learning on Large Data](https://arxiv.org/abs/2502.05564) | 2025 tabular foundation model using in-context learning that scales to large tabular datasets, a recurring need in Kaggle tabular competitions where TabPFN-style ICL models are increasingly used. |
| [Real-TabPFN: Improving Tabular Foundation Models via Continued Pre-training With Real-World Data](https://arxiv.org/abs/2507.03971) | 2025 work improving TabPFN by continued pre-training on real-world tabular data rather than purely synthetic priors — directly relevant to applying tabular foundation models to competition datasets. |
| [LightAutoDS-Tab: Multi-AutoML Agentic System for Tabular Data](https://arxiv.org/abs/2507.13413) | 2025 agentic AutoML system combining LLM code generation with multiple AutoML tools for tabular pipelines, evaluated on Kaggle data-science competitions and beating existing open-source AutoML. |
| [A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data](https://arxiv.org/abs/2407.02112) | NeurIPS 2024 paper arguing for data-centric (vs purely model-centric) evaluation of tabular ML, exposing how preprocessing and dataset choices distort benchmark conclusions — directly relevant to reproducibility/leakage in competitions. |
| [Unreflected Use of Tabular Data Repositories Can Undermine Research Quality](https://arxiv.org/abs/2503.09159) | 2025 study finding data-leakage and preprocessing-contamination issues across widely used tabular benchmark datasets (e.g. Grinsztajn suite) — a reproducibility/validation cautionary reference for tabular competition modeling. |
| [An Empirical Evaluation of Modern MLOps Frameworks](https://arxiv.org/abs/2601.20415) | Hands-on comparison of MLflow, Metaflow, Apache Airflow, and Kubeflow Pipelines across install, interoperability, instrumentation, and reproducibility on MNIST and IMDB+BERT scenarios — practitioner guidance for experiment tracking and reproducible pipelines. |
| [FineVision: Open Data Is All You Need (paper)](https://arxiv.org/abs/2510.17269) | Paper behind FineVision: curation/dedup/decontamination of 24M samples; models trained on it consistently beat existing open VLM mixtures. |
| [FineWeb2: One Pipeline to Scale Them All (multilingual pretraining data)](https://arxiv.org/abs/2506.20920) | HuggingFace paper describing the auto-adapting pipeline that produced FineWeb2, a 20TB / 5B-document multilingual pretraining dataset across 1000+ languages. |

## Survey (2)

| Resource | What it is |
|---|---|
| [Representation Learning for Tabular Data: A Comprehensive Survey](https://arxiv.org/abs/2504.16109) | 2025 survey organizing deep tabular representation learning into specialized, transferable, and general (foundation) models, covering ensembles and multimodal tabular learning — the dominant data type in Kaggle tabular competitions. |
| [A Systematic Review of MLOps Tools: Tool Adoption, Lifecycle Coverage, and Critical Insights](https://arxiv.org/abs/2604.16371) | Systematic review (CAIN '26) mapping MLOps tools across the lifecycle, finding orchestration, data versioning, experiment tracking, and managed cloud platforms most adopted, and that no single tool covers the full lifecycle. Grounds the experiment-tracking/feature-store/reproducibility tooling landscape. |

## Benchmarks & Datasets (8)

| Resource | What it is |
|---|---|
| [TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks (arXiv 2406.19380, ICLR 2025)](https://arxiv.org/abs/2406.19380) | Eight industry-grade tabular datasets with time-based splits; shows method rankings change under temporal vs random CV — a core lesson for designing leak-free cross-validation in real competitions. |
| [TabArena: A Living Benchmark for Machine Learning on Tabular Data (arXiv 2506.16791)](https://arxiv.org/abs/2506.16791) | First continuously maintained living tabular benchmark with a public leaderboard and standardized protocols across GBDTs, deep nets and foundation models — a reference for picking base learners. |
| [MultiTab: A Comprehensive Benchmark Suite for Multi-Dimensional Evaluation in Tabular Domains (arXiv 2505.14312)](https://arxiv.org/abs/2505.14312) | 2025 benchmark suite contrasting GBDTs (XGBoost/CatBoost/LightGBM) and tabular deep models across dataset dimensions, useful for understanding when boosting still dominates. |
| [MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering](https://arxiv.org/abs/2410.07095) | OpenAI benchmark of 75 real Kaggle competitions used to evaluate how well AI agents perform end-to-end ML engineering (data prep, training, experiments, submission). Directly bridges Kaggle competition workflows and agentic AutoML. |
| [ML2B: Multi-Lingual ML Benchmark For AutoML](https://arxiv.org/abs/2509.22768) | First multilingual AutoML/code-generation benchmark built from 30 Kaggle competitions translated into 13 languages (tabular/text/image), with the AIDE end-to-end pipeline evaluator. Quantifies 15-45% degradation on non-English tasks. |
| [AutoML Benchmark with shorter time constraints and early stopping](https://arxiv.org/abs/2504.01222) | 2025 AWS-supported extension of the AutoML Benchmark (AMLB) studying how short time budgets and early stopping affect AutoML framework rankings — practical for time-boxed competition pipelines. |
| [GAIA2 and ARE — new HuggingFace/Meta agent benchmark and environment](https://huggingface.co/blog/gaia2) | GAIA2: 1000 new human-created interactive agent scenarios (time-sensitive, ambiguity, collaboration) plus ARE framework for running/evaluating agents. |
| [Holistic Agent Leaderboard (HAL) — missing infrastructure for AI agent eval](https://arxiv.org/abs/2510.11977) | Standardized parallel agent-evaluation harness; ~22,000 rollouts across 9 models and 9 benchmarks with 2.5B tokens of logs released for the community. |

## Datasets (3)

| Resource | What it is |
|---|---|
| [Mixture-of-Thoughts — 350k verified R1 reasoning traces dataset](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts) | Curated 350k reasoning-trace dataset distilled from DeepSeek-R1 across math (93.7k), code (83.1k) and science (173k); used to train OpenR1-Distill-7B. |
| [OpenR1-Math-220k — large-scale math reasoning dataset](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | 220k math problems with 2-4 R1-generated reasoning traces (from NuminaMath 1.5), released Feb 2025 as part of Open R1. |
| [FineVision — 24M-sample open multimodal dataset (HuggingFace M4)](https://huggingface.co/datasets/HuggingFaceM4/FineVision) | Largest open VLM training corpus: 17.3M images / 24.3M samples / 88.9M turns, 200+ unified sources, decontaminated; boosts VLMs up to 46% over LLaVA mixtures. |

## Tools & Libraries (12)

| Resource | What it is |
|---|---|
| [Stacking vs Hill-climbing: Boosters + Ridge + NN (Kaggle notebook)](https://www.kaggle.com/code/rzatemizel/stacking-vs-hill-climbing-boosters-ridge-nn) | Hands-on Kaggle notebook directly comparing OOF stacking vs greedy hill-climbing blending across GBDTs, Ridge and neural nets — a practical reference implementation of the two dominant ensembling-of-ensembles techniques. |
| [PS S3E3 - Hill Climbing like a GM (Kaggle notebook by Samuel Cortinhas)](https://www.kaggle.com/code/samuelcortinhas/ps-s3e3-hill-climbing-like-a-gm) | Tutorial notebook walking through grandmaster-style greedy hill-climbing ensemble weight search on Playground data, a step-by-step companion to the hillclimbers package. |
| [compsearch.dev — Kaggle Solution Search engine](https://compsearch.dev/) | Search engine indexing Kaggle competition solution write-ups and summaries by technique/competition — a discovery tool complementing the repo's static solution catalogs. |
| [openai/mle-bench (GitHub repository)](https://github.com/openai/mle-bench) | Reference implementation and harness for MLE-bench: lets you run AI agents against 75 offline Kaggle competitions with grading against medal thresholds. Practical tooling for competitive-ML agent evaluation and reproducibility. |
| [autogluon/tabarena (GitHub repository)](https://github.com/autogluon/tabarena) | Open-source codebase/leaderboard backing TabArena, the living tabular ML benchmark from the AutoGluon team. Tooling for reproducible benchmarking of tabular models used in competitions. |
| [OpenFE: Automated Feature Generation with Expert-level Performance](https://arxiv.org/abs/2211.12507) | Automated feature generation framework (feature boosting + two-stage pruning) whose generated features with a simple model beat 99.3-99.6% of teams in two large Kaggle competitions. Open-source feature-engineering tooling for competitive tabular ML. |
| [AutoGluon-Multimodal (AutoMM): Supercharging Multimodal AutoML with Foundation Models](https://arxiv.org/abs/2404.16233) | Paper describing AutoGluon's multimodal AutoML module (image+text+tabular) built on foundation models — a widely used competition-grade AutoML toolkit for mixed-modality Kaggle datasets. |
| [lighteval — HuggingFace all-in-one LLM evaluation toolkit](https://github.com/huggingface/lighteval) | HuggingFace's evaluation framework supporting 1000+ tasks across multiple inference backends; powers Open R1 and HF leaderboards. v0.13.0 released Nov 24 2025. |
| [datatrove — HuggingFace large-scale text data processing library](https://github.com/huggingface/datatrove) | Platform-agnostic pipeline library to process, filter and deduplicate trillion-token text corpora (local/Slurm/Ray); the engine behind FineWeb. |
| [Trackio — lightweight open-source experiment tracking from HuggingFace](https://huggingface.co/blog/trackio) | Local-first, <1000-line drop-in wandb replacement with Gradio dashboards and HF Spaces sync; native Transformers/Accelerate integration. |
| [Open R1 — fully open reproduction of DeepSeek-R1 (HuggingFace)](https://github.com/huggingface/open-r1) | HF project reproducing DeepSeek-R1: SFT/GRPO training scripts, Distilabel data generation, lighteval-based eval, with IOI/CodeForces sandboxes. |
| [Meta Agents Research Environments (ARE) — open agent eval framework](https://github.com/facebookresearch/meta-agents-research-environments) | Open framework simulating dynamic real-world tasks for multi-step agents; runs the GAIA2 benchmark with 800+ scenarios. |

## Competitions (2)

| Resource | What it is |
|---|---|
| [The State of Machine Learning Competitions 2025 (ML Contests)](https://mlcontests.com/state-of-machine-learning-competitions-2025/) | Annual report analyzing 390+ ML competitions (>$16M prizes) across 30+ platforms incl. Kaggle/Tianchi/Codabench; winning methods, tools and Qwen/GBDT trends. |
| [Google Tunix Hackathon — train a Gemma model to show its work (Kaggle)](https://www.kaggle.com/competitions/google-tunix-hackathon) | Kaggle competition to fine-tune open Gemma (2B/3B) with Google's Tunix on TPU for reasoning; final deadline Jan 12 2026, results Jan 26 2026. |

## Courses (1)

| Resource | What it is |
|---|---|
| [The Kaggle Book (2nd ed) Ch.10/12 — Ensembling with Blending and Stacking Solutions (O'Reilly)](https://www.oreilly.com/library/view/the-kaggle-book/9781835083208/Text/Chapter_10.xhtml) | Authoritative reference chapter by Banachewicz & Massaron on blending and stacking strategy, level-1/level-2 model design and avoiding leakage in stacked generalization. |

## Guides & Blogs (4)

| Resource | What it is |
|---|---|
| [Kaggle Grandmasters Unveil Winning Strategies for Data Science Superpowers (NVIDIA Technical Blog)](https://developer.nvidia.com/blog/kaggle-grandmasters-unveil-winning-strategies-for-data-science-superpowers/) | NVIDIA blog with Grandmasters Chris Deotte, David Austin and Ruchi Bhatia on their winning playbook: robust local validation, simulating public/private LB split with multiple CV folds to anticipate shakeups, target transforms, and creative feature engineering. |
| [Winning a Kaggle Competition with Generative AI-Assisted Coding (NVIDIA Technical Blog)](https://developer.nvidia.com/blog/winning-a-kaggle-competition-with-generative-ai-assisted-coding/) | Chris Deotte (4x GM) write-up on a 1st-place Playground finish where LLM agents ran 850 experiments and combined ~150 models into a four-level stack; details the EDA->baseline->FE->stacking workflow with cuDF/cuML. |
| [Inside the Mind of the Ultimate Kaggle Grandmaster — Abhishek Thakur (KNN Ep.10, YouTube)](https://www.youtube.com/watch?v=5v_ulmGv-gc) | Long-form video interview with the world's first quadruple Kaggle Grandmaster on his competitive ML workflow, validation discipline and approach to new competitions. |
| [What it takes to become a Quadruple Kaggle Grandmaster (Panel: Deotte, Rao, Tunguz — YouTube)](https://www.youtube.com/watch?v=d7YvGX3oi3A) | Panel talk with three top Grandmasters discussing competition strategy, team formation, ensembling and how they reach the top of leaderboards across modalities. |

