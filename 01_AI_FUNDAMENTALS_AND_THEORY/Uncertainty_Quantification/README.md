# Uncertainty Quantification

> Methods that attach calibrated confidence to model predictions — calibration, conformal prediction, Bayesian/ensemble methods, and evidential learning — so systems know when (and how much) they might be wrong.

## Why it matters

Point predictions hide risk: a model that is 51% sure and one that is 99% sure look identical until you quantify uncertainty. Calibrated, distribution-aware uncertainty is what makes ML usable for high-stakes decisions (medicine, autonomy, finance), enables abstention and human-in-the-loop routing, and underpins out-of-distribution detection and active learning. It is also central to trustworthy/production ML and to detecting LLM hallucination — yet remains under-indexed across most ML curricula.

## Taxonomy

Uncertainty is conventionally split by **source** and addressed by complementary **method families**.

| Axis | Type | Meaning | Reducible? |
|---|---|---|---|
| Source | **Aleatoric** (data) | Inherent noise/ambiguity in the data (sensor noise, label conflict) | No — not by more data |
| Source | **Epistemic** (model) | Lack of knowledge: limited data, model misspecification, OOD inputs | Yes — with more data/better models |

| Method family | Idea | Representative approaches |
|---|---|---|
| **Calibration (post-hoc)** | Adjust output probabilities to match empirical frequencies | Temperature scaling, Platt/isotonic, ECE diagnostics |
| **Conformal prediction** | Distribution-free prediction sets/intervals with finite-sample coverage guarantees | Split/inductive CP, CQR, RAPS, APS |
| **Bayesian deep learning** | Posterior over weights | Variational inference, Laplace approximation, SGLD/SG-MCMC |
| **Ensembles & sampling** | Empirical posterior via multiple models / stochastic passes | Deep ensembles, MC-dropout, snapshot ensembles |
| **Evidential / single-pass** | Predict a distribution over distributions in one forward pass | Evidential DL (Dirichlet), Deep Evidential Regression, DUQ |

## Key methods & tools

| Tool / Library | Focus | Link |
|---|---|---|
| MAPIE | Conformal prediction (scikit-learn-compatible) | https://github.com/scikit-learn-contrib/MAPIE |
| TorchCP | PyTorch conformal prediction (classifiers, GNNs, LLMs) | https://github.com/ml-stat-Sustech/TorchCP |
| crepes | Conformal classifiers, regressors, predictive systems | https://github.com/henrikbostrom/crepes |
| Fortuna (AWS) | Calibration + conformal + Bayesian inference, framework-agnostic | https://github.com/awslabs/fortuna |
| Laplace (laplace-torch) | Post-hoc Laplace approximation for Bayesian DL | https://github.com/aleximmer/Laplace |
| Uncertainty Baselines (Google) | SOTA UQ/robustness method implementations | https://github.com/google/uncertainty-baselines |
| Lightning-UQ-Box | UQ methods on PyTorch Lightning (cls/reg/seg) | https://github.com/lightning-uq-box/lightning-uq-box |
| Uncertainty Toolbox | Metrics, calibration, visualization for regression UQ | https://github.com/uncertainty-toolbox/uncertainty-toolbox |
| UQLM | UQ-based hallucination detection for LLMs | https://github.com/cvs-health/uqlm |
| awesome-uncertainty-deeplearning | Curated surveys/datasets/papers/code | https://github.com/ENSTA-U2IS-AI/awesome-uncertainty-deeplearning |

## Benchmarks & datasets

| Benchmark | Use | Link |
|---|---|---|
| CIFAR-10-C / CIFAR-100-C / ImageNet-C | Calibration & uncertainty under corruption/shift | https://github.com/hendrycks/robustness |
| Shifts Dataset | Real distributional shift across large-scale tasks | https://arxiv.org/abs/2107.07455 |
| Uncertainty Baselines (datasets+tasks) | Standardized UQ/robustness evaluation | https://github.com/google/uncertainty-baselines |
| Common metrics | ECE/MCE, NLL, Brier score, AUROC/AUPR (OOD), coverage & set size (CP) | — |

## Key papers

| Paper | Year | Link |
|---|---|---|
| On Calibration of Modern Neural Networks (Guo et al.) | 2017 | https://arxiv.org/abs/1706.04599 |
| Simple and Scalable Predictive Uncertainty via Deep Ensembles (Lakshminarayanan et al.) | 2016 | https://arxiv.org/abs/1612.01474 |
| Dropout as a Bayesian Approximation — MC-dropout (Gal & Ghahramani) | 2015 | https://arxiv.org/abs/1506.02142 |
| Evidential Deep Learning to Quantify Classification Uncertainty (Sensoy et al.) | 2018 | https://arxiv.org/abs/1806.01768 |
| Can You Trust Your Model's Uncertainty? Evaluating Under Dataset Shift (Ovadia et al.) | 2019 | https://arxiv.org/abs/1906.02530 |
| A Gentle Introduction to Conformal Prediction (Angelopoulos & Bates) | 2021 | https://arxiv.org/abs/2107.07511 |
| Laplace Redux — Effortless Bayesian Deep Learning (Daxberger et al.) | 2021 | https://arxiv.org/abs/2106.14806 |
| A Survey of Uncertainty in Deep Neural Networks (Gawlikowski et al.) | 2021 | https://arxiv.org/abs/2107.03342 |
| A Comprehensive Survey on UQ for Deep Learning (He et al.) | 2023 | https://arxiv.org/abs/2302.13425 |
| UQ and Confidence Calibration in LLMs: A Survey | 2025 | https://arxiv.org/abs/2503.15850 |

## Cross-references in AIForge

- [../Bayesian_and_Probabilistic_ML/](../Bayesian_and_Probabilistic_ML/) — posteriors, variational inference, probabilistic modeling foundations
- [../Anomaly_and_OOD_Detection/](../Anomaly_and_OOD_Detection/) — out-of-distribution detection, a primary downstream use of UQ
- [../Active_Learning/](../Active_Learning/) — uncertainty-driven sample acquisition
- [../AI_Safety_and_Alignment/](../AI_Safety_and_Alignment/) — trustworthy/robust ML and abstention

## Sources

- https://arxiv.org/abs/2402.12683 — TorchCP: A Python Library for Conformal Prediction
- https://arxiv.org/abs/2410.03390 — Lightning UQ Box framework
- https://github.com/awslabs/fortuna — Fortuna (AWS) UQ library
- https://github.com/aleximmer/Laplace — Laplace approximations for deep learning
- https://github.com/google/uncertainty-baselines — Uncertainty Baselines
- https://github.com/ENSTA-U2IS-AI/awesome-uncertainty-deeplearning — curated UQ resource list
- https://arxiv.org/abs/2302.13425 — A Comprehensive Survey on UQ for Deep Learning
- https://arxiv.org/abs/1906.02530 — Evaluating Predictive Uncertainty Under Dataset Shift

_Seed section — curated from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
