# Anomaly and OOD Detection

> Detecting inputs that deviate from the training distribution — outliers, novel classes, distribution shift — so models can flag, abstain, or escalate instead of silently failing.

## Why it matters

Deployed models are only trustworthy if they know what they do *not* know. Anomaly detection underpins fraud detection, industrial inspection, network/security monitoring, and health diagnostics, while out-of-distribution (OOD) detection is a prerequisite for safe deployment: a classifier that confidently mislabels an unseen class is a silent liability. These tasks span unsupervised outlier detection, one-class learning, and post-hoc OOD scoring on top of pretrained networks, and they sit at the intersection of reliability, uncertainty estimation, and safety.

## Taxonomy

| Sub-area | What it targets | Representative approaches |
|---|---|---|
| **Outlier / unsupervised AD** | Rare points in unlabeled tabular/streaming data | Isolation Forest, LOF, ECOD, kNN, OC-SVM |
| **One-class / deep AD** | Learn a boundary around "normal" data | Deep SVDD, Deep SAD, DROCC, autoencoders |
| **Image / industrial AD** | Pixel- and patch-level defects, localization | PatchCore, PaDiM, FastFlow, EfficientAD |
| **Time-series AD** | Anomalous segments in multivariate streams | Anomaly Transformer, LSTM-AE, USAD, Matrix Profile |
| **OOD detection (post-hoc)** | Reject novel-class / shifted inputs at test time | MSP, ODIN, Energy, Mahalanobis, ViM, KNN, ReAct, DICE, ASH |
| **OOD (training-time)** | Shape representations for separability | Outlier Exposure, LogitNorm, VOS, contrastive (CSI/SSD) |
| **Near- vs far-OOD** | Semantic shift (new class) vs covariate shift | Full-spectrum benchmarks (OpenOOD v1.5) |

A useful mental model: **AD** typically assumes only normal data at train time; **OOD detection** assumes a labeled in-distribution (ID) classifier and asks whether a new input belongs to *any* ID class. Many scores (Energy, Mahalanobis, KNN) apply to both.

## Key methods

### OOD detection scores

| Method | Idea | Link |
|---|---|---|
| MSP (Hendrycks & Gimpel) | Max softmax probability baseline | https://arxiv.org/abs/1610.02136 |
| ODIN | Temperature scaling + input perturbation | https://arxiv.org/abs/1706.02690 |
| Mahalanobis (Lee et al.) | Class-conditional Gaussian distance in feature space | https://arxiv.org/abs/1807.03888 |
| Outlier Exposure | Train against auxiliary outliers | https://arxiv.org/abs/1812.04606 |
| Energy-based OOD (Liu et al.) | Free-energy score, theoretically aligned with density | https://arxiv.org/abs/2010.03759 |
| ReAct | Rectify (clip) penultimate activations | https://arxiv.org/abs/2111.12797 |
| ViM | Virtual-logit matching: logits + feature residual norm | https://arxiv.org/abs/2203.10807 |
| KNN-OOD (Sun et al.) | Non-parametric kth-NN distance on embeddings | https://arxiv.org/abs/2204.06507 |
| DICE | Directed sparsification of the classification weights | https://arxiv.org/abs/2111.09805 |
| LogitNorm | Constant-norm logits to curb overconfidence | https://arxiv.org/abs/2205.09310 |

### Anomaly detection methods

| Method | Domain | Link |
|---|---|---|
| Isolation Forest (Liu et al., ICDM'08) | Tabular, fast unsupervised | https://ieeexplore.ieee.org/document/4781136 |
| Deep SVDD (Ruff et al., ICML'18) | One-class deep AD | https://proceedings.mlr.press/v80/ruff18a.html |
| Deep SAD (Ruff et al., ICLR'20) | Semi-supervised deep AD | https://arxiv.org/abs/1906.02694 |
| PaDiM | Patch-distribution image AD + localization | https://arxiv.org/abs/2011.08785 |
| PatchCore | Memory-bank coreset, SOTA industrial AD | https://arxiv.org/abs/2106.08265 |
| EfficientAD | Millisecond-latency image AD | https://arxiv.org/abs/2303.14535 |
| USAD | Unsupervised adversarial AE for multivariate time series | https://dl.acm.org/doi/10.1145/3394486.3403392 |
| Anomaly Transformer | Association discrepancy for time-series AD | https://arxiv.org/abs/2110.02642 |

## Tools & libraries

| Tool | Scope | Link |
|---|---|---|
| PyOD | 50+ classical & deep outlier detectors (tabular) | https://github.com/yzhao062/pyod |
| PyGOD | Graph outlier detection | https://github.com/pygod-team/pygod |
| TODS / TSB-AD | Time-series outlier/anomaly detection | https://github.com/datamllab/tods |
| anomalib (OpenVINO) | Image/industrial AD, edge deployment | https://github.com/open-edge-platform/anomalib |
| OpenOOD | Unified benchmark, 50+ OOD methods, leaderboard | https://github.com/Jingkang50/OpenOOD |
| pytorch-ood | Library of OOD detectors & loss functions | https://github.com/kkirchheim/pytorch-ood |
| ADBench | Tabular AD benchmark (57 datasets) | https://github.com/Minqi824/ADBench |
| DeepOD | Deep AD for tabular & time series | https://github.com/xuhongzuo/DeepOD |

## Benchmarks & datasets

| Benchmark | Task | Link |
|---|---|---|
| OpenOOD v1.5 | Generalized OOD (near/far, full-spectrum, ImageNet) | https://arxiv.org/abs/2306.09301 |
| MVTec AD | Industrial image anomaly + segmentation | https://www.mvtec.com/company/research/datasets/mvtec-ad |
| VisA | Visual anomaly, 12 object categories | https://github.com/amazon-science/spot-diff |
| ADBench | 57 tabular AD datasets, 30 algorithms | https://github.com/Minqi824/ADBench |
| TSB-AD | Time-series AD benchmark | https://github.com/TheDatumOrg/TSB-AD |
| CIFAR/ImageNet OOD suites | Standard ID/OOD splits (SVHN, Textures, iNaturalist, etc.) | https://github.com/Jingkang50/OpenOOD |

**Common metrics:** AUROC, AUPR, FPR@95 (FPR at 95% TPR) for OOD; image/pixel AUROC and PRO for localized image AD; VUS-PR/affiliation metrics for time series.

## Key papers

- Hendrycks & Gimpel (2016) — *A Baseline for Detecting Misclassified and OOD Examples* — https://arxiv.org/abs/1610.02136
- Liang et al. (2017) — *ODIN: Enhancing Reliability of OOD Detection* — https://arxiv.org/abs/1706.02690
- Lee et al. (2018) — *A Simple Unified Framework for Detecting OOD (Mahalanobis)* — https://arxiv.org/abs/1807.03888
- Ruff et al. (2018) — *Deep One-Class Classification (Deep SVDD)* — https://proceedings.mlr.press/v80/ruff18a.html
- Liu et al. (2020) — *Energy-based Out-of-Distribution Detection* — https://arxiv.org/abs/2010.03759
- Ruff et al. (2021) — *A Unifying Review of Deep and Shallow Anomaly Detection* — https://arxiv.org/abs/2009.11732
- Pang et al. (2021) — *Deep Learning for Anomaly Detection: A Review* — https://arxiv.org/abs/2007.02500
- Sun et al. (2022) — *Out-of-Distribution Detection with Deep Nearest Neighbors* — https://arxiv.org/abs/2204.06507
- Yang et al. (2023) — *OpenOOD v1.5: Enhanced Benchmark for OOD Detection* — https://arxiv.org/abs/2306.09301
- Lu et al. (2025) — *Out-of-Distribution Detection: A Task-Oriented Survey of Recent Advances* (ACM CSUR) — https://github.com/shuolucs/Awesome-Out-Of-Distribution-Detection

## Cross-references in AIForge

- [Uncertainty Quantification](../Uncertainty_Quantification/README.md) — calibration & confidence that complement OOD scores
- [AI Safety and Alignment](../AI_Safety_and_Alignment/README.md) — reliable abstention and failure detection
- [Bayesian and Probabilistic ML](../Bayesian_and_Probabilistic_ML/README.md) — density estimation foundations for novelty detection
- [Domain Adaptation — covariate shift, the sibling of semantic OOD

## Sources

- OpenOOD repo & benchmark — https://github.com/Jingkang50/OpenOOD
- OpenOOD v1.5 paper — https://arxiv.org/abs/2306.09301
- Awesome OOD Detection / CSUR survey — https://github.com/shuolucs/Awesome-Out-Of-Distribution-Detection
- PyOD — https://github.com/yzhao062/pyod
- anomalib — https://github.com/open-edge-platform/anomalib
- ADBench — https://github.com/Minqi824/ADBench
- Energy-based OOD — https://arxiv.org/abs/2010.03759
- ViM (CVPR'22) — https://arxiv.org/abs/2203.10807
- KNN-OOD (ICML'22) — https://arxiv.org/abs/2204.06507
- Anomalib paper — https://arxiv.org/abs/2202.08341

_Expanded from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
