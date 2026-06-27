# Information Theory

> Information theory quantifies uncertainty, compression, and the transmission of information. In ML it is load-bearing: cross-entropy / KL losses, the information bottleneck, mutual-information estimation (MINE / InfoNCE), rate-distortion, and minimum description length (MDL) all derive directly from it.

## Why it matters

Nearly every modern training objective is an information-theoretic quantity in disguise: cross-entropy minimization is equivalent to maximum-likelihood and to minimizing KL divergence to the data distribution. Representation learning, self-supervision, generative modeling, and model selection are all framed as maximizing or bounding mutual information, or as minimizing description length. Understanding entropy, KL, and rate-distortion gives a unifying lens over losses, regularizers, and generalization bounds that otherwise look unrelated.

## Taxonomy

| Sub-area | Core quantity / idea | Where it shows up in ML |
|---|---|---|
| Entropy & coding | Shannon entropy, source coding, channel capacity | Cross-entropy loss, tokenizer/compression, label noise |
| Divergences | KL, Jensen-Shannon, f-divergences | VAE ELBO, GAN objectives, distillation, RLHF KL penalty |
| Mutual information (MI) | I(X;Y), variational lower/upper bounds | Contrastive learning, InfoMax, disentanglement |
| Information bottleneck (IB) | Trade-off min I(X;T) − β I(T;Y) | Representation compression, generalization analysis |
| Rate-distortion | Optimal compression under a distortion budget | Lossy autoencoders, β-VAE, neural compression |
| MDL / Kolmogorov | Description length = model + data\|model | Model selection, generalization bounds, Occam's razor |

## Key methods (MI estimators & IB)

| Method | Idea | Link |
|---|---|---|
| MINE | Neural lower bound on MI via Donsker-Varadhan | https://arxiv.org/abs/1801.04062 |
| InfoNCE / CPC | Contrastive MI lower bound for representation learning | https://arxiv.org/abs/1807.03748 |
| Deep InfoMax (DIM) | Maximize MI between input and learned features | https://arxiv.org/abs/1808.06670 |
| Variational MI bounds | Unified view (NWJ, JS, InfoNCE, leave-one-out) | https://arxiv.org/abs/1905.06922 |
| Deep Variational IB (VIB) | Variational bound on the information bottleneck | https://arxiv.org/abs/1612.00410 |
| InfoGAN | Maximize MI between latent codes and generations | https://arxiv.org/abs/1606.03657 |
| SMILE | Stabilizing variational MI estimators (clipping) | https://arxiv.org/abs/1910.06222 |

## Tools & references

| Resource | Link |
|---|---|
| Elements of Information Theory (Cover & Thomas) — companion materials | https://onlinelibrary.wiley.com/doi/book/10.1002/047174882X |
| Information Theory, Inference & Learning Algorithms (MacKay, free PDF) | https://www.inference.org.uk/mackay/itila/ |
| Information theory (overview) — Wikipedia | https://en.wikipedia.org/wiki/Information_theory |
| Information bottleneck method — Wikipedia | https://en.wikipedia.org/wiki/Information_bottleneck_method |
| MDL / model selection portal | http://www.modelselection.org/mdl/ |

## Benchmarks

| Benchmark / study | Purpose | Link |
|---|---|---|
| Beyond Normal: On the Evaluation of MI Estimators | Standardized benchmark of MI estimators beyond Gaussian | https://arxiv.org/abs/2306.11078 |
| Understanding the Limitations of Variational MI Estimators | Variance / bias failure modes of variational bounds | https://arxiv.org/abs/1910.06222 |

## Key papers

| Year | Paper | Link |
|---|---|---|
| 1948 | A Mathematical Theory of Communication (Shannon) | https://en.wikipedia.org/wiki/A_Mathematical_Theory_of_Communication |
| 2000 | The Information Bottleneck Method (Tishby, Pereira, Bialek) | https://arxiv.org/abs/physics/0004057 |
| 2004 | A Tutorial Introduction to the MDL Principle (Grünwald) | https://arxiv.org/abs/math/0406077 |
| 2015 | Deep Learning and the Information Bottleneck Principle (Tishby & Zaslavsky) | https://arxiv.org/abs/1503.02406 |
| 2016 | Deep Variational Information Bottleneck (Alemi et al.) | https://arxiv.org/abs/1612.00410 |
| 2018 | MINE: Mutual Information Neural Estimation (Belghazi et al.) | https://arxiv.org/abs/1801.04062 |
| 2018 | Representation Learning with Contrastive Predictive Coding / InfoNCE (van den Oord et al.) | https://arxiv.org/abs/1807.03748 |
| 2019 | On Variational Bounds of Mutual Information (Poole et al.) | https://arxiv.org/abs/1905.06922 |
| 2019 | Minimum Description Length Revisited (Grünwald & Roos) | https://arxiv.org/abs/1908.08484 |
| 2024 | MDL and Generalization Guarantees for Representation Learning | https://arxiv.org/abs/2402.03254 |

## Cross-references in AIForge

- [Bayesian and Probabilistic ML](../Bayesian_and_Probabilistic_ML/) — KL divergence, ELBO, and variational inference.
- [Contrastive Learning](../Contrastive_Learning/) — InfoNCE / InfoMax objectives in practice.
- [Generative Models](../Generative_Models/) — rate-distortion, VAEs, and information-theoretic GAN objectives.
- [Optimal Transport](../Optimal_Transport/) — alternative divergences and geometry over distributions.

## Sources

- https://arxiv.org/abs/physics/0004057
- https://arxiv.org/abs/1503.02406
- https://arxiv.org/abs/1612.00410
- https://arxiv.org/abs/1801.04062
- https://arxiv.org/abs/1807.03748
- https://arxiv.org/abs/1905.06922
- https://arxiv.org/abs/2306.11078
- https://arxiv.org/abs/2402.03254
- https://en.wikipedia.org/wiki/Information_theory
- https://www.inference.org.uk/mackay/itila/

_Seed section expanded from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
