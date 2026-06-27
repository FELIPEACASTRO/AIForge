# Bayesian and Probabilistic ML

> Treating model parameters, predictions, and structure as random variables — using priors, likelihoods, and posteriors (via MCMC, variational inference, or Laplace approximations) to reason under uncertainty.

## Why it matters

Bayesian methods turn point predictions into full predictive distributions, giving calibrated uncertainty that is essential for active learning, safety-critical decisions, experimental design, and out-of-distribution detection. Probabilistic programming languages (PPLs) such as Stan, PyMC, and NumPyro make inference declarative, separating model specification from the sampler. The same machinery underpins Gaussian processes, Bayesian optimization, hierarchical models, and the diffusion/variational families now central to generative modeling.

## Taxonomy

| Sub-area | What it is | Representative methods |
|---|---|---|
| Exact / conjugate inference | Closed-form posteriors | Beta-Binomial, Normal-Normal, Dirichlet-Multinomial |
| MCMC sampling | Asymptotically exact posterior samples | Metropolis-Hastings, Gibbs, HMC, NUTS |
| Variational inference (VI) | Optimize a tractable approximation to the posterior | Mean-field VI, ADVI, BBVI, normalizing-flow VI |
| Gaussian processes (GP) | Distributions over functions; nonparametric Bayes | Exact GP, sparse/inducing-point GP, deep kernel learning |
| Bayesian neural networks (BNN) | Posteriors over network weights | Bayes by Backprop, MC Dropout, Laplace, SWAG |
| Deep ensembles | Frequentist proxy for predictive uncertainty | Deep Ensembles, batch ensembles |
| Probabilistic programming | Declarative model + generic inference engines | Stan, PyMC, Pyro/NumPyro, TFP |
| Bayesian optimization | Sample-efficient global optimization | GP-UCB, Expected Improvement, TPE |

## Key frameworks and tools

| Tool | Backend | Focus | Link |
|---|---|---|---|
| Stan | C++ | NUTS/HMC, ADVI; gold-standard PPL | https://github.com/stan-dev/stan |
| PyMC | PyTensor | Pythonic PPL, NUTS, VI | https://github.com/pymc-devs/pymc |
| Pyro | PyTorch | Deep universal probabilistic programming | https://github.com/pyro-ppl/pyro |
| NumPyro | JAX | Pyro API with fast JAX NUTS/SVI | https://github.com/pyro-ppl/numpyro |
| TensorFlow Probability | TensorFlow/JAX | Distributions, layers, Edward2 PPL | https://github.com/tensorflow/probability |
| GPyTorch | PyTorch | Scalable / GPU Gaussian processes | https://github.com/cornellius-gp/gpytorch |
| GPflow | TensorFlow | Gaussian processes, variational GPs | https://github.com/GPflow/GPflow |
| laplace (laplace-torch) | PyTorch | Post-hoc Laplace approximation for BNNs | https://github.com/aleximmer/Laplace |
| Bayesian-Torch | PyTorch | Stochastic BNN layers, uncertainty | https://github.com/IntelLabs/bayesian-torch |
| BoTorch / Ax | PyTorch | Bayesian optimization | https://github.com/pytorch/botorch |
| ArviZ | NumPy | Diagnostics & viz for Bayesian inference | https://github.com/arviz-devs/arviz |
| Bambi | PyMC | Formula-based Bayesian GLMMs | https://github.com/bambinos/bambi |

## Diagnostics and benchmarks

| Resource | Purpose | Link |
|---|---|---|
| posteriordb | Reference posteriors for benchmarking samplers | https://github.com/stan-dev/posteriordb |
| ArviZ diagnostics (R-hat, ESS) | Convergence & effective sample size | https://python.arviz.org/ |
| UCI / Boston-style regression UCI sets | Standard BNN calibration benchmarks | https://archive.ics.uci.edu/ |
| Uncertainty Baselines (Google) | Reproducible BNN/ensemble uncertainty benchmarks | https://github.com/google/uncertainty-baselines |

## Key papers

| Paper | Year | Link |
|---|---|---|
| The No-U-Turn Sampler (Hoffman & Gelman) | 2011 | https://arxiv.org/abs/1111.4246 |
| Stochastic Variational Inference (Hoffman et al.) | 2012 | https://arxiv.org/abs/1206.7051 |
| Auto-Encoding Variational Bayes / VAE (Kingma & Welling) | 2013 | https://arxiv.org/abs/1312.6114 |
| Weight Uncertainty in NNs / Bayes by Backprop (Blundell et al.) | 2015 | https://arxiv.org/abs/1505.05424 |
| Dropout as a Bayesian Approximation (Gal & Ghahramani) | 2015 | https://arxiv.org/abs/1506.02142 |
| Variational Inference: A Review for Statisticians (Blei et al.) | 2016 | https://arxiv.org/abs/1601.00670 |
| Automatic Differentiation Variational Inference (Kucukelbir et al.) | 2016 | https://arxiv.org/abs/1603.00788 |
| Simple and Scalable Predictive Uncertainty / Deep Ensembles (Lakshminarayanan et al.) | 2016 | https://arxiv.org/abs/1612.01474 |
| Laplace Redux — Effortless Bayesian Deep Learning (Daxberger et al.) | 2021 | https://arxiv.org/abs/2106.14806 |

## Key references and books

| Resource | Link |
|---|---|
| Pattern Recognition and Machine Learning (Bishop) — official resources | https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/ |
| Gaussian Processes for Machine Learning (Rasmussen & Williams) | https://gaussianprocess.org/gpml/ |
| Stan User's Guide & Reference Manual | https://mc-stan.org/docs/ |
| NumPyro documentation | https://num.pyro.ai/ |

## Cross-references in AIForge

- [Uncertainty Quantification](../Uncertainty_Quantification/) — calibration, conformal prediction, predictive intervals
- [Active Learning](../Active_Learning/) — acquisition functions driven by posterior uncertainty
- [Causal Inference](../Causal_Inference/) — Bayesian structural and hierarchical causal models
- [Information Theory](../Information_Theory/) — KL divergence, ELBO, and the variational objective

## Sources

- https://github.com/pyro-ppl/numpyro
- https://github.com/pymc-devs/pymc
- https://github.com/stan-dev/stan
- https://github.com/cornellius-gp/gpytorch
- https://github.com/aleximmer/Laplace
- https://github.com/IntelLabs/bayesian-torch
- https://arxiv.org/abs/1111.4246
- https://arxiv.org/abs/1601.00670
- https://arxiv.org/abs/1603.00788
- https://arxiv.org/abs/1506.02142
- https://arxiv.org/abs/1505.05424
- https://arxiv.org/abs/1612.01474
- https://arxiv.org/abs/2106.14806
- https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/

_Seed section — curated from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
