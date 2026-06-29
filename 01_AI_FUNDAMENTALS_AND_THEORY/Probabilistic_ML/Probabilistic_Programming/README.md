# Probabilistic Programming

> Probabilistic programming languages (PPLs) let you express a generative statistical model as ordinary code and then automatically perform Bayesian inference (MCMC, variational, or otherwise) over it.

## Why it matters

Probabilistic programming decouples *model specification* from *inference*: you declare priors, likelihoods, and latent structure as a program, and a general-purpose inference engine computes the posterior — no hand-derived sampler required. This collapses the iteration loop for Bayesian modeling, makes uncertainty a first-class output, and scales the same workflow from a two-parameter regression to deep generative models with neural-network components.

## Core concepts

- **Generative model.** A PPL program defines a joint distribution `p(θ, y) = p(θ) · p(y | θ)` over latent parameters `θ` and observed data `y`. Sampling statements (`x ~ Normal(0, 1)`) declare random variables; observed nodes are conditioned on data.
- **The inference target.** The goal is the posterior `p(θ | y) = p(θ) p(y | θ) / p(y)`, where the evidence `p(y) = ∫ p(θ) p(y | θ) dθ` is usually intractable. Inference engines approximate `p(θ | y)` without computing `p(y)` directly.
- **MCMC.** Markov Chain Monte Carlo draws correlated samples whose stationary distribution is the posterior. **Hamiltonian Monte Carlo (HMC)** augments `θ` with a momentum variable and simulates Hamiltonian dynamics using `∇ log p(θ, y)` to make long, low-rejection proposals; the **No-U-Turn Sampler (NUTS)** adapts the trajectory length automatically.
- **Variational inference (VI).** Casts inference as optimization: pick a tractable family `q_φ(θ)` and maximize the **ELBO**, `L(φ) = E_q[log p(θ, y) − log q_φ(θ)]`, which is equivalent to minimizing `KL(q_φ ‖ p(θ|y))`. Faster than MCMC but yields an approximation.
- **Automatic differentiation.** Modern PPLs compile the model to a differentiable graph (PyTensor, JAX, PyTorch, TensorFlow) so gradients `∇_θ log p(θ, y)` are obtained for free — the enabling ingredient for HMC/NUTS and gradient-based VI.
- **Constraining transforms & reparameterization.** Constrained parameters (e.g. variances `> 0`, simplexes) are mapped to unconstrained space so gradient samplers can operate; the reparameterization trick `θ = g_φ(ε)` gives low-variance ELBO gradients.
- **Diagnostics.** Convergence and validity are checked with R-hat (`R̂`), effective sample size (ESS), divergent transitions, and posterior predictive checks.

## Algorithms / Methods

| Method | Type | Idea | Notes |
|---|---|---|---|
| Metropolis–Hastings | MCMC | Accept/reject random-walk proposals | Simple, mixes poorly in high dimensions |
| Gibbs sampling | MCMC | Sample each variable from its full conditional | Needs conjugacy; used by BUGS/JAGS |
| Hamiltonian Monte Carlo (HMC) | Gradient MCMC | Momentum + Hamiltonian dynamics via gradients | Efficient in high dimensions; needs step-size/length tuning |
| NUTS | Gradient MCMC | HMC that auto-tunes trajectory length | Default in Stan, PyMC, NumPyro, Turing |
| SMC / particle methods | Sequential MC | Population of weighted particles | Good for multimodal/tempered posteriors, state-space models |
| Mean-field VI / ADVI | Variational | Factorized `q`, maximize ELBO automatically | Fast, scalable; underestimates correlations/variance |
| Stochastic VI (SVI) | Variational | Minibatch ELBO gradients | Scales to large data; core of Pyro |
| Normalizing-flow VI | Variational | Flexible `q` via invertible transforms | Richer approximate posteriors |
| Laplace approximation / INLA | Deterministic | Gaussian around the mode | Fast for latent Gaussian models |

## Tools & libraries

| Tool | Backend / Language | Strengths | URL |
|---|---|---|---|
| PyMC | Python (PyTensor) | Pythonic API, NUTS, GLMs, GPs, large ecosystem | https://www.pymc.io/ |
| Stan | C++ (R/Python/Julia interfaces) | Reference NUTS, robust diagnostics, mature | https://mc-stan.org/ |
| Pyro | Python (PyTorch) | Deep probabilistic models, SVI at scale | https://pyro.ai/ |
| NumPyro | Python (JAX) | Pyro API with JAX-accelerated NUTS/HMC | https://num.pyro.ai/ |
| Turing.jl | Julia | Composable inference, native Julia speed | https://turinglang.org/ |
| TensorFlow Probability | Python (TensorFlow) | Distributions, VI, Bayesian deep learning, TPU/GPU | https://www.tensorflow.org/probability |
| ArviZ | Python (backend-agnostic) | Posterior diagnostics & visualization | https://python.arviz.org/ |
| Bambi | Python (on PyMC) | Formula-based Bayesian GLMMs | https://bambinos.github.io/bambi/ |
| brms | R (on Stan) | Formula-based Bayesian regression in R | https://paulbuerkner.com/brms/ |
| JAGS | C++ (BUGS dialect) | Gibbs sampling, teaching | https://mcmc-jags.sourceforge.io/ |

## Learning resources

- **Statistical Rethinking** — Richard McElreath. The standard hands-on Bayesian course (book + lectures), with PyMC/Stan/NumPyro/Turing ports. https://xcelab.net/rm/
- **Bayesian Data Analysis (BDA3)** — Gelman et al. Free PDF; the canonical reference. https://www.stat.columbia.edu/~gelman/book/
- **Probabilistic Machine Learning** — Kevin Murphy. Free books covering inference and PPLs. https://probml.github.io/pml-book/
- **PyMC learning hub & example gallery.** https://www.pymc.io/projects/examples/ — and **Pyro tutorials** https://pyro.ai/examples/
- **Stan User's Guide & documentation.** https://mc-stan.org/users/documentation/
- **Bayesian Methods for Hackers** — Davidson-Pilon. Free, code-first intro (PyMC/TFP). https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers
- **A Conceptual Introduction to Hamiltonian Monte Carlo** — Betancourt (geometric intuition for HMC). https://arxiv.org/abs/1701.02434

## Key papers

- Carpenter et al. (2017), *Stan: A Probabilistic Programming Language*, J. Stat. Softw. 76(1). https://www.jstatsoft.org/article/view/v076i01
- Hoffman & Gelman (2014), *The No-U-Turn Sampler*, JMLR 15. https://jmlr.org/papers/v15/hoffman14a.html (arXiv: https://arxiv.org/abs/1111.4246)
- Bingham et al. (2019), *Pyro: Deep Universal Probabilistic Programming*, JMLR 20. https://www.jmlr.org/papers/volume20/18-403/18-403.pdf (arXiv: https://arxiv.org/abs/1810.09538)
- Kucukelbir et al. (2017), *Automatic Differentiation Variational Inference*, JMLR 18. https://jmlr.org/papers/v18/16-107.html (arXiv: https://arxiv.org/abs/1603.00788)
- Blei, Kucukelbir & McAuliffe (2017), *Variational Inference: A Review for Statisticians*, JASA 112. https://arxiv.org/abs/1601.00670
- Abril-Pla et al. (2023), *PyMC: a modern and comprehensive probabilistic programming framework in Python*, PeerJ CS. https://peerj.com/articles/cs-1516/

## Cross-references in AIForge

- [Bayesian and Probabilistic ML](../../Bayesian_and_Probabilistic_ML/) — foundational Bayesian theory and priors.
- [MCMC and Sampling](../MCMC_and_Sampling/) — the sampling algorithms PPLs build on.
- [Variational Inference](../Variational_Inference/) — the optimization-based inference engines.
- [Optimization Algorithms](../../Optimization_Algorithms/) — stochastic gradient methods behind VI.

## Sources

- PyMC — https://www.pymc.io/ ; PyMC paper (PeerJ) — https://peerj.com/articles/cs-1516/
- Stan — https://mc-stan.org/ ; Stan paper (JSS) — https://www.jstatsoft.org/article/view/v076i01
- Pyro — https://pyro.ai/ ; Pyro paper (JMLR) — https://www.jmlr.org/papers/volume20/18-403/18-403.pdf
- NumPyro — https://num.pyro.ai/
- Turing.jl — https://turinglang.org/
- TensorFlow Probability — https://www.tensorflow.org/probability
- NUTS paper — https://jmlr.org/papers/v15/hoffman14a.html (arXiv https://arxiv.org/abs/1111.4246)
- ADVI — https://jmlr.org/papers/v18/16-107.html (arXiv https://arxiv.org/abs/1603.00788)
- Variational Inference review — https://arxiv.org/abs/1601.00670
- HMC conceptual intro (Betancourt) — https://arxiv.org/abs/1701.02434
