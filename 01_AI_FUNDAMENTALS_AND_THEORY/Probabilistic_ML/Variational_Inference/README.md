# Variational Inference

> Variational Inference (VI) turns Bayesian posterior inference into an optimization problem: pick a tractable family of distributions and find the member closest (in KL divergence) to the true posterior.

## Why it matters

Exact Bayesian inference is intractable for almost all interesting models because the marginal likelihood (evidence) is an unsolvable integral. VI replaces costly sampling (MCMC) with optimization, making approximate posterior inference fast, deterministic, and scalable to massive datasets and high-dimensional models. It is the engine behind modern probabilistic programming, variational autoencoders, and Bayesian deep learning.

## Core concepts

Given observed data `x` and latent variables `z`, we want the posterior `p(z|x) = p(x,z)/p(x)`. The evidence `p(x) = ∫ p(x,z) dz` is intractable. VI introduces a **variational family** `q(z; λ)` parameterized by `λ` and minimizes the KL divergence `KL(q(z;λ) ‖ p(z|x))`.

- **ELBO (Evidence Lower BOund).** Since KL is intractable (it contains `p(x)`), we instead maximize the ELBO:
  `L(λ) = E_q[log p(x,z)] − E_q[log q(z;λ)] = E_q[log p(x|z)] − KL(q(z;λ) ‖ p(z))`.
  Because `log p(x) = L(λ) + KL(q ‖ p(z|x))` and KL ≥ 0, maximizing the ELBO is equivalent to minimizing the posterior KL. The ELBO is a *lower bound* on the log-evidence.
- **Reverse vs. forward KL.** VI minimizes the *reverse* KL `KL(q‖p)`, which is mode-seeking / zero-forcing and tends to **underestimate posterior variance** — an important known limitation. (Forward KL `KL(p‖q)` underlies Expectation Propagation.)
- **Mean-field approximation.** Assume the posterior factorizes: `q(z) = ∏ᵢ qᵢ(zᵢ)`. The optimal coordinate update is `log qⱼ*(zⱼ) = E_{q_{-j}}[log p(x,z)] + const`, giving the **Coordinate Ascent VI (CAVI)** algorithm. Closed-form in conditionally-conjugate exponential-family models.
- **Stochastic optimization.** Replace full-data gradients with noisy minibatch estimates (SVI) for scalability; natural gradients exploit the information geometry of the variational family.
- **Black-box / reparameterized gradients.** For non-conjugate models, estimate `∇_λ L` via Monte Carlo. The **score-function (REINFORCE)** estimator works generally but is high-variance; the **reparameterization trick** (`z = g(ε, λ)`, `ε ∼ p(ε)`) gives low-variance pathwise gradients for continuous latents.
- **Amortized inference.** Learn a shared inference network `q(z|x; φ)` mapping data to variational parameters instead of optimizing per-datapoint — the basis of the VAE encoder.
- **Expressive posteriors.** Mean-field is often too restrictive. **Normalizing flows** apply a sequence of invertible transforms to a simple base density, using the change-of-variables formula to build flexible, tractable-density posteriors.

## Variants / Methods

| Method | Key idea | Best for | Limitation |
|---|---|---|---|
| **CAVI (mean-field)** | Closed-form coordinate-ascent updates | Conjugate exponential-family models | Factorized q; underestimates variance |
| **SVI** | Stochastic natural-gradient ELBO ascent on minibatches | Massive datasets (topic models, etc.) | Needs conjugacy / structure |
| **BBVI** | Monte-Carlo score-function gradients of ELBO | Non-conjugate, generic models | High-variance gradients; needs control variates |
| **ADVI** | Auto-diff + transform to real coords + reparam. gradients | "Just write the model" (Stan/PyMC) | Mostly Gaussian/mean-field in latent space |
| **Reparameterization (VAE)** | Pathwise gradients via inference network | Deep generative / amortized models | Continuous latents; posterior collapse risk |
| **Normalizing-flow VI** | Invertible transforms for flexible q | Complex, multimodal-ish posteriors | Architectural / compute cost |
| **Structured / full-rank VI** | Capture latent correlations (non mean-field) | When dependencies matter | More parameters, costlier |
| **Stein VI (SVGD)** | Particle-based, kernelized gradient flow | Nonparametric, flexible q | Kernel/particle tuning |

## Tools & libraries

| Tool | What it offers | URL |
|---|---|---|
| **PyMC** | Probabilistic programming; ADVI, full-rank & minibatch VI | https://www.pymc.io/ |
| **Stan** | ADVI / Pathfinder variational algorithms | https://mc-stan.org/ |
| **Pyro** | Deep universal PPL on PyTorch; SVI, flows, amortized VI | https://pyro.ai/ |
| **NumPyro** | JAX-based PPL; fast SVI and autoguides | https://num.pyro.ai/ |
| **TensorFlow Probability** | Distributions, bijectors (flows), VI utilities | https://www.tensorflow.org/probability |
| **Edward2 / TFP** | Variational layers and inference primitives | https://github.com/google/edward2 |
| **GPyTorch** | Scalable variational Gaussian processes | https://gpytorch.ai/ |
| **scikit-learn** | `BayesianGaussianMixture` (variational Bayes for GMMs) | https://scikit-learn.org/ |
| **normflows** | Normalizing-flow library for PyTorch | https://github.com/VincentStimper/normalizing-flows |

## Learning resources

- **Murphy — *Probabilistic Machine Learning: Advanced Topics* (2023)** — definitive modern treatment of VI; free PDF. https://probml.github.io/pml-book/book2.html
- **Bishop — *Pattern Recognition and Machine Learning*, Ch. 10** — classic mean-field / CAVI derivations. https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/
- **Deisenroth, Faisal, Ong — *Mathematics for Machine Learning*** — prerequisite probability/optimization. https://mml-book.github.io/
- **Blei — Columbia "Foundations of Graphical Models" lecture notes** — VI from the author of the review. https://www.cs.columbia.edu/~blei/fogm/
- **Stanford CS228 (PGM) notes on Variational Inference** — clear, concise. https://ermongroup.github.io/cs228-notes/inference/variational/
- **Pyro SVI tutorials** — hands-on ELBO, guides, amortization. https://pyro.ai/examples/svi_part_i.html
- **Lilian Weng — "Flow-based Deep Generative Models"** — intuitive normalizing-flow walkthrough. https://lilianweng.github.io/posts/2018-10-13-flow-models/

## Key papers

- Blei, Kucukelbir & McAuliffe (2017), **Variational Inference: A Review for Statisticians**, *JASA*. https://arxiv.org/abs/1601.00670
- Jordan, Ghahramani, Jaakkola & Saul (1999), **An Introduction to Variational Methods for Graphical Models**, *Machine Learning*. https://doi.org/10.1023/A:1007665907178
- Hoffman, Blei, Wang & Paisley (2013), **Stochastic Variational Inference**, *JMLR*. https://jmlr.org/papers/v14/hoffman13a.html
- Ranganath, Gerrish & Blei (2014), **Black Box Variational Inference**, *AISTATS*. https://arxiv.org/abs/1401.0118
- Kingma & Welling (2014), **Auto-Encoding Variational Bayes**, *ICLR* (reparameterization / VAE). https://arxiv.org/abs/1312.6114
- Rezende & Mohamed (2015), **Variational Inference with Normalizing Flows**, *ICML*. https://arxiv.org/abs/1505.05770
- Kucukelbir, Tran, Ranganath, Gelman & Blei (2017), **Automatic Differentiation Variational Inference**, *JMLR*. https://arxiv.org/abs/1603.00788

## Cross-references in AIForge

- [Bayesian and Probabilistic ML](../../Bayesian_and_Probabilistic_ML/README.md) — priors, posteriors, MCMC alternatives
- [Optimization Algorithms — stochastic gradient & natural-gradient methods
- [Deep Learning — VAEs, amortized inference, Bayesian neural nets
- [Machine Learning — graphical models and mixture models

## Sources

- arXiv: Variational Inference: A Review for Statisticians — https://arxiv.org/abs/1601.00670
- JMLR: Stochastic Variational Inference — https://jmlr.org/papers/v14/hoffman13a.html
- arXiv: Black Box Variational Inference — https://arxiv.org/abs/1401.0118
- arXiv: Auto-Encoding Variational Bayes — https://arxiv.org/abs/1312.6114
- arXiv: Variational Inference with Normalizing Flows — https://arxiv.org/abs/1505.05770
- arXiv / JMLR: Automatic Differentiation Variational Inference — https://arxiv.org/abs/1603.00788 , https://jmlr.org/papers/v18/16-107.html
- Springer: Introduction to Variational Methods for Graphical Models — https://doi.org/10.1023/A:1007665907178
- PyMC — https://www.pymc.io/ ; Pyro — https://pyro.ai/ ; Stan — https://mc-stan.org/ ; TFP — https://www.tensorflow.org/probability
- Murphy, *Probabilistic ML: Advanced Topics* — https://probml.github.io/pml-book/book2.html
- Stanford CS228 VI notes — https://ermongroup.github.io/cs228-notes/inference/variational/
