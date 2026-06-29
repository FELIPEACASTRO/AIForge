# Statistics Fundamentals

> The theory and practice of drawing reliable conclusions about populations and models from finite, noisy data — estimating parameters, quantifying uncertainty, and managing the bias–variance trade-off.

## Why it matters

Every supervised learning algorithm is, under the hood, a statistical estimator: it infers parameters from samples and generalizes to unseen data. Understanding estimation, likelihood, confidence intervals, and the bias–variance decomposition is what separates fitting a model from knowing when to trust it. These ideas underpin loss-function design, regularization, model selection, and the uncertainty estimates that production ML systems increasingly require.

## Core concepts

**Estimators and their properties.** An estimator `θ̂` is a function of the data that targets an unknown parameter `θ`. Key properties:
- **Bias**: `Bias(θ̂) = E[θ̂] − θ`. Unbiased means `E[θ̂] = θ`.
- **Variance**: `Var(θ̂) = E[(θ̂ − E[θ̂])²]`.
- **Mean squared error**: `MSE(θ̂) = Bias(θ̂)² + Var(θ̂)` — the central identity behind the bias–variance trade-off.
- **Consistency**: `θ̂ → θ` in probability as `n → ∞`.
- **Efficiency**: an unbiased estimator is efficient if its variance attains the **Cramér–Rao lower bound** `Var(θ̂) ≥ 1 / I(θ)`, where `I(θ)` is the Fisher information.

**Maximum Likelihood Estimation (MLE).** Choose parameters that maximize the likelihood of the observed data: `θ̂_MLE = argmax_θ L(θ; x) = argmax_θ Σ log p(xᵢ | θ)`. MLE is consistent and asymptotically efficient and normal: `θ̂ ≈ N(θ, I(θ)⁻¹)`. Many ML losses are MLE in disguise — squared error ⇔ Gaussian noise; cross-entropy ⇔ Bernoulli/categorical likelihood.

**Maximum A Posteriori (MAP).** Incorporate a prior `p(θ)` via Bayes' rule: `θ̂_MAP = argmax_θ [log p(x | θ) + log p(θ)]`. MAP equals MLE plus a regularizer: a Gaussian prior gives L2 (ridge), a Laplace prior gives L1 (lasso). As data grows, MAP → MLE (the likelihood dominates the prior).

**Confidence intervals (CIs).** A 95% CI is a procedure that, over repeated sampling, contains the true parameter 95% of the time. For a normal mean with known σ: `x̄ ± z_{1−α/2} · σ/√n`; with unknown σ use the **t-distribution**: `x̄ ± t_{n−1, 1−α/2} · s/√n`. Note the frequentist interpretation: the *interval* is random, not `θ`. Contrast with a Bayesian **credible interval**, a region with posterior probability `1−α`.

**Bias–variance trade-off.** Expected test error decomposes as `E[(y − f̂(x))²] = Bias[f̂(x)]² + Var[f̂(x)] + σ²` (irreducible noise). Simple models underfit (high bias); flexible models overfit (high variance). Regularization, more data, and ensembling trade these off. The classical U-shaped curve is complicated by **double descent** in over-parameterized models.

**Resampling.** The **bootstrap** estimates the sampling distribution of any statistic by repeatedly resampling the data with replacement — enabling standard errors and CIs with no closed form. The **jackknife** is a leave-one-out linear approximation.

**Hypothesis testing.** Quantifies evidence against a null hypothesis via p-values and test statistics (z, t, χ², F); errors are Type I (false positive, rate α) and Type II (false negative, rate β; power = 1−β).

## Algorithms / Methods

| Method | Idea | Use case / notes |
|---|---|---|
| Method of moments | Match sample moments to theoretical moments | Simple, consistent; often less efficient than MLE |
| Maximum likelihood (MLE) | Maximize `Σ log p(xᵢ\|θ)` | Default point estimator; asymptotically efficient |
| MAP estimation | Maximize posterior `log p(x\|θ) + log p(θ)` | MLE + regularization; bridges to Bayesian inference |
| Ridge / James–Stein shrinkage | Bias estimates toward a target to cut variance | Dominates MLE for ≥3 dimensions in MSE |
| Bootstrap | Resample data with replacement | Nonparametric SEs and CIs for arbitrary statistics |
| Jackknife | Leave-one-out resampling | Bias/variance estimation; linear bootstrap approximation |
| Wald / t / z confidence intervals | `θ̂ ± critical · SE` | Parametric CIs from asymptotic normality |
| Likelihood-ratio / Wald / score tests | Compare nested models via likelihood | Hypothesis testing and CI construction |
| Cross-validation | Hold-out resampling for risk estimation | Model selection; estimates generalization error |
| AIC / BIC | Penalized likelihood `−2 log L + k·penalty` | Information-theoretic model selection |

## Tools & libraries

| Tool | What it offers | Link |
|---|---|---|
| SciPy (`scipy.stats`) | Distributions, MLE fitting, hypothesis tests, CIs | https://docs.scipy.org/doc/scipy/reference/stats.html |
| statsmodels | Regression, GLMs, tests, time series, robust SEs | https://www.statsmodels.org/stable/ |
| NumPy | Arrays, RNG, moments, bootstrap building blocks | https://numpy.org/ |
| scikit-learn | Estimators, cross-validation, calibration | https://scikit-learn.org/stable/ |
| PyMC | Bayesian modeling, MAP, MCMC posterior inference | https://www.pymc.io/ |
| Pingouin | Friendly statistical tests, effect sizes, CIs | https://pingouin-stats.org/ |
| R + base/`stats` | Reference statistical computing environment | https://www.r-project.org/ |
| arviz | Diagnostics & visualization for Bayesian inference | https://www.arviz.org/ |

## Learning resources

- **Mathematics for Machine Learning** — Deisenroth, Faisal, Ong (free PDF). Probability & statistics chapters tie directly to ML. https://mml-book.com/
- **An Introduction to Statistical Learning (ISLR/ISLP)** — James, Witten, Hastie, Tibshirani (free PDF, Python & R editions). Bias–variance, resampling, model selection. https://www.statlearning.com/
- **The Elements of Statistical Learning (ESL)** — Hastie, Tibshirani, Friedman (free PDF). The deeper companion to ISLR. https://hastie.su.domains/ElemStatLearn/
- **All of Statistics** — Larry Wasserman. Fast, rigorous coverage of inference for CS/ML readers. https://www.stat.cmu.edu/~larry/all-of-statistics/
- **Computer Age Statistical Inference** — Efron & Hastie (free PDF). Bootstrap, shrinkage, empirical Bayes. https://hastie.su.domains/CASI/
- **Probabilistic Machine Learning** — Kevin Murphy (free PDFs). Estimation, MLE/MAP, Bayesian inference. https://probml.github.io/pml-book/
- **StatQuest with Josh Starmer** — intuitive video explanations of MLE, CIs, bias–variance. https://www.youtube.com/@statquest
- **Seeing Theory** — interactive visual intro to probability & statistics. https://seeing-theory.brown.edu/

## Key papers

- Stein, C. (1956). *Inadmissibility of the usual estimator for the mean of a multivariate normal distribution.* Proc. 3rd Berkeley Symposium. https://projecteuclid.org/ebooks/berkeley-symposium-on-mathematical-statistics-and-probability/Proceedings-of-the-Third-Berkeley-Symposium-on-Mathematical-Statistics-and/chapter/Inadmissibility-of-the-Usual-Estimator-for-the-Mean-of-a/bsmsp/1200501656
- James, W. & Stein, C. (1961). *Estimation with quadratic loss.* Proc. 4th Berkeley Symposium. https://projecteuclid.org/ebooks/berkeley-symposium-on-mathematical-statistics-and-probability/Proceedings-of-the-Fourth-Berkeley-Symposium-on-Mathematical-Statistics-and/chapter/Estimation-with-Quadratic-Loss/bsmsp/1200512173
- Akaike, H. (1974). *A new look at the statistical model identification.* IEEE TAC 19(6):716–723. https://doi.org/10.1109/TAC.1974.1100705
- Efron, B. (1979). *Bootstrap methods: another look at the jackknife.* Annals of Statistics 7(1):1–26. https://doi.org/10.1214/aos/1176344552
- Schwarz, G. (1978). *Estimating the dimension of a model.* Annals of Statistics 6(2):461–464 (BIC). https://doi.org/10.1214/aos/1176344136
- Belkin, M. et al. (2019). *Reconciling modern machine-learning practice and the bias–variance trade-off.* PNAS 116(32):15849–15854. https://doi.org/10.1073/pnas.1903070116

## Cross-references in AIForge

- [Probability Theory](../Probability_Theory/) — the probabilistic foundation for likelihoods and estimators.
- [Bayesian and Probabilistic ML](../../Bayesian_and_Probabilistic_ML/) — priors, posteriors, and credible intervals beyond MAP.
- [Model Evaluation](../../Model_Evaluation/) — cross-validation, generalization error, and metric uncertainty.
- [Optimization Algorithms](../../Optimization_Algorithms/) — how MLE/MAP objectives are actually solved.

## Sources

- statsmodels documentation — https://www.statsmodels.org/stable/
- Mathematics for Machine Learning — https://mml-book.com/
- Efron (1979), Annals of Statistics — https://doi.org/10.1214/aos/1176344552
- James & Stein (1961), Berkeley Symposium — https://projecteuclid.org/ebooks/berkeley-symposium-on-mathematical-statistics-and-probability/Proceedings-of-the-Fourth-Berkeley-Symposium-on-Mathematical-Statistics-and/chapter/Estimation-with-Quadratic-Loss/bsmsp/1200512173
- Akaike (1974), IEEE TAC — https://doi.org/10.1109/TAC.1974.1100705
- ISLR / statlearning.com — https://www.statlearning.com/
- Computer Age Statistical Inference (Efron & Hastie) — https://hastie.su.domains/CASI/
