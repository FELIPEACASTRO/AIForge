# Generalized Linear Models (GLMs)

> A unifying regression framework that extends ordinary least squares to response variables whose conditional distribution belongs to the exponential family, connecting the linear predictor to the mean through a link function.

## Why it matters

GLMs are the workhorse for modeling non-Gaussian targets: counts (Poisson), binary/proportion outcomes (logistic/binomial), positive skewed quantities (Gamma), and mixed zero-and-positive data (Tweedie). They retain the interpretability and statistical inference machinery of linear regression (coefficients, standard errors, deviance) while correctly handling the mean-variance relationship of real data, which is why they remain central to actuarial pricing, epidemiology, econometrics, and any setting where calibrated, explainable predictions matter.

## Core concepts

A GLM has three components:

1. **Random component** — the response `y` follows an exponential-family distribution with density
   `f(y; θ, φ) = exp[(yθ − b(θ))/a(φ) + c(y, φ)]`,
   where `θ` is the natural (canonical) parameter, `φ` the dispersion parameter, and `b(θ)` the cumulant function. From this, `E[y] = μ = b'(θ)` and `Var(y) = a(φ)·b''(θ) = a(φ)·V(μ)`, where `V(μ)` is the **variance function**.

2. **Systematic component** — a linear predictor `η = Xβ`.

3. **Link function** — a monotone, differentiable `g` connecting the two: `g(μ) = η = Xβ`, so `μ = g⁻¹(Xβ)`. The **canonical link** sets `θ = η` (e.g., `log` for Poisson, `logit` for Bernoulli, identity for Gaussian).

Key ideas:

- **Mean-variance link**: each family fixes `V(μ)` — Gaussian `V=1`, Poisson `V=μ`, Gamma `V=μ²`, Tweedie `V=μ^p`. Choosing the right variance function (not just the link) is what makes a GLM well-specified.
- **Estimation by IRLS**: maximum-likelihood estimates are obtained via **Iteratively Reweighted Least Squares** (Fisher scoring), where weights and a working response are updated each iteration; this is equivalent to Newton-Raphson with the canonical link.
- **Deviance** `D = 2[ℓ(saturated) − ℓ(model)]` generalizes the residual sum of squares and drives goodness-of-fit, residual analysis, and likelihood-ratio tests.
- **Quasi-likelihood** (Wedderburn) only requires specifying `V(μ)` and a dispersion `φ`, allowing estimation without a fully specified distribution and handling **overdispersion** (`Var > V(μ)`), common in count data.
- **Regularization**: L1/L2 penalties extend GLMs to high dimensions (penalized IRLS / coordinate descent), as in `glmnet` and scikit-learn's penalized GLM regressors.

## Variants

| Family / model | Typical response | Canonical link | Variance `V(μ)` | Common use |
|---|---|---|---|---|
| Gaussian (OLS) | real-valued | identity | `1` | continuous, symmetric errors |
| Binomial / Logistic | binary, proportion | logit | `μ(1−μ)` | classification, risk |
| Poisson | counts ≥ 0 | log | `μ` | event counts, rates |
| Negative Binomial | overdispersed counts | log | `μ + μ²/k` | counts with extra variance |
| Gamma | positive, skewed | log / inverse | `μ²` | costs, durations, severities |
| Inverse Gaussian | positive, heavy-tailed | inverse-squared | `μ³` | reaction times, finance |
| Tweedie (1 < p < 2) | zero-inflated positive | log | `μ^p` (power) | insurance pure premium, rainfall |
| Quasi-Poisson / Quasi-likelihood | counts / proportions | log / logit | `φ·V(μ)` | overdispersion without full likelihood |

## Tools & libraries

| Tool | What it offers | Link |
|---|---|---|
| statsmodels (`GLM`) | Full GLM with all exponential families, links, inference, deviance | https://www.statsmodels.org/stable/glm.html |
| scikit-learn `PoissonRegressor` | Poisson GLM with L2 penalty | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html |
| scikit-learn `GammaRegressor` | Gamma GLM with L2 penalty | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html |
| scikit-learn `TweedieRegressor` | Tweedie/compound-Poisson-Gamma GLM (power link) | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html |
| R `glm()` (stats) | Reference GLM implementation (IRLS) | https://stat.ethz.ch/R-manual/R-devel/library/stats/html/glm.html |
| glmnet | L1/L2-penalized GLMs at scale (R/Python) | https://glmnet.stanford.edu/ |
| XGBoost | Gradient boosting with Poisson, Gamma, Tweedie objectives | https://xgboost.readthedocs.io/en/stable/parameter.html |
| PyMC | Bayesian GLMs via probabilistic programming | https://www.pymc.io/ |
| Optuna | Hyperparameter search (regularization, Tweedie power `p`) | https://optuna.org/ |

## Learning resources

- **An Introduction to Statistical Learning (ISLR)** — James, Witten, Hastie, Tibshirani; Ch. 4 on classification/logistic regression, free PDF: https://www.statlearning.com/
- **The Elements of Statistical Learning (ESL)** — Hastie, Tibshirani, Friedman; free PDF: https://hastie.su.domains/ElemStatLearn/
- **Generalized Linear Models** (2nd ed., 1989) — McCullagh & Nelder, the canonical text: https://www.routledge.com/Generalized-Linear-Models/McCullagh-Nelder/p/book/9780412317606
- **Probabilistic Machine Learning: An Introduction** — Kevin Murphy; exponential family & GLM chapters, free PDF: https://probml.github.io/pml-book/book1.html
- **statsmodels GLM user guide** — practical, worked examples: https://www.statsmodels.org/stable/glm.html
- **scikit-learn GLM user guide** — Poisson/Gamma/Tweedie regression: https://scikit-learn.org/stable/modules/linear_model.html#generalized-linear-models
- **Tweedie regression on insurance claims** (scikit-learn tutorial): https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
- **StatQuest — Logistic / GLM playlists** (Josh Starmer): https://www.youtube.com/c/joshstarmer

## Key papers

- Nelder, J.A. & Wedderburn, R.W.M. (1972). *Generalized Linear Models.* JRSS Series A, 135(3), 370–384. DOI: https://doi.org/10.2307/2344614
- Wedderburn, R.W.M. (1974). *Quasi-Likelihood Functions, Generalized Linear Models, and the Gauss–Newton Method.* Biometrika, 61(3), 439–447. DOI: https://doi.org/10.1093/biomet/61.3.439
- Tweedie, M.C.K. (1984). *An index which distinguishes between some important exponential families.* Statistics: Applications and New Directions (Indian Statistical Institute). Overview: https://en.wikipedia.org/wiki/Tweedie_distribution
- Friedman, J., Hastie, T. & Tibshirani, R. (2010). *Regularization Paths for Generalized Linear Models via Coordinate Descent.* Journal of Statistical Software, 33(1). DOI: https://doi.org/10.18637/jss.v033.i01

## Cross-references in AIForge

- [Machine_Learning](../../Machine_Learning/) — broader supervised-learning context and model families
- [Optimization_Algorithms](../../Optimization_Algorithms/) — IRLS, Newton/Fisher scoring, coordinate descent
- [Bayesian_and_Probabilistic_ML](../../Bayesian_and_Probabilistic_ML/) — Bayesian GLMs and the exponential family
- [Model_Evaluation](../../Model_Evaluation/) — deviance, calibration, and goodness-of-fit metrics

## Sources

- Nelder & Wedderburn (1972), JRSS A — https://doi.org/10.2307/2344614 (Oxford Academic: https://academic.oup.com/jrsssa/article/135/3/370/7110572)
- Wedderburn (1974), Biometrika — https://doi.org/10.1093/biomet/61.3.439
- McCullagh & Nelder (1989), Routledge — https://www.routledge.com/Generalized-Linear-Models/McCullagh-Nelder/p/book/9780412317606
- statsmodels GLM docs — https://www.statsmodels.org/stable/glm.html
- scikit-learn GLM docs & regressors — https://scikit-learn.org/stable/modules/linear_model.html#generalized-linear-models
- Friedman, Hastie & Tibshirani (2010), JSS — https://doi.org/10.18637/jss.v033.i01
- glmnet — https://glmnet.stanford.edu/
