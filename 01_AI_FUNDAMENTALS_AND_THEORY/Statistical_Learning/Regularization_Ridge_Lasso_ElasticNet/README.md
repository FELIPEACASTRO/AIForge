# Regularization: Ridge, Lasso, Elastic Net

> Penalized regression methods that shrink coefficient estimates toward zero to trade a little bias for a large reduction in variance, improving generalization and (for L1) producing sparse, interpretable models.

## Why it matters

Ordinary least squares is unbiased but can have enormous variance when predictors are many, correlated (multicollinear), or outnumber observations (`p > n`), leading to overfitting and unstable coefficients. Regularization adds a penalty on coefficient magnitude that stabilizes estimation, performs automatic feature selection (Lasso/Elastic Net), and almost always improves out-of-sample prediction. These are among the most widely used tools in applied statistics and ML, underpinning high-dimensional genomics, finance, and as the L2 weight-decay term in nearly every deep network.

## Core concepts

Given a linear model `y = Xβ + ε`, penalized regression minimizes the loss plus a penalty:

- **Ridge (L2)**: minimize `RSS + λ Σ βⱼ²`. Has a closed form `β̂ = (XᵀX + λI)⁻¹ Xᵀy`. Shrinks coefficients smoothly toward zero but never exactly to zero; handles multicollinearity well and keeps all features. Equivalent to a Gaussian (Normal) prior on `β` (MAP estimate).
- **Lasso (L1)**: minimize `RSS + λ Σ |βⱼ|`. The non-differentiable L1 ball has corners on the axes, so the solution sets many coefficients exactly to zero → **sparsity** and built-in variable selection. Equivalent to a Laplace (double-exponential) prior. No closed form; solved by coordinate descent or LARS.
- **Elastic Net**: minimize `RSS + λ[α Σ|βⱼ| + (1−α) Σ βⱼ²]`, mixing L1 and L2. Inherits Lasso's sparsity while Ridge's L2 term induces a **grouping effect** — strongly correlated predictors are selected (or dropped) together. Preferred when `p ≫ n` or features are highly correlated, where Lasso alone is erratic (it picks at most `n` variables and arbitrarily one from each correlated group).
- **Bias–variance trade-off**: increasing `λ` increases bias but decreases variance; the optimal `λ` is chosen by cross-validation (e.g., minimum CV error, or the "1-SE rule" for a sparser model).
- **Regularization path**: the full trajectory of `β̂(λ)` as `λ` varies from large (all zero) to small (≈ OLS). Lasso paths are piecewise linear in `λ`; coordinate descent computes the whole path cheaply via warm starts.
- **Standardization**: penalties are scale-dependent, so predictors are typically centered and standardized to unit variance before fitting; the intercept is left unpenalized.
- **Beyond linear**: the same penalties attach to logistic regression and other GLMs (penalized log-likelihood), and the L2 penalty is exactly **weight decay** in neural networks.

## Variants

| Method | Penalty | Sparsity | Key property | Typical use |
|---|---|---|---|---|
| Ridge | `λ‖β‖₂²` (L2) | No | Smooth shrinkage, closed form, handles collinearity | Many correlated predictors, want all kept |
| Lasso | `λ‖β‖₁` (L1) | Yes | Variable selection; ≤ `n` nonzeros | Sparse signal, interpretable models |
| Elastic Net | `λ(α‖β‖₁ + (1−α)‖β‖₂²)` | Yes | Grouping effect; stable under correlation | `p ≫ n`, correlated groups |
| Adaptive Lasso | weighted `λ Σ wⱼ|βⱼ|` | Yes | Oracle property (consistent selection) | When unbiased selection matters |
| Group Lasso | L2 norm per predefined group | Group-wise | Selects/drops whole feature groups | Categorical dummies, multi-task |
| Fused Lasso | L1 on coeffs + L1 on differences | Yes | Smoothness over ordered features | Signals, genomics (CGH) |
| SCAD / MCP | non-convex folded penalties | Yes | Reduced bias on large coeffs | Less shrinkage of true large effects |
| Bayesian Lasso / Horseshoe | sparsity-inducing priors | Soft | Full posterior + uncertainty | Probabilistic, small-sample inference |

## Tools & libraries

| Tool | Language | Notes | Link |
|---|---|---|---|
| scikit-learn | Python | `Ridge`, `Lasso`, `ElasticNet`, `*CV`, `lars_path`, `enet_path` | https://scikit-learn.org/stable/modules/linear_model.html |
| glmnet | R / Python | Reference coordinate-descent path solver (Friedman/Hastie/Tibshirani) | https://glmnet.stanford.edu/ |
| statsmodels | Python | `OLS.fit_regularized` (elastic-net), GLM penalties | https://www.statsmodels.org/stable/regression.html |
| celer | Python | Fast Lasso/Elastic-Net via dual extrapolation, sklearn-compatible | https://mathurinm.github.io/celer/ |
| scikit-glm / `liblinear` | Python | L1/L2-penalized logistic regression backends | https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression |
| H2O | Python/R/Java | Distributed GLM with elastic-net at scale | https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/glm.html |
| PyMC | Python | Bayesian Lasso / Horseshoe via priors | https://www.pymc.io/ |
| Optuna | Python | Hyperparameter search for `λ`/`α` | https://optuna.org/ |
| CVXPY | Python | Prototype custom penalized convex programs | https://www.cvxpy.org/ |

## Learning resources

| Resource | Type | Link |
|---|---|---|
| *The Elements of Statistical Learning* (Hastie, Tibshirani, Friedman) — Ch. 3 | Book (free PDF) | https://hastie.su.domains/ElemStatLearn/ |
| *An Introduction to Statistical Learning* (ISLR/ISLP) — Ch. 6 | Book (free PDF, R & Python) | https://www.statlearning.com/ |
| *Statistical Learning with Sparsity: The Lasso and Generalizations* (Hastie, Tibshirani, Wainwright) | Book (free PDF) | https://hastie.su.domains/StatLearnSparsity/ |
| *Mathematics for Machine Learning* (Deisenroth, Faisal, Ong) | Book (free PDF) | https://mml-book.github.io/ |
| *Probabilistic Machine Learning* (Murphy) | Book (free PDF) | https://probml.github.io/pml-book/ |
| *Convex Optimization* (Boyd & Vandenberghe) | Book (free PDF) | https://web.stanford.edu/~boyd/cvxbook/ |
| StatQuest — Ridge / Lasso / Elastic Net | Video series | https://www.youtube.com/playlist?list=PLblh5JKOoLUIzaEkCLIUxQFjPIlapw8nU |
| scikit-learn user guide — Linear Models | Tutorial / docs | https://scikit-learn.org/stable/modules/linear_model.html |
| glmnet vignette | Tutorial | https://glmnet.stanford.edu/articles/glmnet.html |

## Key papers

- Hoerl, A.E. & Kennard, R.W. (1970). *Ridge Regression: Biased Estimation for Nonorthogonal Problems.* Technometrics 12(1):55–67. DOI: https://doi.org/10.1080/00401706.1970.10488634
- Tibshirani, R. (1996). *Regression Shrinkage and Selection via the Lasso.* JRSS B 58(1):267–288. DOI: https://doi.org/10.1111/j.2517-6161.1996.tb02080.x
- Efron, Hastie, Johnstone & Tibshirani (2004). *Least Angle Regression.* Annals of Statistics 32(2):407–499. DOI: https://doi.org/10.1214/009053604000000067
- Zou, H. & Hastie, T. (2005). *Regularization and Variable Selection via the Elastic Net.* JRSS B 67(2):301–320. DOI: https://doi.org/10.1111/j.1467-9868.2005.00503.x
- Zou, H. (2006). *The Adaptive Lasso and Its Oracle Properties.* JASA 101(476):1418–1429. DOI: https://doi.org/10.1198/016214506000000735
- Yuan, M. & Lin, Y. (2006). *Model Selection and Estimation in Regression with Grouped Variables.* JRSS B 68(1):49–67. DOI: https://doi.org/10.1111/j.1467-9868.2005.00532.x
- Friedman, J., Hastie, T. & Tibshirani, R. (2010). *Regularization Paths for Generalized Linear Models via Coordinate Descent.* Journal of Statistical Software 33(1):1–22. DOI: https://doi.org/10.18637/jss.v033.i01

## Cross-references in AIForge

- [Linear Regression](../Linear_Regression/) — the unpenalized baseline (OLS) these methods extend
- [Generalized Linear Models](../Generalized_Linear_Models/) — penalized GLMs (logistic, Poisson) reuse the same penalties
- [Resampling and Cross-Validation](../Resampling_and_Cross_Validation/) — how `λ` and `α` are tuned
- [Optimization Algorithms](../../Optimization_Algorithms/) — coordinate descent, LARS, proximal methods behind the solvers
- [Bayesian and Probabilistic ML](../../Bayesian_and_Probabilistic_ML/) — Ridge/Lasso as MAP under Gaussian/Laplace priors
- [Model Evaluation](../../Model_Evaluation/) — bias–variance trade-off and generalization assessment

## Sources

- Tibshirani 1996 (Lasso): https://academic.oup.com/jrsssb/article/58/1/267/7027929 · https://doi.org/10.1111/j.2517-6161.1996.tb02080.x
- Zou & Hastie 2005 (Elastic Net): https://academic.oup.com/jrsssb/article/67/2/301/7109482 · https://doi.org/10.1111/j.1467-9868.2005.00503.x
- Hoerl & Kennard 1970 (Ridge): https://www.tandfonline.com/doi/abs/10.1080/00401706.1970.10488634
- Friedman, Hastie & Tibshirani 2010 (glmnet / coordinate descent): https://www.jstatsoft.org/v33/i01/ · https://doi.org/10.18637/jss.v033.i01
- scikit-learn Linear Models user guide: https://scikit-learn.org/stable/modules/linear_model.html
- glmnet documentation: https://glmnet.stanford.edu/
- ESL (Hastie, Tibshirani, Friedman): https://hastie.su.domains/ElemStatLearn/
- ISLR/ISLP (statlearning.com): https://www.statlearning.com/
- Statistical Learning with Sparsity: https://hastie.su.domains/StatLearnSparsity/
