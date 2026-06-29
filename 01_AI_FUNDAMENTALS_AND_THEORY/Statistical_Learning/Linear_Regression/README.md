# Linear Regression

> A supervised method that models a continuous target as a linear combination of input features, fit by minimizing squared residuals (ordinary least squares).

## Why it matters

Linear regression is the foundational model of statistical learning: it is interpretable, has a closed-form solution, and underpins generalized linear models, regularization, and most of classical econometrics. Mastering its assumptions and diagnostics is a prerequisite for understanding bias-variance tradeoffs, inference, and nearly every more complex predictor. It remains a strong, hard-to-beat baseline on tabular data and a default tool for explanatory modeling.

## Core concepts

- **Model.** Assume `y = Xβ + ε`, where `X` is the `n × p` design matrix (with an intercept column), `β` the coefficient vector, and `ε` mean-zero noise. Predictions are `ŷ = Xβ̂`.
- **Ordinary Least Squares (OLS).** Choose `β̂` to minimize the residual sum of squares `RSS(β) = ‖y − Xβ‖²`. The normal equations give the closed form `β̂ = (XᵀX)⁻¹Xᵀy` (when `XᵀX` is invertible).
- **Geometry.** `ŷ` is the orthogonal projection of `y` onto the column space of `X`; residuals `e = y − ŷ` are orthogonal to every predictor. The hat matrix is `H = X(XᵀX)⁻¹Xᵀ`.
- **Gauss–Markov theorem.** Under linearity, exogeneity (`E[ε|X]=0`), homoscedasticity, and no autocorrelation, OLS is the Best Linear Unbiased Estimator (BLUE) — minimum variance among linear unbiased estimators. Normality of errors additionally licenses exact `t`/`F` inference.
- **Inference.** With `Var(ε)=σ²I`, `Var(β̂) = σ²(XᵀX)⁻¹`; standard errors yield `t`-tests per coefficient, `F`-tests for joint significance, and confidence/prediction intervals.
- **Goodness of fit.** `R² = 1 − RSS/TSS` (variance explained); adjusted `R²` penalizes added predictors. AIC/BIC trade fit against complexity.
- **OLS assumptions.** (1) Linearity in parameters, (2) exogeneity, (3) homoscedasticity, (4) no autocorrelation of errors, (5) no perfect multicollinearity, (6) (for exact inference) normal errors.
- **Diagnostics.** Residuals-vs-fitted (linearity, heteroscedasticity), Q–Q plot (normality), leverage/Cook's distance (influence), Variance Inflation Factor (VIF > 5–10 flags multicollinearity), Durbin–Watson (autocorrelation), Breusch–Pagan / White (heteroscedasticity). When homoscedasticity fails, use **robust (HC) standard errors**.
- **Fitting beyond the normal equations.** SVD or QR decomposition is preferred numerically over inverting `XᵀX`; gradient descent scales to very large `n`/`p`.

## Variants / Methods

| Method | Objective / idea | When to use |
|---|---|---|
| OLS | Minimize `‖y − Xβ‖²` | Baseline; assumptions hold, `p ≪ n` |
| Ridge (L2) | `RSS + α‖β‖²` | Multicollinearity, `p` large; shrinks coefficients |
| Lasso (L1) | `RSS + α‖β‖₁` | Sparse models / feature selection |
| Elastic Net | `RSS + α(ρ‖β‖₁ + (1−ρ)‖β‖²)` | Correlated groups of features |
| Weighted LS (WLS) | Weight observations by `1/variance` | Known heteroscedasticity |
| Generalized LS (GLS) | Model correlated/non-constant error covariance | Autocorrelated or heteroscedastic errors |
| Polynomial / basis expansion | Linear in transformed features (`x², splines`) | Nonlinear relationships, still linear in `β` |
| Robust regression | Huber / RANSAC / quantile loss | Outliers, heavy-tailed noise |
| Bayesian linear regression | Posterior over `β` with priors | Uncertainty quantification, regularization-as-prior |

## Tools & libraries

| Tool | Purpose | URL |
|---|---|---|
| scikit-learn `LinearRegression` | OLS via `scipy.linalg.lstsq`; prediction-focused API | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html |
| scikit-learn linear models guide | Ridge, Lasso, Elastic Net, robust regressors | https://scikit-learn.org/stable/modules/linear_model.html |
| statsmodels `OLS` | Full inference: SEs, `t`/`F`, CIs, robust SEs, diagnostics | https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html |
| statsmodels regression docs | OLS/WLS/GLS overview and examples | https://www.statsmodels.org/stable/regression.html |
| SciPy `linalg.lstsq` | Low-level least-squares solver | https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html |
| NumPy `polyfit` / `lstsq` | Quick polynomial / least-squares fits | https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html |
| PyMC | Bayesian linear regression / GLMs | https://www.pymc.io/ |
| R `stats::lm` | Reference implementation for linear models | https://www.rdocumentation.org/packages/stats/topics/lm |

## Learning resources

- **An Introduction to Statistical Learning (ISLR, James, Witten, Hastie, Tibshirani)** — Ch. 3 on linear regression; free PDF + R/Python labs. https://www.statlearning.com/
- **The Elements of Statistical Learning (ESL, Hastie, Tibshirani, Friedman)** — Ch. 3, deeper theory; free PDF. https://hastie.su.domains/ElemStatLearn/
- **Mathematics for Machine Learning (Deisenroth, Faisal, Ong)** — Ch. 9 linear regression from a probabilistic/linear-algebra view; free PDF. https://mml-book.github.io/
- **Pattern Recognition and Machine Learning (Bishop)** — Ch. 3 linear models for regression; free PDF. https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/
- **StatQuest with Josh Starmer** — intuitive videos on least squares, R², and assumptions. https://www.youtube.com/c/joshstarmer
- **scikit-learn OLS vs Ridge example** — hands-on comparison. https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols_ridge.html
- **MIT 18.650 Statistics for Applications (OCW)** — formal treatment of linear regression and inference. https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/

## Key papers

- Tibshirani, R. (1996). *Regression Shrinkage and Selection via the Lasso.* JRSS-B 58(1), 267–288. https://doi.org/10.1111/j.2517-6161.1996.tb02080.x
- Hoerl, A. E. & Kennard, R. W. (1970). *Ridge Regression: Biased Estimation for Nonorthogonal Problems.* Technometrics 12(1), 55–67. https://doi.org/10.1080/00401706.1970.10488634
- Zou, H. & Hastie, T. (2005). *Regularization and Variable Selection via the Elastic Net.* JRSS-B 67(2), 301–320. https://doi.org/10.1111/j.1467-9868.2005.00503.x
- White, H. (1980). *A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity.* Econometrica 48(4), 817–838. https://doi.org/10.2307/1912934
- Cook, R. D. (1977). *Detection of Influential Observation in Linear Regression.* Technometrics 19(1), 15–18. https://doi.org/10.1080/00401706.1977.10489493
- Akaike, H. (1974). *A New Look at the Statistical Model Identification.* IEEE Trans. Automatic Control 19(6), 716–723. https://doi.org/10.1109/TAC.1974.1100705

## Cross-references in AIForge

- [Regularization: Ridge / Lasso / Elastic Net](../Regularization_Ridge_Lasso_ElasticNet/) — penalized extensions of OLS
- [Generalized Linear Models](../Generalized_Linear_Models/) — linear regression generalized to non-Gaussian targets
- [Logistic Regression](../Logistic_Regression/) — the classification analog
- [Bayesian and Probabilistic ML](../../Bayesian_and_Probabilistic_ML/) — Bayesian linear regression and priors-as-regularization
- [Model Evaluation](../../Model_Evaluation/) — `R²`, RMSE, cross-validated error
- [Optimization Algorithms](../../Optimization_Algorithms/) — gradient-based fitting of large-scale linear models

## Sources

- statsmodels OLS / Linear Regression docs — https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html , https://www.statsmodels.org/stable/regression.html
- scikit-learn LinearRegression & linear models — https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html , https://scikit-learn.org/stable/modules/linear_model.html
- An Introduction to Statistical Learning — https://www.statlearning.com/
- The Elements of Statistical Learning — https://hastie.su.domains/ElemStatLearn/
- Mathematics for Machine Learning — https://mml-book.github.io/
- Tibshirani (1996), Lasso — https://doi.org/10.1111/j.2517-6161.1996.tb02080.x
