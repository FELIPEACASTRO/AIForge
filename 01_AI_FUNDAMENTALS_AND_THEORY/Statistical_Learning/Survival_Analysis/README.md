# Survival Analysis

> Statistical methods for modeling **time-to-event** data in the presence of *censoring* — where the event of interest (death, churn, failure, relapse) has not occurred for some subjects by the end of observation.

## Why it matters

Survival analysis answers "how long until an event, and what drives it?" using data that ordinary regression mishandles: subjects who never experience the event within the study window carry partial (censored) information that must not be discarded or treated as event-free. It is foundational in clinical trials, reliability engineering, customer churn/retention, credit risk, and predictive maintenance, where ignoring censoring biases estimates and inflated risk conclusions.

## Core concepts

Let `T` be a non-negative random variable for the time until an event.

- **Survival function** `S(t) = P(T > t)` — probability of surviving beyond time `t`; monotone decreasing from `S(0)=1`.
- **Hazard function** `h(t) = lim_{dt->0} P(t <= T < t+dt | T >= t) / dt` — the *instantaneous* event rate given survival up to `t`.
- **Cumulative hazard** `H(t) = ∫_0^t h(u) du`, with the fundamental relations `S(t) = exp(-H(t))` and `h(t) = f(t)/S(t)` where `f` is the density.
- **Censoring** — *right* (most common: subject leaves before event), *left*, and *interval* censoring. Standard estimators assume **non-informative** censoring (censoring time independent of event time given covariates).
- **Truncation** — subjects only enter the risk set under a condition (e.g. left truncation / delayed entry).

**Kaplan–Meier (KM)** estimates `S(t)` non-parametrically: `Ŝ(t) = ∏_{t_i <= t} (1 - d_i / n_i)`, where `d_i` events occur among `n_i` at risk at distinct event time `t_i`. Variance via Greenwood's formula. The **Nelson–Aalen** estimator targets `H(t)` analogously. Groups are compared with the **log-rank test**.

**Cox proportional hazards (Cox PH)** is semi-parametric: `h(t | x) = h_0(t) · exp(β^T x)`. The baseline hazard `h_0(t)` is left unspecified; coefficients `β` are estimated by maximizing the **partial likelihood**, so `exp(β_j)` is the **hazard ratio** for covariate `j`. The defining assumption is **proportional hazards** (covariate effects are constant over time, hazard curves do not cross), checkable via Schoenfeld residuals. Ties handled by Breslow or Efron approximations.

**Parametric / AFT models** assume a distribution for `T` (Exponential, Weibull, Log-Normal, Log-Logistic). The **Accelerated Failure Time (AFT)** form models `log T = β^T x + σε`, where covariates scale time directly rather than the hazard.

**Evaluation** uses **Harrell's concordance index (C-index)** — the probability that predicted risk ordering agrees with observed survival ordering (0.5 = random, 1.0 = perfect) — and the **time-dependent Brier score** / Integrated Brier Score for calibration + discrimination.

## Algorithms / Methods

| Method | Type | Models | Key assumption / note |
|---|---|---|---|
| Kaplan–Meier | Non-parametric | `S(t)` | No covariates; descriptive survival curve |
| Nelson–Aalen | Non-parametric | `H(t)` | Cumulative hazard estimator |
| Log-rank test | Non-parametric | Group comparison | Tests equality of survival curves |
| Cox PH | Semi-parametric | `h(t|x)` | Proportional hazards; partial likelihood |
| Cox + time-varying covariates | Semi-parametric | `h(t|x(t))` | Relaxes static-covariate constraint |
| Stratified / penalized Cox (Coxnet) | Semi-parametric | `h(t|x)` | Strata for PH violation; elastic-net for high-dim |
| Weibull / Exponential / Log-Normal AFT | Parametric | `log T` | Distributional assumption; extrapolation |
| Random Survival Forests | ML / ensemble | `H(t|x)`, `S(t|x)` | Non-linear, handles interactions, no PH needed |
| Gradient-boosted survival (Cox/AFT loss) | ML / boosting | risk score | XGBoost/scikit-survival GBM |
| DeepSurv / DeepHit | Deep learning | risk / discrete `S(t)` | Neural nets for non-linear effects, competing risks |
| Competing risks (Fine–Gray) | Sub-distribution | CIF | Multiple mutually exclusive event types |

## Tools & libraries

| Library | Language | Scope | URL |
|---|---|---|---|
| lifelines | Python | KM, Nelson–Aalen, Cox PH, parametric/AFT, plotting | https://lifelines.readthedocs.io/ |
| scikit-survival | Python | sklearn-compatible Cox, Coxnet, RSF, GBM, SVM, metrics | https://scikit-survival.readthedocs.io/ |
| statsmodels (duration) | Python | Cox PHReg, KM (`SurvfuncRight`) | https://www.statsmodels.org/stable/duration.html |
| scikit-learn | Python | Base estimators/pipelines (no native survival) | https://scikit-learn.org/ |
| XGBoost (`survival:cox`, AFT) | Python/R/C++ | Gradient-boosted Cox & AFT objectives | https://xgboost.readthedocs.io/en/stable/tutorials/aft_survival_analysis.html |
| pycox | Python (PyTorch) | DeepSurv, DeepHit, Cox-Time | https://github.com/havakv/pycox |
| auton-survival | Python | DSM, regression, counterfactual, evaluation | https://github.com/autonlab/auton-survival |
| PyMC | Python | Bayesian survival / parametric hazard models | https://www.pymc.io/ |
| survival (R) | R | Reference implementation (`coxph`, `survfit`) | https://cran.r-project.org/package=survival |
| randomForestSRC (R) | R | Random Survival Forests (canonical) | https://cran.r-project.org/package=randomForestSRC |

## Learning resources

- **The Elements of Statistical Learning** (Hastie, Tibshirani, Friedman) — Ch. on survival / Cox regression; free PDF: https://hastie.su.domains/ElemStatLearn/
- **An Introduction to Statistical Learning** (James, Witten, Hastie, Tibshirani) — Ch. 11 "Survival Analysis and Censored Data"; free PDF: https://www.statlearning.com/
- **Survival Analysis: A Self-Learning Text** (Kleinbaum & Klein) — applied, intuition-first standard text: https://link.springer.com/book/10.1007/978-1-4419-6646-9
- **Modeling Survival Data: Extending the Cox Model** (Therneau & Grambsch) — the definitive companion to the R `survival` package: https://link.springer.com/book/10.1007/978-1-4757-3294-8
- **scikit-survival user guide** — practical, code-first introduction to time-to-event ML: https://scikit-survival.readthedocs.io/en/stable/user_guide/00-introduction.html
- **lifelines "Introduction to survival analysis"** — gentle, runnable tutorial: https://lifelines.readthedocs.io/en/latest/Survival%20Analysis%20intro.html
- **StatQuest (Josh Starmer)** — video intros to survival curves, KM, and hazard ratios: https://www.youtube.com/c/joshstarmer

## Key papers

- Kaplan, E. L. & Meier, P. (1958). *Nonparametric Estimation from Incomplete Observations.* JASA 53(282):457–481. https://doi.org/10.1080/01621459.1958.10501452
- Cox, D. R. (1972). *Regression Models and Life-Tables.* J. R. Stat. Soc. B 34(2):187–220. https://doi.org/10.1111/j.2517-6161.1972.tb00899.x
- Harrell, F. E. et al. (1982). *Evaluating the Yield of Medical Tests* (C-index). JAMA 247(18):2543–2546. https://doi.org/10.1001/jama.1982.03320430047030
- Fine, J. P. & Gray, R. J. (1999). *A Proportional Hazards Model for the Subdistribution of a Competing Risk.* JASA 94(446):496–509. https://doi.org/10.1080/01621459.1999.10474144
- Ishwaran, H., Kogalur, U. B., Blackstone, E. H. & Lauer, M. S. (2008). *Random Survival Forests.* Annals of Applied Statistics 2(3):841–860. https://doi.org/10.1214/08-AOAS169
- Katzman, J. L. et al. (2018). *DeepSurv: Personalized Treatment Recommender System Using a Cox Proportional Hazards Deep Neural Network.* BMC Med. Res. Methodol. 18:24. https://doi.org/10.1186/s12874-018-0482-1 (arXiv: https://arxiv.org/abs/1606.00931)

## Cross-references in AIForge

- [Generalized Linear Models](../Generalized_Linear_Models/) — shared exponential-family and link-function machinery underlying parametric hazard/AFT models.
- [Statistical Time Series](../Statistical_Time_Series/) — sibling temporal-modeling toolkit; contrast hazard modeling with forecasting.
- [Model Evaluation](../../Model_Evaluation/) — context for C-index, Brier score, and calibration metrics.
- [Bayesian and Probabilistic ML](../../Bayesian_and_Probabilistic_ML/) — priors and posterior inference for Bayesian survival models (e.g. PyMC).

## Sources

- lifelines documentation — https://lifelines.readthedocs.io/
- scikit-survival documentation — https://scikit-survival.readthedocs.io/en/stable/user_guide/00-introduction.html
- statsmodels duration / PHReg — https://www.statsmodels.org/stable/duration.html
- XGBoost AFT survival tutorial — https://xgboost.readthedocs.io/en/stable/tutorials/aft_survival_analysis.html
- An Introduction to Statistical Learning — https://www.statlearning.com/
- The Elements of Statistical Learning — https://hastie.su.domains/ElemStatLearn/
- Kaplan & Meier (1958), JASA — https://doi.org/10.1080/01621459.1958.10501452
- Cox (1972), JRSS-B — https://doi.org/10.1111/j.2517-6161.1972.tb00899.x
- Ishwaran et al. (2008), AOAS — https://doi.org/10.1214/08-AOAS169
- Katzman et al. (2018), DeepSurv — https://doi.org/10.1186/s12874-018-0482-1
