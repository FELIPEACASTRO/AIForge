# Resampling: Bootstrap and Cross-Validation

> Resampling repeatedly draws subsets from the data to estimate the sampling distribution of a statistic and to obtain honest, out-of-sample estimates of model generalization error.

## Why it matters

Resampling lets you quantify uncertainty (standard errors, confidence intervals) and estimate predictive performance **without** strong parametric assumptions or a separate held-out test set you can rarely afford. It is the backbone of honest model evaluation, hyperparameter selection, and model comparison — and getting it wrong (data leakage, dependence, double-dipping) is one of the most common causes of over-optimistic benchmarks in applied ML.

## Core concepts

- **Plug-in / empirical distribution.** The bootstrap replaces the unknown population distribution `F` with the empirical distribution `F̂` (the observed sample, each point weight `1/n`). Any functional `θ = t(F)` is estimated by `θ̂ = t(F̂)`, and its variability is approximated by resampling from `F̂`.
- **Bootstrap.** Draw `B` resamples of size `n` **with replacement** from the data; recompute the statistic on each to get `θ̂*₁,…,θ̂*_B`. Then `SE_boot = sd(θ̂*)`, and percentile / BCa intervals come from the quantiles of the `θ̂*` distribution. A given observation is omitted from a resample with probability `(1 − 1/n)^n → e⁻¹ ≈ 0.368`, so each resample contains ≈ 63.2% unique points — the basis of the **.632** estimator.
- **Cross-validation (CV).** Partition data into `K` folds; train on `K−1`, validate on the held-out fold, rotate, and average the fold errors: `CV(K) = (1/K) Σ_k L(fold_k)`. **Leave-one-out (LOOCV)** is `K = n`.
- **Bias–variance of K.** Small `K` (e.g. 5) → higher bias (smaller training sets), lower variance, cheaper. Large `K`/LOOCV → low bias, but high variance and `n` refits. `K = 5` or `10` is the standard pragmatic compromise.
- **Independence assumption.** Standard CV/bootstrap assume i.i.d. (exchangeable) data. Violations — temporal autocorrelation, grouped/clustered records, identical entities across rows — leak information between train and validation and inflate scores. Use grouped/time-aware variants instead.
- **Selection bias / double-dipping.** Using the *same* CV loop to both tune hyperparameters and report performance optimistically biases the estimate. **Nested CV** separates tuning (inner loop) from evaluation (outer loop) to yield a nearly unbiased generalization estimate.

## Variants

| Method | Idea | Best for | Caveats |
|---|---|---|---|
| **Validation set (holdout)** | Single train/test split | Very large data; quick checks | High-variance estimate; wastes data |
| **k-Fold CV** | K rotating folds, average error | General-purpose evaluation | Assumes i.i.d.; choose K=5/10 |
| **Stratified k-Fold** | Preserve class proportions per fold | Classification, imbalanced labels | Stratify on the right variable |
| **Repeated k-Fold** | Repeat k-fold with new splits | Reduce estimate variance | Linear cost increase |
| **Leave-One-Out (LOOCV)** | K = n | Small n, low-bias need | High variance; n refits |
| **Leave-P-Out / LPGO** | Hold out p samples / p groups | Small, structured data | Combinatorially expensive |
| **Group k-Fold** | No group spans train+test | Clustered/repeated subjects | Need a valid group key |
| **Time-Series Split (forward chaining)** | Expanding/rolling window, train on past | Forecasting, temporal data | Never shuffle; respect order |
| **Blocked / purged + embargo CV** | Remove leakage-adjacent samples | Financial / autocorrelated series | Tune gap/embargo size |
| **Nested CV** | Inner loop tunes, outer loop scores | Honest perf. with model selection | Expensive (loops multiply) |
| **Bootstrap (.632 / .632+)** | Resample-with-replacement error estimate | SE/CI, small samples | .632+ corrects optimism/overfit bias |
| **Out-of-bag (OOB)** | Score on samples not in a bootstrap | Bagging / random forests (free CV) | Only for bagged ensembles |

## Tools & libraries

| Tool | What it provides | URL |
|---|---|---|
| scikit-learn `model_selection` | KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit, cross_val_score, GridSearchCV (nested CV) | https://scikit-learn.org/stable/modules/cross_validation.html |
| statsmodels | Bootstrap utilities, statistical inference | https://www.statsmodels.org/ |
| SciPy `scipy.stats.bootstrap` | BCa/percentile bootstrap confidence intervals | https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html |
| mlxtend | `bootstrap_point632_score`, bootstrap & CV evaluators | https://rasbt.github.io/mlxtend/user_guide/evaluate/bootstrap_point632_score/ |
| XGBoost / LightGBM `cv` | Built-in k-fold CV with early stopping | https://xgboost.readthedocs.io/en/stable/python/python_api.html |
| Optuna | Hyperparameter search wrapping CV objectives | https://optuna.org/ |
| MAPIE | Conformal / resampling-based prediction intervals | https://mapie.readthedocs.io/ |
| arch (`arch.bootstrap`) | Block/stationary bootstrap for dependent data | https://bashtage.github.io/arch/bootstrap/bootstrap.html |
| caret / rsample (R / tidymodels) | Resampling objects, nested CV, bootstrap (R) | https://rsample.tidymodels.org/ |
| PyMC | Bayesian inference (posterior-based alternative to bootstrap) | https://www.pymc.io/ |

## Learning resources

- **The Elements of Statistical Learning (ESL)** — Hastie, Tibshirani, Friedman, Ch. 7 (model assessment & selection, CV, bootstrap). Free PDF: https://hastie.su.domains/ElemStatLearn/
- **An Introduction to Statistical Learning (ISLR / ISLP)** — James, Witten, Hastie, Tibshirani, Ch. 5 (Resampling Methods); Python + R editions. Free PDF: https://www.statlearning.com/
- **An Introduction to the Bootstrap** — Efron & Tibshirani (1993), the canonical book: https://doi.org/10.1201/9780429246593
- **Computer Age Statistical Inference** — Efron & Hastie (bootstrap, jackknife, CV in context). Free PDF: https://hastie.su.domains/CASI/
- **Probabilistic Machine Learning** — Kevin Murphy (evaluation, model selection). Free: https://probml.github.io/pml-book/
- **scikit-learn user guide: Cross-validation** — practical, code-first: https://scikit-learn.org/stable/modules/cross_validation.html
- **StatQuest (Josh Starmer)** — Cross Validation, intuitive video: https://www.youtube.com/watch?v=fSytzGwwBVw
- **Rob Hyndman, *Forecasting: Principles and Practice*** — time-series CV (tsCV, rolling origin): https://otexts.com/fpp3/

## Key papers

- Stone, M. (1974). *Cross-Validatory Choice and Assessment of Statistical Predictions.* JRSS B 36(2):111–147. https://doi.org/10.1111/j.2517-6161.1974.tb00994.x
- Efron, B. (1979). *Bootstrap Methods: Another Look at the Jackknife.* Annals of Statistics 7(1):1–26. https://doi.org/10.1214/aos/1176344552
- Efron, B. & Tibshirani, R. (1997). *Improvements on Cross-Validation: The .632+ Bootstrap Method.* JASA 92(438):548–560. https://doi.org/10.2307/2965703
- Kohavi, R. (1995). *A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection.* IJCAI. https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf
- Varma, S. & Simon, R. (2006). *Bias in Error Estimation When Using Cross-Validation for Model Selection.* BMC Bioinformatics 7:91 (nested CV). https://doi.org/10.1186/1471-2105-7-91
- Cawley, G. & Talbot, N. (2010). *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation.* JMLR 11:2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html
- Bergmeir, C., Hyndman, R.J. & Koo, B. (2018). *A Note on the Validity of Cross-Validation for Evaluating Autoregressive Time Series Prediction.* CSDA 120:70–83. https://doi.org/10.1016/j.csda.2017.11.003
- Bates, S., Hastie, T. & Tibshirani, R. (2024). *Cross-Validation: What Does It Estimate and How Well Does It Do It?* JASA. https://doi.org/10.1080/01621459.2023.2197686 (preprint: https://arxiv.org/abs/2104.00673)

## Cross-references in AIForge

- [Model Evaluation](../../Model_Evaluation/) — metrics, validation strategy, error analysis
- [Machine Learning](../../Machine_Learning/) — estimators these methods evaluate and tune
- [Bayesian and Probabilistic ML](../../Bayesian_and_Probabilistic_ML/) — posterior inference as an alternative to resampling
- [Optimization Algorithms](../../Optimization_Algorithms/) — hyperparameter search wrapped around CV objectives

## Sources

- Efron (1979), Annals of Statistics: https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.full
- Stone (1974), JRSS B: https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1974.tb00994.x
- Efron & Tibshirani (1997), JASA .632+: https://www.tandfonline.com/doi/abs/10.1080/01621459.1997.10474007
- Varma & Simon (2006), BMC Bioinformatics: https://link.springer.com/article/10.1186/1471-2105-7-91
- Bergmeir, Hyndman & Koo (2018), CSDA: https://robjhyndman.com/publications/cv-time-series/
- scikit-learn cross-validation user guide: https://scikit-learn.org/stable/modules/cross_validation.html
- ESL & ISLR: https://hastie.su.domains/ElemStatLearn/ , https://www.statlearning.com/
- mlxtend .632+ bootstrap: https://rasbt.github.io/mlxtend/user_guide/evaluate/bootstrap_point632_score/
