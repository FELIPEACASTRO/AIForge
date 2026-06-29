# Model Selection and Validation

> The disciplined process of choosing among competing models (and their hyperparameters) and rigorously estimating how well the chosen model will generalize to unseen data.

## Why it matters

Picking the model that scores best on the training set — or even on a single held-out split — reliably overestimates real-world performance because of overfitting and selection bias. Sound validation (proper data splitting, cross-validation, learning-curve diagnosis, and statistically honest comparison) is what separates a model that *looks* good from one that *is* good. Getting this wrong is one of the most common and most expensive mistakes in applied ML.

## Core concepts

**Generalization and the train/validation/test split.** The goal is to minimize *expected* (population) risk, not empirical (training) risk. A model is fit on the training set, hyperparameters are tuned on a validation set, and a final, untouched test set gives an unbiased estimate of generalization. Touching the test set during selection contaminates it.

**Bias–variance decomposition.** For squared-error loss the expected error at a point decomposes as:

`E[(y − f̂(x))²] = Bias[f̂(x)]² + Var[f̂(x)] + σ²`

- **Bias** = error from wrong assumptions / too-simple models (underfitting).
- **Variance** = sensitivity to the particular training sample (overfitting).
- **σ²** = irreducible noise.

Increasing model complexity lowers bias but raises variance; the sweet spot minimizes total error. (Modern over-parameterized networks complicate the classic U-curve via *double descent*.)

**Cross-validation (CV).** Reuse data by partitioning into *k* folds, training on *k−1* and testing on the held-out fold, then averaging. *k*-fold CV trades bias against variance and computation; **10-fold stratified CV** is the long-standing default (Kohavi, 1995). Leave-one-out (LOO) is nearly unbiased but high-variance and expensive. **Crucially, there is no universal unbiased estimator of the variance of *k*-fold CV** (Bengio & Grandvalet, 2004), so confidence statements need care.

**Nested CV.** When hyperparameters are tuned *and* performance is estimated, a single CV loop leaks information and biases the estimate optimistically (Cawley & Talbot, 2010). **Nested CV** uses an inner loop for tuning and an outer loop for unbiased evaluation.

**Learning curves.** Plot training and validation error vs. training-set size. *High bias*: both curves converge to a high error (more data won't help — add capacity/features). *High variance*: a large gap persists (more data, regularization, or simpler model will help).

**Statistical comparison.** A difference in mean CV scores may be noise. Use paired tests that respect the data, and correct for multiplicity when comparing many models/datasets.

## Methods / Variants

| Technique | What it does | When to use | Caveats |
|---|---|---|---|
| Hold-out (train/val/test) | Single fixed split | Very large datasets; quick checks | High-variance estimate on small data |
| k-fold CV | Average over k rotations | Default for small–medium data | Folds not independent |
| Stratified k-fold | Preserves class proportions per fold | Classification, imbalanced classes | — |
| Leave-One-Out (LOO) | k = n | Tiny datasets | Expensive; high variance |
| Repeated k-fold | Multiple shuffled k-fold runs | Tighter, more stable estimates | More compute |
| Group k-fold | Keeps groups (e.g., patients) in one fold | Grouped/clustered data | Needs group labels |
| Time-series split | Forward-chaining, no future leakage | Temporal/sequential data | No shuffling |
| Nested CV | Inner tune + outer evaluate | Unbiased estimate with tuning | k×k cost |
| Bootstrap / .632+ | Resample with replacement | Small data, CI estimation | Optimism bias needs correction |
| Information criteria (AIC/BIC/MDL) | Penalize complexity analytically | Likelihood models, no CV budget | Assumptions on likelihood |
| Paired t-test (10-fold) | Compare two algorithms | Quick two-model comparison | Inflated Type I error if misused |
| 5×2cv paired t / McNemar | Lower Type-I-error comparisons | Comparing two classifiers | See Dietterich (1998) |
| Wilcoxon / Friedman + Nemenyi | Non-parametric multi-model comparison | Many models over many datasets | Demšar (2006) recommended |

## Tools & libraries

| Tool | Use | URL |
|---|---|---|
| scikit-learn `model_selection` | CV splitters, grid/random search, learning curves | https://scikit-learn.org/stable/model_selection.html |
| scikit-learn cross-validation guide | Authoritative CV reference + code | https://scikit-learn.org/stable/modules/cross_validation.html |
| Optuna | Define-by-run hyperparameter optimization | https://optuna.org/ |
| Hyperopt | TPE / Bayesian hyperparameter search | http://hyperopt.github.io/hyperopt/ |
| Ray Tune | Distributed/scalable HPO | https://docs.ray.io/en/latest/tune/index.html |
| statsmodels | AIC/BIC, statistical tests | https://www.statsmodels.org/ |
| SciPy `stats` | Wilcoxon, Friedman, t-tests | https://docs.scipy.org/doc/scipy/reference/stats.html |
| MLxtend | 5×2cv, McNemar, bias-variance decomp, CD diagrams | https://rasbt.github.io/mlxtend/ |
| Autorank | Automated Demšar-style multi-model comparison | https://github.com/sherbold/autorank |
| XGBoost (`cv`) | Built-in CV with early stopping | https://xgboost.readthedocs.io/en/stable/python/python_api.html |
| PyMC | Bayesian model comparison (WAIC, LOO-CV) | https://www.pymc.io/ |
| ArviZ | Bayesian model comparison & LOO/WAIC | https://python.arviz.org/ |

## Learning resources

| Resource | Type | Link |
|---|---|---|
| *The Elements of Statistical Learning* (Hastie, Tibshirani, Friedman) — Ch. 7 Model Assessment & Selection | Book (free PDF) | https://hastie.su.domains/ElemStatLearn/ |
| *An Introduction to Statistical Learning* (ISLR/ISLP) — Ch. 5 Resampling | Book (free PDF + labs) | https://www.statlearning.com/ |
| *Mathematics for Machine Learning* | Book (free PDF) | https://mml-book.github.io/ |
| *Probabilistic Machine Learning* (Murphy) | Book (free PDF) | https://probml.github.io/pml-book/ |
| scikit-learn — Cross-validation user guide | Tutorial | https://scikit-learn.org/stable/modules/cross_validation.html |
| scikit-learn — Underfitting vs. overfitting / validation & learning curves | Tutorial | https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html |
| StatQuest — Cross Validation / Bias & Variance | Video | https://www.youtube.com/watch?v=fSytzGwwBVw |
| Andrew Ng — Machine Learning (advice for applying ML) | Course | https://www.coursera.org/specializations/machine-learning-introduction |
| MLxtend — Model evaluation guide | Tutorial | https://rasbt.github.io/mlxtend/user_guide/evaluate/ |

## Key papers

- Kohavi, R. (1995). *A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection.* IJCAI. https://www.semanticscholar.org/paper/8c70a0a39a686bf80b76cb1b77f9eef156f6432d
- Dietterich, T. G. (1998). *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms.* Neural Computation 10(7):1895–1923. https://doi.org/10.1162/089976698300017197
- Bengio, Y. & Grandvalet, Y. (2004). *No Unbiased Estimator of the Variance of K-Fold Cross-Validation.* JMLR 5:1089–1105. https://www.jmlr.org/papers/v5/grandvalet04a.html
- Demšar, J. (2006). *Statistical Comparisons of Classifiers over Multiple Data Sets.* JMLR 7:1–30. https://jmlr.org/papers/v7/demsar06a.html
- Cawley, G. C. & Talbot, N. L. C. (2010). *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation.* JMLR 11:2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html
- Arlot, S. & Celisse, A. (2010). *A Survey of Cross-Validation Procedures for Model Selection.* Statistics Surveys 4:40–79. https://doi.org/10.1214/09-SS054
- Belkin, M. et al. (2019). *Reconciling Modern Machine-Learning Practice and the Classical Bias–Variance Trade-off.* PNAS 116(32). https://doi.org/10.1073/pnas.1903070116

## Cross-references in AIForge

- [Model Evaluation (parent)](../) — metrics, calibration, error analysis
- [Machine Learning](../../Machine_Learning/) — supervised/unsupervised foundations
- [Optimization Algorithms](../../Optimization_Algorithms/) — hyperparameter optimization, search
- [Bayesian and Probabilistic ML](../../Bayesian_and_Probabilistic_ML/) — WAIC, LOO-CV, model comparison

## Sources

- scikit-learn — Model selection and evaluation: https://scikit-learn.org/stable/model_selection.html
- scikit-learn — Cross-validation: https://scikit-learn.org/stable/modules/cross_validation.html
- Kohavi (1995): https://www.semanticscholar.org/paper/8c70a0a39a686bf80b76cb1b77f9eef156f6432d
- Dietterich (1998): https://doi.org/10.1162/089976698300017197
- Bengio & Grandvalet (2004): https://www.jmlr.org/papers/v5/grandvalet04a.html
- Demšar (2006): https://jmlr.org/papers/v7/demsar06a.html
- Cawley & Talbot (2010): https://www.jmlr.org/papers/v11/cawley10a.html
- Arlot & Celisse (2010): https://doi.org/10.1214/09-SS054
- Belkin et al. (2019), double descent: https://doi.org/10.1073/pnas.1903070116
- ESL: https://hastie.su.domains/ElemStatLearn/ · ISLR: https://www.statlearning.com/
