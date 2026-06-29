# Linear and Quadratic Discriminant Analysis

> Generative classifiers that model each class as a Gaussian density and apply Bayes' rule, yielding linear (LDA) or quadratic (QDA) decision boundaries in closed form.

## Why it matters

Discriminant analysis is one of the oldest and most reliable classification methods: it has a closed-form solution, no hyperparameters to tune in its basic form, is naturally multiclass, and often performs surprisingly well on tabular data. LDA doubles as a supervised dimensionality-reduction technique (projecting to at most `C-1` dimensions for `C` classes), making it a workhorse baseline and a stepping stone to understanding Gaussian generative models and Bayes-optimal classification.

## Core concepts

**Generative setup.** Both methods model the class-conditional density `P(x | y=k)` as a multivariate Gaussian `N(μ_k, Σ_k)` and combine it with class priors `π_k` via Bayes' rule:

`P(y=k | x) ∝ π_k · N(x; μ_k, Σ_k)`.

Classification assigns `x` to the class maximizing the posterior, equivalently maximizing the **discriminant function** `δ_k(x)`.

**QDA.** Each class keeps its own covariance `Σ_k`. The log-discriminant is

`δ_k(x) = -½ log|Σ_k| - ½ (x - μ_k)ᵀ Σ_k⁻¹ (x - μ_k) + log π_k`.

The quadratic term in `x` does not cancel between classes, so the boundary `δ_k(x) = δ_l(x)` is **quadratic**.

**LDA.** Assume a **shared** covariance `Σ_k = Σ` for all classes. The quadratic term cancels, leaving

`δ_k(x) = xᵀ Σ⁻¹ μ_k - ½ μ_kᵀ Σ⁻¹ μ_k + log π_k`,

which is **linear** in `x`; decision boundaries are hyperplanes.

**Estimation.** Parameters are the maximum-likelihood plug-ins: `π_k = n_k/n`, class means `μ_k`, and the (pooled, for LDA) sample covariance.

**Fisher's linear discriminant.** Fisher (1936) derived the LDA projection without Gaussian assumptions: find directions `w` that maximize the **Rayleigh quotient** `(wᵀ S_B w) / (wᵀ S_W w)`, the ratio of between-class scatter `S_B` to within-class scatter `S_W`. The solution is the leading eigenvectors of `S_W⁻¹ S_B`, giving a rank-`(C-1)` discriminative subspace — this is LDA's dimensionality-reduction view, and it coincides with the Gaussian classifier when class covariances are equal.

**Bias–variance trade-off.** QDA estimates `O(C·d²)` covariance parameters vs. `O(d²)` for LDA, so QDA is more flexible but needs more data; LDA is more robust when `n` is small relative to `d`. Regularization (RDA) interpolates between them and toward the identity.

## Variants

| Variant | Boundary | Covariance assumption | When to use |
|---|---|---|---|
| **LDA** | Linear | Shared `Σ` across classes | Baseline; small-to-medium `n`, roughly equal covariances |
| **QDA** | Quadratic | Per-class `Σ_k` | Classes with clearly different spreads/orientations; enough data |
| **Fisher LDA (dim. reduction)** | Linear projection | Within/between scatter | Supervised projection to `≤ C-1` dims, visualization |
| **Regularized DA (RDA)** | Linear↔quadratic | Shrink `Σ_k` toward pooled `Σ` and toward `σ²I` (params `λ, γ`) | High-dim / small-sample; tune flexibility (Friedman 1989) |
| **Shrinkage LDA** | Linear | Ledoit–Wolf / OAS shrinkage of `Σ` | `d` large relative to `n`; ill-conditioned covariance |
| **Naive Bayes (Gaussian)** | Linear/quadratic | Diagonal `Σ_k` | Extreme small-sample; treats features as independent |
| **Flexible DA (FDA)** | Nonlinear | LDA via nonparametric optimal scoring/regression | Nonlinear boundaries (Hastie–Tibshirani–Buja 1994) |
| **Penalized DA (PDA)** | Linear | Penalized within-class scatter | Many correlated predictors (images, signals) |
| **Mixture DA (MDA)** | Nonlinear | Each class = Gaussian mixture | Multimodal class densities |

## Tools & libraries

| Tool | What it provides | URL |
|---|---|---|
| scikit-learn | `LinearDiscriminantAnalysis`, `QuadraticDiscriminantAnalysis` (SVD/LSQR/eigen solvers, shrinkage) | https://scikit-learn.org/stable/modules/lda_qda.html |
| statsmodels | Multivariate statistics / MANOVA tooling complementing DA | https://www.statsmodels.org/stable/index.html |
| MASS (R) | `lda()`, `qda()` — classic reference implementations | https://cran.r-project.org/package=MASS |
| klaR (R) | `rda()` Regularized Discriminant Analysis | https://cran.r-project.org/package=klaR |
| mda (R) | Mixture & Flexible Discriminant Analysis (`fda`, `mda`) | https://cran.r-project.org/package=mda |
| tidymodels / discrim | `discrim_linear`, `discrim_quad`, `discrim_regularized` | https://parsnip.tidymodels.org/reference/discrim_regularized.html |
| Optuna | Hyperparameter search for RDA/shrinkage parameters | https://optuna.org/ |

## Learning resources

| Resource | Type | Link |
|---|---|---|
| *The Elements of Statistical Learning* (Hastie, Tibshirani, Friedman), Ch. 4.3 | Book (free PDF) | https://hastie.su.domains/ElemStatLearn/ |
| *An Introduction to Statistical Learning* (ISLR/ISLP), Ch. 4 | Book (free PDF) | https://www.statlearning.com/ |
| *Mathematics for Machine Learning* (Deisenroth, Faisal, Ong) | Book (free PDF) | https://mml-book.github.io/ |
| *Probabilistic Machine Learning* (Murphy), Gaussian discriminant analysis | Book (free PDF) | https://probml.github.io/pml-book/ |
| scikit-learn LDA/QDA user guide + covariance-ellipsoid example | Tutorial | https://scikit-learn.org/stable/auto_examples/classification/plot_lda_qda.html |
| StatQuest — LDA clearly explained (Josh Starmer) | Video | https://www.youtube.com/watch?v=azXCzI57Yfc |
| Machine Learning Mastery — LDA with Python | Tutorial | https://machinelearningmastery.com/linear-discriminant-analysis-with-python/ |

## Key papers

| Year | Paper | Link |
|---|---|---|
| 1936 | Fisher, R. A. — *The Use of Multiple Measurements in Taxonomic Problems* (origin of linear discriminant; Iris data) | https://onlinelibrary.wiley.com/doi/10.1111/j.1469-1809.1936.tb02137.x |
| 1948 | Rao, C. R. — *The Utilization of Multiple Measurements in Problems of Biological Classification* (multiclass generalization) | https://doi.org/10.1111/j.2517-6161.1948.tb00008.x |
| 1989 | Friedman, J. H. — *Regularized Discriminant Analysis* (JASA 84:165–175) | https://www.tandfonline.com/doi/abs/10.1080/01621459.1989.10478752 |
| 1994 | Hastie, Tibshirani, Buja — *Flexible Discriminant Analysis by Optimal Scoring* (JASA) | https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476866 |
| 1995 | Hastie, Buja, Tibshirani — *Penalized Discriminant Analysis* (Annals of Statistics 23:73–102) | https://projecteuclid.org/journals/annals-of-statistics/volume-23/issue-1/Penalized-Discriminant-Analysis/10.1214/aos/1176324456.full |
| 1996 | Hastie, Tibshirani — *Discriminant Analysis by Gaussian Mixtures* (JRSS-B) | https://doi.org/10.1111/j.2517-6161.1996.tb02073.x |

## Cross-references in AIForge

- [Machine_Learning](../../Machine_Learning/) — broader supervised learning context and baselines
- [Bayesian_and_Probabilistic_ML](../../Bayesian_and_Probabilistic_ML/) — generative modeling, Bayes' rule, Gaussian densities
- [Naive_Bayes](../Naive_Bayes/) — closely related generative classifier (diagonal-covariance special case)
- [Dimensionality_Reduction](../Dimensionality_Reduction/) — Fisher LDA as supervised projection vs. PCA
- [Model_Evaluation](../../Model_Evaluation/) — assessing classifier accuracy and misclassification risk

## Sources

- scikit-learn — Linear and Quadratic Discriminant Analysis: https://scikit-learn.org/stable/modules/lda_qda.html
- Fisher (1936), Annals of Eugenics (DOI 10.1111/j.1469-1809.1936.tb02137.x): https://onlinelibrary.wiley.com/doi/10.1111/j.1469-1809.1936.tb02137.x
- Friedman (1989), Regularized Discriminant Analysis (JASA): https://www.tandfonline.com/doi/abs/10.1080/01621459.1989.10478752
- Hastie, Buja, Tibshirani (1995), Penalized Discriminant Analysis (Project Euclid): https://projecteuclid.org/journals/annals-of-statistics/volume-23/issue-1/Penalized-Discriminant-Analysis/10.1214/aos/1176324456.full
- The Elements of Statistical Learning, Ch. 4: https://hastie.su.domains/ElemStatLearn/
- An Introduction to Statistical Learning: https://www.statlearning.com/
