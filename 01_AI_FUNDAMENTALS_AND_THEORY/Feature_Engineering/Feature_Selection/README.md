# Feature Selection

> Feature selection is the process of choosing a subset of relevant input variables for model construction, discarding redundant, irrelevant, or noisy features to improve accuracy, interpretability, and efficiency.

## Why it matters

High-dimensional data inflates variance, slows training/inference, and invites overfitting through spurious correlations (the curse of dimensionality). Removing uninformative features sharpens generalization, shrinks models, lowers data-collection cost, and makes the remaining signal interpretable to humans. Unlike dimensionality reduction (PCA, autoencoders), feature *selection* keeps original, named variables — preserving semantics for explainability, auditing, and regulatory contexts.

## Core concepts

- **Relevance vs. redundancy.** A feature can be individually predictive (relevant) yet add nothing once correlated features are present (redundant). Good selection maximizes joint relevance while minimizing redundancy.
- **Strong vs. weak relevance.** A feature is *strongly relevant* if removing it degrades the optimal predictor; *weakly relevant* if it helps only in some contexts. "All-relevant" methods (Boruta) keep both; "minimal-optimal" methods seek the smallest sufficient subset.
- **Mutual information.** A model-free dependency measure capturing linear *and* nonlinear relations: `I(X;Y) = H(Y) − H(Y|X) = ΣΣ p(x,y) log( p(x,y) / (p(x)p(y)) )`. Zero iff `X` and `Y` are independent. Basis for many filter scores.
- **mRMR criterion.** Select feature set `S` maximizing `max [ (1/|S|) Σ I(x_i;y) − (1/|S|²) Σ I(x_i;x_j) ]` — relevance to target minus pairwise redundancy.
- **Embedded sparsity.** L1 (Lasso) penalty `min ‖y − Xβ‖² + λ‖β‖₁` drives coefficients to exactly zero, selecting features during fitting. Elastic Net adds an L2 term for grouped/correlated features.
- **Shadow features (Boruta).** Permuted copies of real features create a noise baseline; a real feature is kept only if its importance significantly exceeds the best shadow's, tested over many iterations.
- **Stability & leakage.** Selection must occur *inside* cross-validation folds; selecting on the full dataset before CV leaks the target and inflates scores. Stability (consistency of the chosen subset across resamples) is a key quality metric.

## Algorithms / Methods

| Method | Category | Idea | Pros | Cons |
|---|---|---|---|---|
| Variance threshold | Filter | Drop near-constant features | Trivial, fast | Ignores target |
| Correlation / Chi² / ANOVA F-test | Filter | Univariate statistical score vs. target | Fast, scalable | Misses interactions; assumptions on data |
| Mutual information | Filter | Nonlinear dependency `I(X;Y)` | Captures nonlinearity | Density estimation noisy; univariate |
| mRMR | Filter | Max relevance − min redundancy | Handles redundancy | Greedy; MI estimation cost |
| Relief / ReliefF | Filter | Weight features by neighbor contrast | Detects interactions | Sensitive to noise/distance metric |
| Forward / Backward selection | Wrapper | Greedy add/remove by CV score | Model-aware | Expensive; greedy local optima |
| Recursive Feature Elimination (RFE) | Wrapper | Iteratively prune lowest-importance | Strong with linear/tree models | Costly; needs importance signal |
| Sequential Floating (SFFS/SFBS) | Wrapper | Add/remove with backtracking | Escapes some local optima | Even more expensive |
| Boruta | Wrapper (all-relevant) | Compare to shadow features via RF | Finds all relevant vars; statistical test | Slow on wide data |
| Lasso / Elastic Net | Embedded | L1/L1+L2 sparsity penalty | Selects while fitting; scalable | Linear; unstable under collinearity |
| Tree / GBM importance | Embedded | Split gain or permutation importance | Nonlinear; cheap | Biased toward high-cardinality features |
| SHAP-based (e.g. BorutaShap, shap-select) | Embedded/hybrid | Rank/test via Shapley contributions | Consistent, theoretically grounded | Compute-heavy on large data |

## Tools & libraries

| Tool | What it offers | URL |
|---|---|---|
| scikit-learn `feature_selection` | VarianceThreshold, SelectKBest, mutual_info, RFE/RFECV, SelectFromModel | https://scikit-learn.org/stable/modules/feature_selection.html |
| statsmodels | Stepwise/regression diagnostics, ANOVA, p-values | https://www.statsmodels.org/ |
| Boruta (R) | Original all-relevant wrapper around Random Forest | https://cran.r-project.org/package=Boruta |
| BorutaPy | Python port of Boruta | https://github.com/scikit-learn-contrib/boruta_py |
| BorutaShap | Boruta + SHAP importance | https://github.com/Ekeany/Boruta-Shap |
| SHAP | Shapley-value feature attribution & selection | https://github.com/shap/shap |
| shap-select | Lightweight SHAP + regression feature selection | https://github.com/transferwise/shap-select |
| mrmr_selection | Fast mRMR implementation | https://github.com/smazzanti/mrmr |
| praznik (R) | Information-theoretic selectors (mRMR, JMI, CMIM) | https://cran.r-project.org/package=praznik |
| XGBoost / LightGBM | Built-in gain & SHAP feature importance | https://xgboost.readthedocs.io/ |
| MLxtend | Sequential Feature Selector (SFS/SFFS) | https://rasbt.github.io/mlxtend/ |
| featurewiz | Automated mRMR + SULOV pipeline | https://github.com/AutoViML/featurewiz |

## Learning resources

- **The Elements of Statistical Learning (ESL)** — Hastie, Tibshirani, Friedman. Ch. 3 (subset selection, Lasso), Ch. 18 (high-dimensional). Free PDF: https://hastie.su.domains/ElemStatLearn/
- **An Introduction to Statistical Learning (ISLR/ISLP)** — James et al. Ch. 6 (subset selection, shrinkage). Free PDF: https://www.statlearning.com/
- **Mathematics for Machine Learning** — Deisenroth, Faisal, Ong. Linear algebra & optimization underpinning sparse methods. Free: https://mml-book.github.io/
- **Feature Engineering and Selection** — Kuhn & Johnson. Full book online: https://bookdown.org/max/FES/
- **scikit-learn user guide: Feature selection** — practical, code-first: https://scikit-learn.org/stable/modules/feature_selection.html
- **StatQuest (Josh Starmer)** — intuitive videos on Lasso/Ridge & regularization: https://www.youtube.com/c/joshstarmer
- **A Review of Feature Selection Methods Based on Mutual Information** (survey): https://arxiv.org/abs/1509.07577

## Key papers

- Guyon & Elisseeff (2003), *An Introduction to Variable and Feature Selection*, JMLR 3:1157–1182 — https://jmlr.org/papers/v3/guyon03a.html
- Tibshirani (1996), *Regression Shrinkage and Selection via the Lasso*, JRSS B 58(1):267–288 — https://doi.org/10.1111/j.2517-6161.1996.tb02080.x
- Peng, Long & Ding (2005), *Feature Selection Based on Mutual Information: mRMR*, IEEE TPAMI 27(8):1226–1238 — https://doi.org/10.1109/TPAMI.2005.159
- Kursa & Rudnicki (2010), *Feature Selection with the Boruta Package*, J. Stat. Software 36(11) — https://doi.org/10.18637/jss.v036.i11
- Kira & Rendell (1992), *A Practical Approach to Feature Selection (Relief)*, ML Proceedings 1992 — https://doi.org/10.1016/B978-1-55860-247-2.50037-1
- Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions (SHAP)*, NeurIPS — https://arxiv.org/abs/1705.07874
- Zou & Hastie (2005), *Regularization and Variable Selection via the Elastic Net*, JRSS B 67(2):301–320 — https://doi.org/10.1111/j.1467-9868.2005.00503.x

## Cross-references in AIForge

- [Feature Engineering Techniques](../Feature_Engineering_Techniques/) — constructing the features you then select from
- [Machine Learning](../../Machine_Learning/) — models that consume selected features
- [Model Evaluation](../../Model_Evaluation/) — cross-validation to avoid selection leakage
- [Explainable AI](../../Explainable_AI/) — SHAP and importance methods shared with selection
- [Information Theory](../../Information_Theory/) — mutual information foundations

## Sources

- scikit-learn — Feature selection: https://scikit-learn.org/stable/modules/feature_selection.html
- Guyon & Elisseeff, JMLR (2003): https://jmlr.org/papers/v3/guyon03a.html
- Tibshirani, Lasso (1996): https://academic.oup.com/jrsssb/article/58/1/267/7027929
- Peng, Long & Ding, mRMR (2005): https://www.scirp.org/reference/referencespapers?referenceid=2057988
- Kursa & Rudnicki, Boruta, J. Stat. Software (2010): https://www.jstatsoft.org/v36/i11/
- Lundberg & Lee, SHAP (2017): https://arxiv.org/abs/1705.07874
- A Review of Feature Selection Methods Based on Mutual Information: https://arxiv.org/abs/1509.07577
- BorutaShap repository: https://github.com/Ekeany/Boruta-Shap
- shap-select: https://arxiv.org/html/2410.06815v1
- ESL: https://hastie.su.domains/ElemStatLearn/ · ISLR: https://www.statlearning.com/ · FES book: https://bookdown.org/max/FES/
