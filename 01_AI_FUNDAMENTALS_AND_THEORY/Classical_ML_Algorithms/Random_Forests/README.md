# Random Forests

> An ensemble of de-correlated decision trees, each trained on a bootstrap sample with feature subsampling at every split, whose predictions are aggregated (majority vote for classification, average for regression).

## Why it matters

Random Forests are one of the most reliable off-the-shelf supervised learners: they handle non-linear interactions, mixed feature types, and missing-value-robust splits with minimal tuning, and rarely overfit as the number of trees grows. They remain a default strong baseline for tabular data and ship with cheap built-in tools for generalization estimation (out-of-bag error) and feature ranking (importances).

## Core concepts

- **Bagging (Bootstrap Aggregating).** Train each tree on a bootstrap sample (n points drawn with replacement) of the training set. Averaging many high-variance, low-bias trees reduces variance without increasing bias. For B i.i.d. estimators each with variance `σ²`, the average has variance `σ²/B`; for trees correlated by `ρ`, the variance floor is `ρσ² + (1−ρ)σ²/B`, so the goal is to *decorrelate* the trees.
- **Random feature subspace.** At each split, only a random subset of `m` features (often `m = √p` for classification, `m = p/3` for regression, with `p` total features) is considered. This is the key idea beyond plain bagging — it breaks the correlation `ρ` between trees, lowering ensemble variance.
- **Split criteria.** Trees grow by greedily minimizing node impurity: **Gini** `1 − Σ_k p_k²` or **entropy** `−Σ_k p_k log p_k` for classification, and squared-error / MSE reduction for regression. Trees are typically grown deep (low bias) and left unpruned.
- **Aggregation.** Classification uses majority (or soft probability) vote across trees; regression averages tree outputs. Probabilities come from the mean of per-tree class frequencies.
- **Out-of-Bag (OOB) estimation.** Each bootstrap leaves out ~`1/e ≈ 36.8%` of samples per tree. Predicting each point using only the trees that did *not* train on it yields the OOB error — a nearly unbiased generalization estimate without a separate validation set.
- **Feature importance.** Two main flavors: **Mean Decrease in Impurity (MDI / Gini importance)**, the total impurity reduction attributed to a feature across all trees (fast but biased toward high-cardinality / continuous features); and **permutation importance**, the drop in accuracy when a feature's values are shuffled (model-agnostic, more reliable, recommended on held-out or OOB data).
- **Proximities.** The fraction of trees in which two samples land in the same leaf defines a similarity matrix usable for clustering, outlier detection, and missing-value imputation.

## Variants

| Variant | Key idea | When to use |
|---|---|---|
| **Bagged trees** | Bootstrap only, all features considered at each split | Baseline; when `p` is small |
| **Random Forest** (Breiman 2001) | Bagging + random feature subspace at each split | General-purpose default |
| **Extremely Randomized Trees (ExtraTrees)** | Random cut-point per feature; often whole sample (no bootstrap) | Lower variance, faster training |
| **Random Subspace / Random Decision Forests** (Ho 1995/1998) | Subsample features per tree (not per split) | High-dimensional, redundant features |
| **Rotation Forest** | PCA on random feature subsets before tree induction | Boost diversity on correlated features |
| **Quantile Regression Forests** | Keep full leaf distributions to estimate conditional quantiles | Prediction intervals / uncertainty |
| **Generalized Random Forests (GRF)** | Forests as adaptive local-weighting for general estimating equations | Causal / heterogeneous treatment effects |
| **Mondrian Forests** | Online, tree built from a Mondrian process | Streaming / incremental learning |

## Tools & libraries

| Tool | Notes | URL |
|---|---|---|
| scikit-learn | `RandomForestClassifier`, `RandomForestRegressor`, `ExtraTrees*`, `permutation_importance` | https://scikit-learn.org/stable/modules/ensemble.html#forest |
| ranger | Fast C++ RF for R/Python, large datasets, survival forests | https://github.com/imbs-hl/ranger |
| randomForest (R) | Breiman/Cutler's original Fortran-backed R package | https://cran.r-project.org/package=randomForest |
| H2O | Distributed RF/DRF at scale, AutoML | https://docs.h2o.ai/ |
| cuML (RAPIDS) | GPU-accelerated Random Forest | https://docs.rapids.ai/api/cuml/stable/ |
| EconML / grf | Generalized & causal random forests | https://github.com/grf-labs/grf |
| Optuna | Hyperparameter search (`n_estimators`, `max_depth`, `max_features`) | https://optuna.org/ |
| SHAP | Tree-aware feature attribution for forests | https://github.com/shap/shap |
| Spark MLlib | Distributed RandomForest on Spark | https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forests |

## Learning resources

- **The Elements of Statistical Learning (ESL)** — Hastie, Tibshirani, Friedman, Ch. 15 "Random Forests" (free PDF): https://hastie.su.domains/ElemStatLearn/
- **An Introduction to Statistical Learning (ISLR/ISLP)** — James et al., Ch. 8 "Tree-Based Methods" (free PDF + Python labs): https://www.statlearning.com/
- **scikit-learn User Guide — Forests of randomized trees**: https://scikit-learn.org/stable/modules/ensemble.html#forest
- **scikit-learn — Permutation vs MDI importance pitfalls**: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html
- **scikit-learn — OOB errors example**: https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html
- **Leo Breiman & Adele Cutler — original Random Forests homepage** (intuition, proximities, importances): https://www.stat.berkeley.edu/~breiman/RandomForests/
- **StatQuest — Random Forests (video series)**: https://www.youtube.com/watch?v=J4Wdy0Wc_xQ
- **Understanding Random Forests: From Theory to Practice** — Gilles Louppe's PhD thesis (deep, rigorous): https://arxiv.org/abs/1407.7502

## Key papers

- Breiman, L. (2001). *Random Forests.* Machine Learning 45:5–32. https://doi.org/10.1023/A:1010933404324
- Ho, T.K. (1998). *The Random Subspace Method for Constructing Decision Forests.* IEEE TPAMI 20(8):832–844. https://doi.org/10.1109/34.709601
- Geurts, P., Ernst, D., Wehenkel, L. (2006). *Extremely Randomized Trees.* Machine Learning 63:3–42. https://doi.org/10.1007/s10994-006-6226-1
- Breiman, L. (1996). *Bagging Predictors.* Machine Learning 24:123–140. https://doi.org/10.1007/BF00058655
- Meinshausen, N. (2006). *Quantile Regression Forests.* JMLR 7:983–999. https://www.jmlr.org/papers/v7/meinshausen06a.html
- Athey, S., Tibshirani, J., Wager, S. (2019). *Generalized Random Forests.* Annals of Statistics 47(2):1148–1178. https://doi.org/10.1214/18-AOS1709 (arXiv: https://arxiv.org/abs/1610.01271)
- Scornet, E., Biau, G., Vert, J.-P. (2015). *Consistency of Random Forests.* Annals of Statistics 43(4):1716–1741. https://arxiv.org/abs/1405.2881
- Strobl, C. et al. (2007). *Bias in Random Forest Variable Importance Measures.* BMC Bioinformatics 8:25. https://doi.org/10.1186/1471-2105-8-25

## Cross-references in AIForge

- [Decision Trees](../Decision_Trees/) — the base learner aggregated by forests
- [Ensemble Methods](../Ensemble_Methods/) — bagging, boosting, stacking overview
- [Gradient Boosting](../Gradient_Boosting/) — the sequential tree-ensemble alternative
- [Model Evaluation](../../Model_Evaluation/) — OOB, cross-validation, generalization estimation
- [Feature Engineering](../../Feature_Engineering/) — importances and selection workflows

## Sources

- Springer / Machine Learning — Breiman (2001): https://link.springer.com/article/10.1023/A:1010933404324
- Springer / Machine Learning — Geurts et al. (2006): https://link.springer.com/article/10.1007/s10994-006-6226-1
- IEEE TPAMI — Ho (1998): https://dl.acm.org/doi/10.1109/34.709601
- scikit-learn RandomForestClassifier docs: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- scikit-learn ensemble user guide: https://scikit-learn.org/stable/modules/ensemble.html#forest
- Breiman & Cutler RF homepage: https://www.stat.berkeley.edu/~breiman/RandomForests/
- ESL: https://hastie.su.domains/ElemStatLearn/ — ISLR: https://www.statlearning.com/
- Louppe thesis: https://arxiv.org/abs/1407.7502
