# Decision Trees

> A non-parametric supervised learning model that recursively partitions the feature space into axis-aligned regions, predicting via a tree of if-then rules learned greedily from data.

## Why it matters

Decision trees are among the most interpretable predictive models: a fitted tree reads as a flowchart of human-readable rules, requires little data preprocessing (no scaling, native handling of mixed types), and captures non-linear interactions automatically. They are also the foundational building block of the dominant tabular-ML ensembles — Random Forests and gradient-boosted trees (XGBoost, LightGBM, CatBoost) — making a solid grasp of trees a prerequisite for understanding modern structured-data modeling.

## Core concepts

A tree is grown **top-down** by recursive binary (or multiway) partitioning. At each node the learner searches over candidate splits (feature + threshold) and picks the one that most reduces an **impurity** (classification) or **error** (regression) measure of the resulting children.

**Impurity measures (classification).** For a node with class proportions $p_k$:

- **Gini impurity:** $G = \sum_{k} p_k(1 - p_k) = 1 - \sum_k p_k^2$ (default in CART / scikit-learn).
- **Entropy:** $H = -\sum_k p_k \log_2 p_k$.
- **Information gain:** $IG = H(\text{parent}) - \sum_{c} \frac{N_c}{N} H(c)$ — the criterion ID3 maximizes.
- **Gain ratio:** $IG / \text{SplitInfo}$, normalizing gain by the intrinsic entropy of the split to counter ID3's bias toward high-cardinality attributes; introduced in C4.5.

**Splitting (regression).** Choose the split minimizing weighted child variance / sum of squared error: $\sum_{c} \sum_{i \in c} (y_i - \bar{y}_c)^2$. Leaf prediction is the mean (or median) of its samples.

**Stopping & complexity.** Growth halts on a pure node, minimum samples per leaf/split, maximum depth, or a minimum impurity decrease. An unconstrained tree overfits (zero training error), so depth/leaf constraints (**pre-pruning**) or a separate pruning pass are essential.

**Pruning.** **Cost-complexity (weakest-link) pruning** — CART's method — minimizes $R_\alpha(T) = R(T) + \alpha |\tilde{T}|$, trading training error $R(T)$ against the number of leaves $|\tilde{T}|$; the regularization strength $\alpha$ is chosen by cross-validation (exposed as `ccp_alpha` in scikit-learn). C4.5 instead uses **pessimistic error pruning** based on confidence-interval upper bounds on the per-node error.

**Properties & limits.** Finding the globally optimal tree is **NP-complete** (Hyafil & Rivest, 1976), so all practical learners are greedy. Single trees are high-variance and unstable (small data changes can reshape the tree), are biased toward axis-aligned boundaries, and cannot extrapolate beyond the training range in regression — motivations for ensembling.

## Algorithms / Variants

| Algorithm | Splits | Criterion | Targets | Pruning | Notes |
|-----------|--------|-----------|---------|---------|-------|
| **ID3** (Quinlan, 1986) | Multiway, categorical only | Information gain | Classification | None (prone to overfit) | Cannot handle continuous features or missing values natively |
| **C4.5** (Quinlan, 1993) | Multiway | Gain ratio | Classification | Pessimistic error pruning | Continuous attributes, missing values, rule post-pruning; C5.0 is the faster commercial successor |
| **CART** (Breiman et al., 1984) | Strictly binary | Gini / variance (MSE) | Classification + regression | Cost-complexity (CV) | Surrogate splits for missing data; basis of most modern libraries |
| **CHAID** (Kass, 1980) | Multiway | Chi-square test | Classification (+ regression) | Statistical stopping | Significance-based merging of categories |
| **Conditional Inference Trees** | Binary | Permutation tests | Both | Statistical stopping | Unbiased split selection, no selection bias toward many-level features |
| **Oblique / optimal trees** | Linear-combination or globally optimal | Various (MILP/heuristic) | Both | Built-in | Non-axis-aligned (OC1) or certifiably optimal (OCT, GOSDT) |

## Tools & libraries

| Tool | Language | What it provides | URL |
|------|----------|------------------|-----|
| scikit-learn (`sklearn.tree`) | Python | CART-style `DecisionTreeClassifier`/`Regressor`, `ccp_alpha` pruning, `plot_tree`, `export_text` | https://scikit-learn.org/stable/modules/tree.html |
| rpart | R | Recursive partitioning (CART) with cost-complexity pruning | https://cran.r-project.org/package=rpart |
| party / partykit (`ctree`) | R | Conditional inference trees, unbiased splits | https://cran.r-project.org/package=partykit |
| XGBoost | Python/R/JVM | Gradient-boosted decision trees (trees as base learners) | https://xgboost.readthedocs.io |
| LightGBM | Python/R/C++ | Leaf-wise, histogram-based boosted trees | https://lightgbm.readthedocs.io |
| CatBoost | Python/R | Oblivious (symmetric) boosted trees, native categoricals | https://catboost.ai |
| dtreeviz | Python | High-quality tree visualization and decision-path plots | https://github.com/parrt/dtreeviz |
| Optuna | Python | Hyperparameter search (depth, min-samples, `ccp_alpha`) | https://optuna.org |

## Learning resources

| Resource | Type | Link |
|----------|------|------|
| *An Introduction to Statistical Learning* (ISLR) — Ch. 8, Tree-Based Methods | Free book | https://www.statlearning.com |
| *The Elements of Statistical Learning* (ESL) — §9.2 CART | Free book | https://hastie.su.domains/ElemStatLearn/ |
| scikit-learn Decision Trees user guide | Official docs | https://scikit-learn.org/stable/modules/tree.html |
| StatQuest — Decision Trees / Regression Trees (Josh Starmer) | Video series | https://www.youtube.com/watch?v=_L39rN6gz7Y |
| C4.5 tutorial / FAQ (Sebastian Raschka — impurity metrics & binary splits) | Tutorial | https://sebastianraschka.com/faq/docs/decision-tree-binary.html |
| Quinlan, *C4.5: Programs for Machine Learning* (Morgan Kaufmann, 1993) | Book | https://dl.acm.org/doi/book/10.5555/152181 |

## Key papers

| Paper | Authors / Year | Link |
|-------|----------------|------|
| Induction of Decision Trees (ID3) | Quinlan, 1986, *Machine Learning* 1(1) | https://doi.org/10.1007/BF00116251 |
| Classification and Regression Trees (CART) | Breiman, Friedman, Olshen, Stone, 1984 | https://doi.org/10.1201/9781315139470 |
| Constructing Optimal Binary Decision Trees is NP-Complete | Hyafil & Rivest, 1976, *Inf. Process. Lett.* 5(1) | https://doi.org/10.1016/0020-0190(76)90095-8 |
| An Exploratory Technique for Investigating Large Quantities of Categorical Data (CHAID) | Kass, 1980, *Applied Statistics* 29(2) | https://doi.org/10.2307/2986296 |
| Unbiased Recursive Partitioning: A Conditional Inference Framework | Hothorn, Hornik, Zeileis, 2006, *JCGS* 15(3) | https://doi.org/10.1198/106186006X133933 |
| Optimal Classification Trees | Bertsimas & Dunn, 2017, *Machine Learning* 106 | https://doi.org/10.1007/s10994-017-5633-9 |

## Cross-references in AIForge

- [Ensemble Methods](../Ensemble_Methods/) — bagging/boosting/stacking that aggregate trees
- [Random Forests](../Random_Forests/) — bagged decision-tree ensembles
- [Gradient Boosting](../Gradient_Boosting/) — sequential boosted trees (XGBoost/LightGBM/CatBoost)
- [Model Evaluation](../../Model_Evaluation/) — cross-validation and pruning-parameter selection

## Sources

- scikit-learn — 1.10. Decision Trees: https://scikit-learn.org/stable/modules/tree.html
- Quinlan, Induction of Decision Trees (Springer): https://doi.org/10.1007/BF00116251
- Breiman et al., Classification and Regression Trees (Routledge): https://www.routledge.com/Classification-and-Regression-Trees/Breiman-Friedman-Stone-Olshen/p/book/9780412048418
- Hyafil & Rivest, NP-completeness (ScienceDirect): https://www.sciencedirect.com/science/article/abs/pii/0020019076900958
- ISLR: https://www.statlearning.com
- ESL: https://hastie.su.domains/ElemStatLearn/
- Raschka — decision tree FAQ: https://sebastianraschka.com/faq/docs/decision-tree-binary.html
