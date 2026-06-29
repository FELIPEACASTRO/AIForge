# Gradient Boosting (XGBoost, LightGBM, CatBoost)

> Gradient boosting builds an additive ensemble of weak learners (usually shallow decision trees) sequentially, where each new tree fits the negative gradient of a differentiable loss — turning boosting into gradient descent in function space.

## Why it matters

Gradient-boosted decision trees (GBDT) remain the dominant approach for **tabular / structured data**, routinely outperforming deep nets on heterogeneous columns and small-to-medium datasets, and they win a large share of Kaggle competitions. The three modern libraries — XGBoost, LightGBM, CatBoost — give production-grade speed, regularization, missing-value handling, and categorical support, making GBDT the default first model for most real-world tabular problems.

## Core concepts

- **Additive model.** The prediction is a sum of $M$ weak learners: $F_M(x) = \sum_{m=1}^{M} \nu \, h_m(x)$, where $\nu \in (0,1]$ is the **learning rate** (shrinkage) and $h_m$ is typically a regression tree.
- **Forward stagewise / functional gradient descent.** At step $m$, fit $h_m$ to the **negative gradient** (pseudo-residuals) of the loss w.r.t. the current prediction: $r_{im} = -\left[\partial L(y_i, F(x_i)) / \partial F(x_i)\right]_{F=F_{m-1}}$. For squared error this reduces to ordinary residuals $y_i - F_{m-1}(x_i)$ (Friedman, 2001).
- **Second-order (Newton) boosting.** XGBoost expands the loss to second order using gradients $g_i$ and Hessians $h_i$. The optimal leaf weight is $w_j^* = -\frac{\sum_{i\in I_j} g_i}{\sum_{i\in I_j} h_i + \lambda}$, and the split-gain criterion is $\frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma$ (Chen & Guestrin, 2016).
- **Regularization.** Shrinkage ($\nu$), tree depth / number of leaves, L1/L2 penalties on leaf weights ($\alpha$, $\lambda$), minimum child weight, and **row/column subsampling** (stochastic gradient boosting, Friedman 2002) all control overfitting.
- **Loss flexibility.** Any twice-differentiable loss works: squared/absolute error and Huber for regression; logistic / softmax for classification; plus ranking, Poisson, quantile, Tweedie, and survival objectives.
- **Split finding.** Exact greedy is $O(\text{data} \times \text{features})$; **histogram-based** methods bin features into discrete buckets for large speedups (used by all three libraries' fast modes).

## Variants / Key techniques

| Technique | Library origin | What it does |
|---|---|---|
| Newton (2nd-order) boosting | XGBoost | Uses Hessians for split gain & leaf weights; more accurate steps |
| Sparsity-aware split finding | XGBoost | Learns a default direction for missing/sparse values per split |
| Weighted quantile sketch | XGBoost | Approximate split proposals on weighted data at scale |
| Histogram binning | LightGBM (popularized) | Bins continuous features → fast, low-memory split search |
| GOSS (Gradient-based One-Side Sampling) | LightGBM | Keeps large-gradient rows, subsamples small-gradient rows |
| EFB (Exclusive Feature Bundling) | LightGBM | Bundles sparse mutually-exclusive features to cut dimensionality |
| Leaf-wise (best-first) growth | LightGBM | Splits the highest-gain leaf, not level-wise → deeper, more accurate |
| Ordered boosting | CatBoost | Permutation scheme that removes target-leakage / prediction shift |
| Ordered target statistics | CatBoost | Leakage-free encoding of categorical features |
| Oblivious (symmetric) trees | CatBoost | Same split per level → fast, regularized, cache-friendly inference |
| Stochastic gradient boosting | Friedman (2002) | Subsamples rows each iteration for variance reduction |
| DART | XGBoost / LightGBM | Drops trees during boosting (dropout) to fight over-specialization |

## Tools & libraries

| Tool | What it is | URL |
|---|---|---|
| XGBoost | Scalable, regularized GBDT; multi-language, GPU, distributed | https://xgboost.readthedocs.io/ |
| LightGBM | Fast histogram + leaf-wise GBDT (GOSS/EFB) by Microsoft | https://lightgbm.readthedocs.io/ |
| CatBoost | Ordered boosting + native categorical support by Yandex | https://catboost.ai/ |
| scikit-learn `HistGradientBoosting` | Built-in histogram GBDT (LightGBM-inspired) | https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting |
| scikit-learn `GradientBoosting` | Classic exact GBDT implementation | https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html |
| Optuna | Hyperparameter optimization (TPE, pruning) | https://optuna.org/ |
| Hyperopt | Bayesian / TPE hyperparameter search | http://hyperopt.github.io/hyperopt/ |
| SHAP | Tree-aware feature attribution for GBDT | https://shap.readthedocs.io/ |

## Learning resources

- **The Elements of Statistical Learning (ESL)** — Hastie, Tibshirani, Friedman. Ch. 10 (Boosting) is the canonical treatment. Free PDF: https://hastie.su.domains/ElemStatLearn/
- **An Introduction to Statistical Learning (ISLR/ISLP)** — gentler intro to trees & boosting. Free PDFs: https://www.statlearning.com/
- **StatQuest: Gradient Boost (playlist)** — Josh Starmer's step-by-step visual series. https://www.youtube.com/playlist?list=PLblh5JKOoLUJjeXUvUE0maghNuY2_5fY6
- **XGBoost docs — Introduction to Boosted Trees** (math derivation): https://xgboost.readthedocs.io/en/stable/tutorials/model.html
- **LightGBM — Features** (GOSS/EFB/leaf-wise explained): https://lightgbm.readthedocs.io/en/latest/Features.html
- **CatBoost — How it works**: https://catboost.ai/docs/en/concepts/algorithm-main-stages
- **scikit-learn user guide — Ensembles / Gradient Boosting**: https://scikit-learn.org/stable/modules/ensemble.html
- **Kaggle: tuning XGBoost / GBDT** notebooks & courses: https://www.kaggle.com/learn

## Key papers

- Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine.* Annals of Statistics 29(5):1189–1232. DOI: https://doi.org/10.1214/aos/1013203451
- Friedman, J. H. (2002). *Stochastic Gradient Boosting.* Computational Statistics & Data Analysis 38(4):367–378. DOI: https://doi.org/10.1016/S0167-9473(01)00065-2
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD 2016. arXiv: https://arxiv.org/abs/1603.02754
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017. https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
- Prokhorenkova, L. et al. (2018). *CatBoost: Unbiased Boosting with Categorical Features.* NeurIPS 2018. arXiv: https://arxiv.org/abs/1706.09516
- Dorogush, A. V., Ershov, V., Gulin, A. (2018). *CatBoost: Gradient Boosting with Categorical Features Support.* arXiv: https://arxiv.org/abs/1810.11363
- Lundberg, S. M. et al. (2020). *From Local Explanations to Global Understanding with Explainable AI for Trees.* Nature Machine Intelligence. arXiv: https://arxiv.org/abs/1905.04610

## Cross-references in AIForge

- [Ensemble Methods](../Ensemble_Methods/) — bagging, stacking, and boosting overview
- [Decision Trees](../Decision_Trees/) — the base learners GBDT relies on
- [Random Forests](../Random_Forests/) — the bagging counterpart to boosting
- [Optimization Algorithms](../../Optimization_Algorithms/) — gradient descent, the engine of functional boosting
- [Model Evaluation](../../Model_Evaluation/) — cross-validation & metrics for tuning GBDT

## Sources

- arXiv 1603.02754 — XGBoost: https://arxiv.org/abs/1603.02754
- NeurIPS 2017 — LightGBM: https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
- arXiv 1706.09516 — CatBoost: https://arxiv.org/abs/1706.09516
- arXiv 1810.11363 — CatBoost (categorical support): https://arxiv.org/abs/1810.11363
- Annals of Statistics — Friedman 2001 (DOI 10.1214/aos/1013203451): https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5
- Computational Statistics & Data Analysis — Friedman 2002: https://doi.org/10.1016/S0167-9473(01)00065-2
- XGBoost docs: https://xgboost.readthedocs.io/ · LightGBM docs: https://lightgbm.readthedocs.io/ · CatBoost docs: https://catboost.ai/
- scikit-learn ensembles: https://scikit-learn.org/stable/modules/ensemble.html
- ESL: https://hastie.su.domains/ElemStatLearn/ · ISLR: https://www.statlearning.com/
