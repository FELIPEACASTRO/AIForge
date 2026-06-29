# Ensemble Methods (Bagging, Boosting, Stacking)

> Combining many base learners into a single stronger predictor that lowers variance, bias, or both versus any individual model.

## Why it matters

Ensembles are the empirically dominant approach for structured/tabular learning: gradient-boosted trees (XGBoost, LightGBM, CatBoost) win the majority of Kaggle tabular competitions and power large-scale ranking, fraud, and risk systems in production. The core insight — that aggregating diverse, imperfect models cancels their uncorrelated errors — applies far beyond trees, underpinning random forests, model stacking, and even deep-learning ensembles. Understanding the bias–variance trade-off behind bagging vs. boosting is fundamental to choosing and tuning these methods.

## Core concepts

- **Bias–variance decomposition.** Expected error decomposes as `E[(y - f̂)²] = Bias² + Variance + σ²`. Ensembles attack different terms: bagging mainly reduces **variance**, boosting mainly reduces **bias** (and some variance).
- **Bagging (Bootstrap AGGregatING).** Train `M` base learners on bootstrap resamples `D_b` of the data and average: `f̂(x) = (1/M) Σ f̂_b(x)` (regression) or majority vote (classification). Variance of the average of `M` models with pairwise correlation `ρ` and variance `σ²` is `ρσ² + (1-ρ)σ²/M`, so decorrelating base learners is key — this is exactly what **random forests** add via random feature subsampling at each split. Out-of-bag (OOB) samples give a free validation estimate.
- **Boosting.** Fit base learners **sequentially**, each correcting the residual errors of the current ensemble: `F_m(x) = F_{m-1}(x) + ν·h_m(x)`, with learning rate (shrinkage) `ν`. **AdaBoost** reweights misclassified examples and combines weak learners with weights `α_t = ½ ln((1-ε_t)/ε_t)`. **Gradient boosting** generalizes this: `h_m` is fit to the negative gradient (pseudo-residuals) of a differentiable loss `L`, i.e. functional gradient descent in model space. Regularization via shrinkage, subsampling (stochastic GB), tree depth, and L1/L2 leaf penalties (XGBoost) controls overfitting.
- **Stacking (stacked generalization).** Train diverse base (level-0) models, then train a **meta-learner** (level-1) on their out-of-fold predictions to learn the optimal combination. Out-of-fold prediction is essential to avoid target leakage.
- **Blending.** A simpler variant of stacking that uses a single held-out validation set (not cross-validated folds) to train the meta-learner — faster but uses less data and can overfit the holdout.
- **Voting.** The simplest combiner: **hard voting** (majority class) or **soft voting** (average predicted probabilities), optionally weighted. No meta-learner is trained.
- **Why diversity matters.** Gains require base learners whose errors are uncorrelated; identical models give no benefit. Diversity is induced by data resampling, feature subsampling, different algorithms, or different hyperparameters.

## Algorithms / Methods

| Method | Type | How models combine | Reduces mainly | Notes |
|---|---|---|---|---|
| Bagging | Parallel | Average / majority vote | Variance | Generic wrapper over any high-variance learner |
| Random Forest | Parallel (bagging) | Vote / average | Variance | Bagging + random feature subsets per split; OOB error |
| Extra-Trees | Parallel | Vote / average | Variance | Random split thresholds → more bias, less variance |
| AdaBoost | Sequential (boosting) | Weighted vote | Bias | Reweights misclassified samples each round |
| Gradient Boosting (GBM) | Sequential (boosting) | Additive, shrinkage | Bias (+variance) | Fits pseudo-residuals of any differentiable loss |
| XGBoost | Sequential (boosting) | Additive, regularized | Bias (+variance) | 2nd-order loss, L1/L2, sparsity-aware, scalable |
| LightGBM | Sequential (boosting) | Additive | Bias (+variance) | Histogram + leaf-wise growth, GOSS, EFB; very fast |
| CatBoost | Sequential (boosting) | Additive | Bias (+variance) | Ordered boosting + native categorical handling |
| Voting | Combiner | Hard/soft (weighted) vote | Variance | No meta-model; combines heterogeneous models |
| Stacking | Combiner | Meta-learner on OOF preds | Bias + variance | Learns optimal blend; needs cross-validation |
| Blending | Combiner | Meta-learner on holdout | Bias + variance | Simpler/faster stacking; risks holdout overfit |

## Tools & libraries

| Tool | Focus | URL |
|---|---|---|
| scikit-learn | Bagging, RF, AdaBoost, GradientBoosting, Hist GBM, Voting, Stacking | https://scikit-learn.org/stable/modules/ensemble.html |
| XGBoost | Scalable regularized gradient boosting | https://xgboost.readthedocs.io/ |
| LightGBM | Fast histogram-based gradient boosting | https://lightgbm.readthedocs.io/ |
| CatBoost | Boosting with categorical-feature support | https://catboost.ai/ |
| H2O AutoML | Automated stacked ensembles | https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html |
| mlxtend | StackingClassifier / EnsembleVoteClassifier | https://rasbt.github.io/mlxtend/ |
| Optuna | Hyperparameter tuning for boosting models | https://optuna.org/ |
| imbalanced-learn | Balanced bagging/boosting for imbalanced data | https://imbalanced-learn.org/ |

## Learning resources

- **The Elements of Statistical Learning (ESL)** — Hastie, Tibshirani, Friedman. Chs. 8, 10, 15, 16 (bagging, boosting, random forests, ensembles). Free PDF: https://hastie.su.domains/ElemStatLearn/
- **An Introduction to Statistical Learning (ISLR/ISLP)** — James, Witten, Hastie, Tibshirani. Ch. 8 (tree-based methods, bagging, RF, boosting). Free: https://www.statlearning.com/
- **Probabilistic Machine Learning** — Kevin Murphy. Free PDFs (ensemble/tree chapters): https://probml.github.io/pml-book/
- **scikit-learn User Guide — Ensembles** — practical, code-first reference: https://scikit-learn.org/stable/modules/ensemble.html
- **StatQuest (Josh Starmer)** — clear video intros to AdaBoost, Gradient Boost, Random Forests, XGBoost: https://www.youtube.com/c/joshstarmer
- **Kaggle — Intermediate ML (XGBoost lesson)**: https://www.kaggle.com/learn/intermediate-machine-learning
- **Foundations of Machine Learning** — Mohri, Rostamizadeh, Talwalkar (boosting theory chapter): https://cs.nyu.edu/~mohri/mlbook/

## Key papers

- Breiman, L. (1996). *Bagging Predictors.* Machine Learning 24(2):123–140. https://doi.org/10.1007/BF00058655
- Freund, Y., Schapire, R. (1997). *A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting (AdaBoost).* J. Computer and System Sciences 55(1):119–139. https://doi.org/10.1006/jcss.1997.1504
- Wolpert, D. H. (1992). *Stacked Generalization.* Neural Networks 5(2):241–259. https://doi.org/10.1016/S0893-6080(05)80023-1
- Breiman, L. (2001). *Random Forests.* Machine Learning 45(1):5–32. https://doi.org/10.1023/A:1010933404324
- Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine.* Annals of Statistics 29(5):1189–1232. https://doi.org/10.1214/aos/1013203451
- Chen, T., Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD '16. https://arxiv.org/abs/1603.02754
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017. https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
- Prokhorenkova, L. et al. (2018). *CatBoost: Unbiased Boosting with Categorical Features.* NeurIPS 2018. https://arxiv.org/abs/1706.09516

## Cross-references in AIForge

- [../](../) — Classical ML Algorithms (parent: decision trees, base learners)
- [../../Machine_Learning/](../../Machine_Learning/) — supervised learning foundations
- [../../Optimization_Algorithms/](../../Optimization_Algorithms/) — gradient descent behind gradient boosting
- [../../Model_Evaluation/](../../Model_Evaluation/) — cross-validation, OOB, and stacking validation
- [../../Bayesian_and_Probabilistic_ML/](../../Bayesian_and_Probabilistic_ML/) — Bayesian model averaging as a probabilistic ensemble

## Sources

- Breiman, *Bagging Predictors*: https://doi.org/10.1007/BF00058655
- Freund & Schapire, *AdaBoost*: https://doi.org/10.1006/jcss.1997.1504
- Wolpert, *Stacked Generalization*: https://doi.org/10.1016/S0893-6080(05)80023-1
- Friedman, *Gradient Boosting Machine*: https://doi.org/10.1214/aos/1013203451
- Chen & Guestrin, *XGBoost*: https://arxiv.org/abs/1603.02754
- Ke et al., *LightGBM*: https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
- Prokhorenkova et al., *CatBoost*: https://arxiv.org/abs/1706.09516
- scikit-learn Ensembles User Guide: https://scikit-learn.org/stable/modules/ensemble.html
- ESL (Hastie et al.): https://hastie.su.domains/ElemStatLearn/
- ISLR (statlearning.com): https://www.statlearning.com/
