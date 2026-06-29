# Hyperparameter Optimization

> Hyperparameter optimization (HPO) is the search for the configuration of a learning algorithm's non-learned settings (e.g. learning rate, tree depth, regularization strength) that minimizes a validation loss or maximizes a held-out metric.

## Why it matters

A model's parameters are fit by training, but its *hyperparameters* must be chosen before or around training and often dominate final performance — a poorly tuned strong model routinely loses to a well-tuned weak one. Because each evaluation requires training a model, HPO is an expensive, noisy, often non-differentiable black-box optimization problem, which is why principled search strategies (rather than manual "graduate-student descent") matter. Good HPO improves accuracy, reproducibility, and compute efficiency, and is the core engine of AutoML.

## Core concepts

- **Objective.** Find `x* = argmin_{x ∈ X} f(x)` where `x` is a hyperparameter configuration, `X` is the search space, and `f(x)` is a (stochastic, expensive) validation metric, e.g. k-fold cross-validation loss. `f` has no gradient and each call costs a full training run.
- **Search space `X`.** A mix of continuous (learning rate, often log-scaled), integer (depth, n_estimators), categorical (optimizer, kernel), and **conditional** dimensions (e.g. `kernel_width` only exists if `kernel = RBF`). Tree-structured / conditional spaces are common.
- **Black-box vs. multi-fidelity.** *Black-box* methods treat `f` as a single expensive evaluation. *Multi-fidelity* methods exploit cheap approximations `f̃` (fewer epochs, subsampled data, fewer estimators) and allocate more budget to promising configs — the basis of early stopping.
- **Exploration vs. exploitation.** Bayesian methods build a surrogate `p(f | data)` and pick the next point by maximizing an **acquisition function** — commonly **Expected Improvement** `EI(x) = E[max(0, f_best − f(x))]`, or Upper/Lower Confidence Bound (UCB/LCB), or Probability of Improvement.
- **Sequential Model-Based Optimization (SMBO).** The general loop: fit a surrogate to observed `(x, f(x))` pairs, optimize the acquisition function to propose the next `x`, evaluate, repeat. Surrogates include Gaussian Processes (GP), Tree-structured Parzen Estimators (TPE), and random forests (SMAC).
- **TPE.** Instead of modeling `p(f|x)`, TPE models `p(x|f)` via two densities — `l(x)` (configs with good scores) and `g(x)` (the rest) — and maximizes the ratio `l(x)/g(x)`, which is proportional to EI. Cost is `O(N)` vs. GP's `O(N³)`, and it handles conditional spaces naturally.
- **Successive Halving (SHA).** Allocate a small budget to many configs, keep the top `1/η`, multiply survivors' budget by `η`, repeat. **Hyperband** runs several SHA "brackets" with different starting budgets to hedge the exploration/exploitation tradeoff. **ASHA** removes synchronization barriers for massively parallel async tuning.
- **Validation hygiene.** Tune on a validation split (or nested CV); report on a held-out test set. Over-tuning on the validation metric causes optimization overfitting.

## Algorithms / Methods

| Method | Idea | Multi-fidelity | Parallelism | Best for |
|---|---|---|---|---|
| Grid search | Exhaustive Cartesian product of discrete values | No | Embarrassingly parallel | Few (≤3) low-cardinality dims |
| Random search | Sample configs i.i.d. from `X` | No | Embarrassingly parallel | Strong baseline; high-dim spaces with few effective dims |
| Bayesian opt (GP) | GP surrogate + acquisition (EI/UCB) | No (by default) | Limited (sequential-ish) | Smooth, low-dim, expensive `f` |
| TPE | Density-ratio `l(x)/g(x)` surrogate (SMBO) | Optional | Good | Conditional / mixed spaces; default in Optuna & Hyperopt |
| SMAC | Random-forest surrogate, handles categoricals | Optional | Good | Mixed/categorical & algorithm-config problems |
| Successive Halving | Budget reallocation, drop bottom `1−1/η` | Yes | Synchronous | Cheap-to-approximate objectives |
| Hyperband | Multiple SHA brackets over budgets | Yes | Synchronous | Robust early stopping with no surrogate |
| BOHB | Hyperband schedule + TPE-style model | Yes | Yes | Strong anytime + asymptotic performance |
| ASHA | Asynchronous Successive Halving | Yes | Massively parallel | Large clusters, many workers |
| PBT | Population-based; evolve weights + hyperparams jointly | Online | Massively parallel | Schedules (LR, RL), neural nets |
| CMA-ES | Evolutionary covariance-matrix adaptation | No | Population | Continuous, non-convex spaces |

## Tools & libraries

| Tool | What it does | URL |
|---|---|---|
| Optuna | Define-by-run HPO; TPE/CMA-ES, pruners (ASHA/Hyperband), dashboards | https://optuna.org/ |
| Hyperopt | Distributed TPE / random / anneal search | https://github.com/hyperopt/hyperopt |
| Ray Tune | Scalable distributed tuning; ASHA, PBT, BOHB, integrates Optuna/Hyperopt | https://docs.ray.io/en/latest/tune/index.html |
| scikit-learn | `GridSearchCV`, `RandomizedSearchCV`, `HalvingGridSearchCV` | https://scikit-learn.org/stable/modules/grid_search.html |
| scikit-optimize | GP/forest-based Bayesian opt (`BayesSearchCV`) | https://scikit-optimize.github.io/ |
| SMAC3 | Random-forest SMBO for algorithm configuration | https://github.com/automl/SMAC3 |
| HpBandSter | Reference BOHB + Hyperband implementation | https://github.com/automl/HpBandSter |
| Hyperband/ASHA via Tune | Trial schedulers in Ray Tune | https://docs.ray.io/en/latest/tune/api/schedulers.html |
| Keras Tuner | HPO for Keras/TensorFlow models | https://keras.io/keras_tuner/ |
| Ax / BoTorch | Adaptive experimentation; GP Bayesian opt on PyTorch | https://ax.dev/ |
| Weights & Biases Sweeps | Hosted sweep orchestration (grid/random/Bayes) | https://docs.wandb.ai/guides/sweeps |
| XGBoost / LightGBM | Gradient-boosting libraries with many hyperparameters to tune | https://xgboost.readthedocs.io/ |

## Learning resources

- **AutoML: Methods, Systems, Challenges** — Hutter, Kotthoff, Vanschoren (eds.), free book; Ch. 1 covers HPO in depth: https://www.automl.org/book/
- **The Elements of Statistical Learning (ESL)** — Hastie, Tibshirani, Friedman (free PDF); model assessment/selection, CV: https://hastie.su.domains/ElemStatLearn/
- **An Introduction to Statistical Learning (ISLR/ISLP)** — James et al. (free PDF); resampling & tuning intuition: https://www.statlearning.com/
- **scikit-learn user guide — Tuning the hyper-parameters of an estimator**: https://scikit-learn.org/stable/modules/grid_search.html
- **Optuna tutorials** — key features and pruning walkthroughs: https://optuna.readthedocs.io/en/stable/tutorial/index.html
- **Ray Tune tutorials & key concepts**: https://docs.ray.io/en/latest/tune/getting-started.html
- **Distill / blog: A Visual Exploration of Gaussian Processes** (background for GP-based BO): https://distill.pub/2019/visual-exploration-gaussian-processes/
- **AutoML.org Hyperband / BOHB blog**: https://www.automl.org/blog-bohb/

## Key papers

- Bergstra & Bengio (2012), *Random Search for Hyper-Parameter Optimization*, JMLR 13:281–305 — https://jmlr.org/papers/v13/bergstra12a.html
- Bergstra, Bardenet, Bengio, Kégl (2011), *Algorithms for Hyper-Parameter Optimization* (introduces TPE), NeurIPS — https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html
- Snoek, Larochelle, Adams (2012), *Practical Bayesian Optimization of Machine Learning Algorithms*, NeurIPS — https://arxiv.org/abs/1206.2944
- Li, Jamieson, DeSalvo, Rostamizadeh, Talwalkar (2017), *Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization*, JMLR 18 — https://arxiv.org/abs/1603.06560
- Falkner, Klein, Hutter (2018), *BOHB: Robust and Efficient Hyperparameter Optimization at Scale*, ICML — https://arxiv.org/abs/1807.01774
- Li et al. (2018/2020), *A System for Massively Parallel Hyperparameter Tuning* (ASHA), MLSys — https://arxiv.org/abs/1810.05934
- Akiba, Sano, Yanase, Ohta, Koyama (2019), *Optuna: A Next-generation Hyperparameter Optimization Framework*, KDD — https://arxiv.org/abs/1907.10902
- Liaw, Liang, Nishihara, Moritz, Gonzalez, Stoica (2018), *Tune: A Research Platform for Distributed Model Selection and Training* — https://arxiv.org/abs/1807.05118

## Cross-references in AIForge

- [Machine_Learning](../../Machine_Learning/) — core supervised/unsupervised learning that HPO tunes
- [Optimization_Algorithms](../../Optimization_Algorithms/) — gradient and black-box optimization foundations
- [Bayesian_and_Probabilistic_ML](../../Bayesian_and_Probabilistic_ML/) — Gaussian processes and acquisition functions behind Bayesian HPO
- [Model_Evaluation](../../Model_Evaluation/) — cross-validation and metrics that define the HPO objective
- [AutoML_Frameworks](../AutoML_Frameworks/) — end-to-end systems that wrap HPO

## Sources

- Optuna paper — https://arxiv.org/abs/1907.10902
- Random Search (JMLR) — https://jmlr.org/papers/v13/bergstra12a.html
- TPE / Algorithms for HPO (NeurIPS 2011) — https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html
- Practical Bayesian Optimization — https://arxiv.org/abs/1206.2944
- Hyperband — https://arxiv.org/abs/1603.06560
- BOHB — https://arxiv.org/abs/1807.01774
- ASHA / Massively Parallel Hyperparameter Tuning — https://arxiv.org/abs/1810.05934
- Tune (Ray Tune) — https://arxiv.org/abs/1807.05118
- scikit-learn grid search guide — https://scikit-learn.org/stable/modules/grid_search.html
- AutoML book (Hutter et al.) — https://www.automl.org/book/
