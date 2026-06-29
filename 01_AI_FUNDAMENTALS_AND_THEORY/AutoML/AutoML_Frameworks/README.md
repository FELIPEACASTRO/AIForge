# AutoML Frameworks

> End-to-end libraries that automate model selection, hyperparameter tuning, feature processing, and ensembling so a high-quality predictor can be trained from raw data with minimal human intervention.

## Why it matters

AutoML frameworks collapse weeks of manual experimentation into a few lines of code, letting domain experts and non-specialists build competitive models while freeing ML practitioners to focus on problem framing and deployment. By systematically searching the joint space of algorithms and hyperparameters — and ensembling the survivors — modern frameworks routinely match or beat carefully hand-tuned baselines on tabular benchmarks. They also standardize evaluation, reduce researcher-degrees-of-freedom, and make ML reproducible across teams.

## Core concepts

- **CASH (Combined Algorithm Selection and Hyperparameter optimization).** The foundational formalization: jointly choose a learner $A^\*$ and its hyperparameters $\lambda^\*$ to minimize cross-validated loss,
  $$A^\*_{\lambda^\*} = \arg\min_{A^{(j)}\in\mathcal{A},\,\lambda\in\Lambda^{(j)}} \frac{1}{K}\sum_{i=1}^{K} \mathcal{L}\big(A^{(j)}_\lambda; D_{\text{train}}^{(i)}, D_{\text{valid}}^{(i)}\big).$$
  This is a mixed (categorical + continuous + conditional) optimization problem, typically attacked with Bayesian optimization (e.g., SMAC / random-forest surrogates) rather than grid search.
- **Pipeline search.** A pipeline = preprocessing + feature engineering + estimator. Search is over the whole graph, not just the final model. TPOT represents pipelines as trees and evolves them with genetic programming.
- **Surrogate-based optimization.** A cheap probabilistic model $p(\text{loss}\mid\lambda)$ guides where to evaluate next via an acquisition function (e.g., Expected Improvement), trading exploration vs. exploitation.
- **Meta-learning / warm-starting.** Use performance on *prior* datasets (meta-features) to initialize the search near promising configurations, as in auto-sklearn.
- **Multi-layer stacking & ensembling.** Instead of returning a single best model, combine many base models. AutoGluon stacks models in multiple layers with repeated k-fold bagging, which often beats per-model HPO under the same time budget.
- **Cost-frugal search.** FLAML's CFO/BlendSearch start from low-cost configurations and expand toward higher-cost regions, optimizing accuracy *per unit compute* under a wall-clock budget.
- **Time-budgeted anytime behavior.** Frameworks accept a `time_limit` and aim to return the best model found so far, making search anytime and resource-aware.

## Algorithms / Methods

| Method | Search strategy | Core idea | Used by |
|---|---|---|---|
| Bayesian optimization (SMAC) | Random-forest surrogate + EI | Model the loss surface, sample informative configs | auto-sklearn, Auto-WEKA |
| Genetic programming | Evolutionary search over pipeline trees | Mutate/crossover preprocessing+estimator graphs | TPOT |
| Multi-layer stacking + bagging | No HPO search; ensemble many models | Repeated k-fold stacked ensembles dominate single-model tuning | AutoGluon-Tabular |
| Random search over a fixed grid | Train diverse families + stacked ensemble | Cheap, parallelizable, robust baseline | H2O AutoML |
| Cost-frugal optimization (CFO/BlendSearch) | Local search from low-cost seed + BO blend | Maximize accuracy per compute under budget | FLAML |
| Meta-learning warm-start | Initialize from similar past datasets | Skip cold-start by reusing prior knowledge | auto-sklearn |
| Successive halving / Hyperband | Bandit-based resource allocation | Kill bad configs early, reallocate budget | many (as a tuner) |

## Tools & libraries

| Tool | What it does | License | URL |
|---|---|---|---|
| AutoGluon | Tabular/multimodal/time-series AutoML via multi-layer stacking | Apache-2.0 | https://github.com/autogluon/autogluon |
| H2O AutoML | Distributed AutoML with stacked ensembles | Apache-2.0 | https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html |
| FLAML | Fast, cost-frugal AutoML & tuning | MIT | https://github.com/microsoft/FLAML |
| auto-sklearn | Bayesian-opt + meta-learning on scikit-learn | BSD-3 | https://github.com/automl/auto-sklearn |
| TPOT | Genetic-programming pipeline optimization | LGPL-3.0 | https://github.com/EpistasisLab/tpot |
| PyCaret | Low-code AutoML wrapper over scikit-learn | MIT | https://github.com/pycaret/pycaret |
| scikit-learn | Estimators, pipelines, CV that AutoML builds on | BSD-3 | https://scikit-learn.org/ |
| Optuna | General hyperparameter optimization framework | MIT | https://optuna.org/ |
| Ray Tune | Distributed HPO (FLAML/scaling backend) | Apache-2.0 | https://docs.ray.io/en/latest/tune/index.html |
| XGBoost | Gradient boosting; a core base learner in most frameworks | Apache-2.0 | https://xgboost.readthedocs.io/ |
| AMLB | Standardized AutoML benchmark harness | MIT | https://github.com/openml/automlbenchmark |

## Learning resources

- **AutoML: Methods, Systems, Challenges** (Hutter, Kotthoff, Vanschoren, eds.), Springer 2019 — the canonical open-access AutoML book. https://www.automl.org/book/
- **AutoGluon documentation & tutorials** — practical quick-start to advanced stacking. https://auto.gluon.ai/
- **FLAML documentation** — task-oriented examples for tuning and AutoML. https://microsoft.github.io/FLAML/
- **auto-sklearn documentation** — manual, API, and examples. https://automl.github.io/auto-sklearn/master/
- **PyCaret docs** — low-code end-to-end workflow tutorials. https://pycaret.gitbook.io/docs
- **H2O AutoML user guide** — leaderboard, stacked ensembles, explainability. https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
- **AMLB results website** — up-to-date head-to-head framework comparisons. https://openml.github.io/automlbenchmark/

## Key papers

- Thornton, Hutter, Hoos, Leyton-Brown (2013). *Auto-WEKA: Combined Selection and Hyperparameter Optimization of Classification Algorithms* — defines the CASH problem. https://arxiv.org/abs/1208.3719
- Feurer et al. (2015). *Efficient and Robust Automated Machine Learning* (auto-sklearn), NeurIPS 2015. https://proceedings.neurips.cc/paper/2015/hash/11d0e6287202fced83f79975ec59a3a6-Abstract.html
- Olson & Moore (2016). *TPOT: A Tree-based Pipeline Optimization Tool for Automating Machine Learning*, PMLR. https://proceedings.mlr.press/v64/olson_tpot_2016.html
- Wang, Wu, et al. (2019). *FLAML: A Fast and Lightweight AutoML Library*. https://arxiv.org/abs/1911.04706
- Wu, Wang, et al. (2021). *Frugal Optimization for Cost-related Hyperparameters* (CFO), AAAI 2021. https://www.microsoft.com/en-us/research/publication/frugal-optimization-for-cost-related-hyperparameters/
- Erickson et al. (2020). *AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data*. https://arxiv.org/abs/2003.06505
- LeDell & Poirier (2020). *H2O AutoML: Scalable Automatic Machine Learning*, ICML AutoML Workshop. https://www.automl.org/wp-content/uploads/2020/07/AutoML_2020_paper_61.pdf
- Gijsbers et al. (2024). *AMLB: an AutoML Benchmark*, JMLR 25(101). https://jmlr.org/papers/v25/22-0493.html

## Cross-references in AIForge

- [Machine Learning — base learners and supervised learning foundations
- [Optimization Algorithms — Bayesian optimization, evolutionary and bandit methods underpinning AutoML search
- [Model Evaluation — cross-validation and leaderboard metrics used as AutoML objectives
- [Bayesian and Probabilistic ML](../../Bayesian_and_Probabilistic_ML/README.md) — surrogate models and acquisition functions

## Sources

- AutoGluon-Tabular paper — https://arxiv.org/abs/2003.06505
- AutoGluon GitHub — https://github.com/autogluon/autogluon
- FLAML paper — https://arxiv.org/abs/1911.04706
- FLAML GitHub — https://github.com/microsoft/FLAML
- CFO (Microsoft Research) — https://www.microsoft.com/en-us/research/publication/frugal-optimization-for-cost-related-hyperparameters/
- auto-sklearn (NeurIPS 2015) — https://proceedings.neurips.cc/paper/2015/hash/11d0e6287202fced83f79975ec59a3a6-Abstract.html
- auto-sklearn docs — https://automl.github.io/auto-sklearn/master/
- TPOT paper (PMLR) — https://proceedings.mlr.press/v64/olson_tpot_2016.html
- Auto-WEKA / CASH — https://arxiv.org/abs/1208.3719
- H2O AutoML docs — https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
- H2O AutoML paper — https://www.automl.org/wp-content/uploads/2020/07/AutoML_2020_paper_61.pdf
- PyCaret — https://github.com/pycaret/pycaret , https://pycaret.gitbook.io/docs
- AMLB benchmark — https://jmlr.org/papers/v25/22-0493.html , https://arxiv.org/abs/2207.12560
- AutoML book — https://www.automl.org/book/
