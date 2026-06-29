# Automated Feature Engineering

> The algorithmic generation, transformation, and selection of predictive features from raw (often relational or temporal) data, replacing manual, domain-driven feature crafting.

## Why it matters

Feature engineering is historically the most time-consuming and impactful part of an ML pipeline, yet it depends on scarce domain expertise and is hard to reproduce. Automated Feature Engineering (AutoFE) systematizes this step — stacking primitive operations over relational tables or time series to surface signal that raw columns hide — and is a core pillar of AutoML alongside hyperparameter optimization and model selection. Done well, it shortens the path from raw data to a strong baseline and exposes interpretable features that hand-tuned deep models often obscure.

## Core concepts

- **Feature primitives.** Atomic operations applied to data. *Transform primitives* act within a row/entity (e.g., `log`, `month`, `time_since`), while *aggregation primitives* summarize a child table relative to a parent (e.g., `MEAN`, `COUNT`, `STD`).
- **Deep Feature Synthesis (DFS).** Stack primitives across relationships in an `EntitySet`. Each stacked primitive increases the feature's *depth* `d`. A depth-2 feature might be `MEAN(transactions.amount WHERE log(amount) > 0)` per customer. `max_depth` bounds combinatorial blow-up.
- **Relational traversal.** DFS follows foreign-key relationships from a *target entity* to base fields, applying functions along the path — turning a one-to-many schema into one flat feature matrix.
- **Generate-then-select.** AutoFE typically over-generates candidate features, then prunes. Selection scores a candidate's marginal value, e.g., via correlation with the target residual `r = y - ŷ`, mutual information `I(X; Y)`, or incremental gain in a gradient-boosted model.
- **Statistical filtering (FRESH / tsfresh).** Compute a large bank of time-series characteristics, run a univariate hypothesis test per feature (Mann–Whitney U, Kolmogorov–Smirnov, Fisher exact, Kendall τ depending on feature/target type), then control the **false discovery rate** with the Benjamini–Yekutieli procedure at level `α`.
- **Search formulations.** Feature construction is a search over an exponentially large transformation space; framed as greedy hierarchical search (Cognito), reinforcement learning over a transformation graph, or expected-gain-ranked boosting (OpenFE).
- **Leakage & validation.** Features computed using future or target information inflate offline scores; cutoff times and per-fold fitting (fit-on-train, transform-on-test) are essential.

## Algorithms / Methods

| Method | Idea | Data shape | Selection mechanism |
|---|---|---|---|
| Deep Feature Synthesis (DFS) | Stack transform + aggregation primitives across relationships | Relational / multi-table | Depth bound + downstream model importance |
| FRESH / tsfresh | Bank of ~package-defined characteristics + scalable hypothesis tests | Time series | Univariate p-values, BY-FDR control |
| ExploreKit | Generate candidate features, rank with a learned meta-classifier | Tabular | Meta-learned ranking + greedy add |
| Cognito | Hierarchical greedy exploration of a transformation tree | Tabular | Greedy accuracy maximization |
| RL-based FE (Khurana et al.) | Q-learning over a transformation graph | Tabular | Learned policy, budget-constrained |
| autofeat | Generate non-linear feature pool, fit sparse linear model | Tabular | Correlation w/ residual + L1 / model importance |
| OpenFE | Enumerate operator-on-feature candidates, rank by expected gain | Tabular | Feature boosting + two-stage pruning |

## Tools & libraries

| Tool | What it does | URL |
|---|---|---|
| Featuretools | Reference DFS implementation over `EntitySet`s | https://github.com/alteryx/featuretools |
| tsfresh | Automatic extraction + FDR-controlled selection for time series | https://github.com/blue-yonder/tsfresh |
| autofeat | Non-linear feature generation + selection for linear models | https://github.com/cod3licious/autofeat |
| OpenFE | Expert-level automated feature generation for tabular data | https://github.com/IIIS-Li-Group/OpenFE |
| Feature-engine | Scikit-learn-compatible transformers for FE | https://github.com/feature-engine/feature_engine |
| scikit-learn | Pipelines, transformers, `PolynomialFeatures`, selection | https://scikit-learn.org/ |
| tsflex / sktime | Time-series feature extraction frameworks | https://github.com/predict-idlab/tsflex |
| H2O AutoML | End-to-end AutoML incl. feature transforms | https://github.com/h2oai/h2o-3 |

## Learning resources

| Resource | Type | Link |
|---|---|---|
| Featuretools — Deep Feature Synthesis guide | Official docs | https://featuretools.alteryx.com/en/stable/getting_started/afe.html |
| tsfresh documentation | Official docs | https://tsfresh.readthedocs.io/en/latest/ |
| Feature Engineering and Selection (Kuhn & Johnson) | Book (free online) | http://www.feat.engineering/ |
| Feature Engineering for Machine Learning (Zheng & Casari) | Book | https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/ |
| The Elements of Statistical Learning (Hastie, Tibshirani, Friedman) | Book (free PDF) | https://hastie.su.domains/ElemStatLearn/ |
| An Introduction to Statistical Learning (ISLR) | Book (free PDF) | https://www.statlearning.com/ |
| Automated Machine Learning (Hutter, Kotthoff, Vanschoren, eds.) | Book (open access) | https://www.automl.org/book/ |
| inovex — AutoFE with open-source libraries | Tutorial | https://www.inovex.de/en/blog/automated-feature-engineering-open-source-libraries/ |

## Key papers

- Kanter & Veeramachaneni (2015), *Deep Feature Synthesis: Towards Automating Data Science Endeavors*, IEEE DSAA — http://groups.csail.mit.edu/EVO-DesignOpt/groupWebSite/uploads/Site/DSAA_DSM_2015.pdf
- Christ, Braun, Neuffer & Kempa-Liehr (2018), *Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh)*, Neurocomputing 307:72–77 — https://doi.org/10.1016/j.neucom.2018.03.067
- Christ, Kempa-Liehr & Feindt (2016/17), *Distributed and parallel time series feature extraction for industrial big data applications* — https://arxiv.org/abs/1610.07717
- Horn, Pack & Rieger (2019), *The autofeat Python Library for Automated Feature Engineering and Selection* — https://arxiv.org/abs/1901.07329
- Zhang et al. (2023), *OpenFE: Automated Feature Generation with Expert-level Performance*, ICML — https://arxiv.org/abs/2211.12507
- Khurana, Turaga, Samulowitz & Parthasarathy (2016), *Cognito: Automated Feature Engineering for Supervised Learning*, IEEE ICDMW — https://ieeexplore.ieee.org/document/7836821
- Katz, Shin & Song (2016), *ExploreKit: Automatic Feature Generation and Selection*, IEEE ICDM — https://ieeexplore.ieee.org/document/7837936

## Cross-references in AIForge

- [Feature_Engineering](../../Feature_Engineering/) — manual and learned feature techniques
- [Hyperparameter_Optimization](../Hyperparameter_Optimization/) — sibling AutoML pillar
- [Machine_Learning](../../Machine_Learning/) — base models that consume engineered features
- [Model_Evaluation](../../Model_Evaluation/) — validation, leakage avoidance, scoring

## Sources

- https://featuretools.alteryx.com/en/stable/getting_started/afe.html
- https://docs.featuretools.com/en/latest/
- https://github.com/blue-yonder/tsfresh
- https://tsfresh.readthedocs.io/en/latest/
- https://arxiv.org/abs/1901.07329
- https://arxiv.org/abs/1610.07717
- https://arxiv.org/abs/2211.12507
- http://groups.csail.mit.edu/EVO-DesignOpt/groupWebSite/uploads/Site/DSAA_DSM_2015.pdf
- https://github.com/cod3licious/autofeat
- https://www.inovex.de/en/blog/automated-feature-engineering-open-source-libraries/
