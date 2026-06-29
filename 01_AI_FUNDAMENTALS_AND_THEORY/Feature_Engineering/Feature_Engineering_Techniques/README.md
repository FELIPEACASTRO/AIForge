# Feature Engineering Techniques

> The craft of transforming raw data into informative input variables — through interactions, binning, datetime decomposition, text/image representations, and aggregations — while rigorously avoiding leakage.

## Why it matters

Most of the predictive signal a model can use is determined by how the inputs are represented, not by the choice of estimator: well-constructed features routinely beat a fancier model on the same raw columns. Good feature engineering encodes domain knowledge, exposes non-linear and relational structure to simple learners, and makes models more sample-efficient and interpretable. The flip side is that careless transforms — especially target-aware ones — are the single most common source of data leakage and over-optimistic offline metrics.

## Core concepts

- **Interactions & polynomials.** Linear models cannot represent effects like "price matters only for new customers." Adding products/ratios of features ($x_i x_j$, $x_i / x_j$) or polynomial expansions exposes them. Degree-2 expansion of $[a,b]$ gives $[1,a,b,a^2,ab,b^2]$. Trees discover interactions implicitly via splits, so explicit interactions help linear/GLM/kNN models most.
- **Binning (discretization).** Map a continuous $x$ into $K$ ordinal/one-hot buckets. **Equal-width** uses fixed-size intervals; **equal-frequency (quantile)** uses percentile edges so each bin has $\approx n/K$ rows; **k-means** places bins at 1-D cluster centers; **supervised** binning (e.g. decision-tree / MDLP) chooses cut points that maximize target separation. Binning adds robustness to outliers and lets linear models fit step-wise non-linearity, at the cost of resolution.
- **Datetime features.** Decompose a timestamp into calendar parts (year, month, day, dayofweek, hour, is_weekend, is_holiday), elapsed-time deltas, and **cyclical encodings**: $x_{\sin}=\sin(2\pi t/P)$, $x_{\cos}=\cos(2\pi t/P)$ for period $P$ (24h, 7d, 12mo). Both sine and cosine are needed so that, e.g., hour 23 and hour 0 are adjacent and every value is unique.
- **Categorical encoding (target-aware).** **Target/mean encoding** replaces a level $\ell$ with a smoothed estimate of $E[y\mid \ell]$: $\hat{\mu}_\ell=\frac{n_\ell \bar{y}_\ell + m\,\bar{y}}{n_\ell + m}$ (shrinkage toward the global mean $\bar{y}$ by strength $m$). Must be fit with cross-fitting / out-of-fold to avoid leakage; see also `Feature_Scaling_and_Encoding`.
- **Aggregations & relational features.** Group-by statistics (count, mean, std, min/max, recency) over related rows/tables. **Deep Feature Synthesis** formalizes stacking such primitives across linked tables.
- **Text features.** Bag-of-words / n-grams, **TF-IDF** ($\text{tfidf}(t,d)=\text{tf}(t,d)\cdot\log\frac{N}{df(t)}$), hashing vectorizer, and dense embeddings (word2vec, fastText, transformer sentence embeddings).
- **Image features.** Classic descriptors (HOG, SIFT, color histograms, LBP) vs. learned features from pretrained CNN/ViT backbones used as fixed extractors.
- **Leakage.** Any feature derived using information unavailable at prediction time — future values, the target, or full-dataset statistics — inflates offline scores and fails in production. Defenses: fit all transforms **inside a `Pipeline` on training folds only**, use time-aware splits for temporal data, and compute target/aggregate features out-of-fold (or with explicit cutoff times).

## Methods

| Technique | What it does | Typical tool | Notes / pitfalls |
|---|---|---|---|
| Polynomial / interaction terms | Adds $x_i x_j$, powers | `PolynomialFeatures` | Combinatorial blow-up; pair with regularization |
| Equal-width / equal-freq / k-means binning | Discretize continuous vars | `KBinsDiscretizer` | Quantile bins robust to skew |
| Supervised binning (MDLP, tree-based) | Cut points maximize target info | `feature-engine`, custom | Target-aware → fit per fold |
| Datetime decomposition | Extract calendar parts | `pandas.dt`, `feature-engine` | Watch timezone / DST |
| Cyclical (sin/cos) encoding | Wrap periodic features | `feature-engine` `CyclicalFeatures` | Use both sin & cos |
| Target / mean encoding | $E[y\mid\text{cat}]$ with smoothing | `TargetEncoder`, `category_encoders` | High leakage risk; cross-fit |
| Group-by aggregations | Stats over related rows | pandas, featuretools | Out-of-fold for target stats |
| Deep Feature Synthesis | Auto multi-table features | `featuretools` | Use cutoff times vs. leakage |
| Time-series feature extraction | 700+ TS statistics | `tsfresh` | Pair with FDR-controlled selection |
| TF-IDF / n-grams / hashing | Sparse text vectors | `TfidfVectorizer`, `HashingVectorizer` | Fit vocab on train only |
| Text/image embeddings | Dense learned features | `sentence-transformers`, `timm` | Pretrained backbone as extractor |
| Missing-value indicators | Binary "was missing" flag | `MissingIndicator` | Captures informative missingness |

## Tools & libraries

| Tool | Scope | URL |
|---|---|---|
| scikit-learn (`preprocessing`, `compose`, `pipeline`) | Polynomial, binning, target encoding, TF-IDF, leak-safe pipelines | https://scikit-learn.org/stable/modules/preprocessing.html |
| Feature-engine | Pandas-native binning, datetime, cyclical, encoders, selection | https://feature-engine.trainindata.com/ |
| category_encoders | Target, WOE, James-Stein, CatBoost, hashing encoders | https://contrib.scikit-learn.org/category_encoders/ |
| Featuretools | Automated / Deep Feature Synthesis across relational tables | https://www.featuretools.com/ |
| tsfresh | Automatic time-series feature extraction + selection | https://tsfresh.readthedocs.io/ |
| pandas | Datetime accessors, group-by aggregations | https://pandas.pydata.org/docs/ |
| Sentence-Transformers | Dense text/sentence embeddings | https://www.sbert.net/ |
| timm (PyTorch Image Models) | Pretrained image backbones as feature extractors | https://huggingface.co/docs/timm |

## Learning resources

- **Feature Engineering for Machine Learning** — Alice Zheng & Amanda Casari (O'Reilly, 2018). https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/
- **Feature Engineering and Selection** — Max Kuhn & Kjell Johnson (free online book). https://bookdown.org/max/FES/
- **Approaching (Almost) Any Machine Learning Problem** — Abhishek Thakur (categorical & feature chapters). https://github.com/abhishekkrthakur/approachingalmost
- **scikit-learn User Guide — Preprocessing & Column Transformer.** https://scikit-learn.org/stable/modules/preprocessing.html
- **scikit-learn example — Time-related feature engineering** (cyclical, spline, trees vs. linear). https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html
- **Kaggle Learn — Feature Engineering** (free micro-course). https://www.kaggle.com/learn/feature-engineering
- **The Elements of Statistical Learning** (basis expansions, splines — ch. 5). https://hastie.su.domains/ElemStatLearn/
- **MachineLearningMastery — Data preparation without data leakage.** https://machinelearningmastery.com/data-preparation-without-data-leakage/

## Key papers

- Kanter, J. M. & Veeramachaneni, K. (2015). *Deep Feature Synthesis: Towards Automating Data Science Endeavors.* IEEE DSAA. https://doi.org/10.1109/DSAA.2015.7344858
- Christ, M., Braun, N., Neuffer, J. & Kempa-Liehr, A. W. (2018). *Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh — A Python package).* Neurocomputing 307. https://doi.org/10.1016/j.neucom.2018.03.067
- Pargent, F., Pfisterer, F., Thomas, J. & Bischl, B. (2022). *Regularized target encoding outperforms traditional methods in supervised ML with high-cardinality features.* Computational Statistics. https://doi.org/10.1007/s00180-022-01207-6 — arXiv: https://arxiv.org/abs/2104.00629
- Micci-Barreca, D. (2001). *A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems.* ACM SIGKDD Explorations. https://doi.org/10.1145/507533.507538
- Fayyad, U. & Irani, K. (1993). *Multi-interval discretization of continuous-valued attributes for classification learning (MDLP).* IJCAI. https://hdl.handle.net/2014/35171
- Kaufman, S., Rosset, S., Perlich, C. & Stitelman, O. (2012). *Leakage in data mining: formulation, detection, and avoidance.* ACM TKDD. https://doi.org/10.1145/2382577.2382579

## Cross-references in AIForge

- [Feature Scaling and Encoding](../Feature_Scaling_and_Encoding/) — scaling, normalization, one-hot/target encoders.
- [Feature Selection](../Feature_Selection/) — pruning the features you create.
- [Model Evaluation](../../Model_Evaluation/) — leak-safe cross-validation and pipelines.
- [Machine Learning](../../Machine_Learning/) — how estimators consume engineered features.

## Sources

- scikit-learn — KBinsDiscretizer: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html
- scikit-learn — PolynomialFeatures: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
- scikit-learn — TargetEncoder: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html
- Feature-engine — CyclicalFeatures: https://feature-engine.trainindata.com/en/latest/user_guide/creation/CyclicalFeatures.html
- category_encoders docs: https://contrib.scikit-learn.org/category_encoders/
- Featuretools / Deep Feature Synthesis: https://github.com/alteryx/featuretools
- tsfresh: https://github.com/blue-yonder/tsfresh
- MachineLearningMastery — avoiding leakage: https://machinelearningmastery.com/data-preparation-without-data-leakage/
- dotData — preventing leakage in feature engineering: https://dotdata.com/blog/preventing-data-leakage-in-feature-engineering-strategies-and-solutions/
- NVIDIA — encoding time information as features: https://developer.nvidia.com/blog/three-approaches-to-encoding-time-information-as-features-for-ml-models/
