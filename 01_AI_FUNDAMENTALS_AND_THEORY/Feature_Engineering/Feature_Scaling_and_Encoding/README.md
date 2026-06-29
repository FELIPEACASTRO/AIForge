# Feature Scaling and Encoding

> Transforming raw numeric and categorical features into a model-ready numeric representation by rescaling magnitudes and mapping categories to numbers.

## Why it matters

Most ML algorithms assume features live on comparable scales: distance-based methods (k-NN, k-means, SVM with RBF), gradient-descent learners (linear/logistic regression, neural nets), and PCA are all distorted when one feature dominates by raw magnitude. Categorical variables, meanwhile, cannot be consumed directly by numeric models and naive integer codes inject spurious ordinality. Correct scaling and encoding therefore decide convergence speed, regularization fairness, and the difference between a leaky model and a generalizing one.

## Core concepts

**Standardization (z-score).** Center to zero mean, scale to unit variance:
`z = (x − μ) / σ`. Produces unbounded values, preserves outlier shape, ideal for Gaussian-ish features and gradient-based / distance-based models. Fit μ, σ on the **training set only**, then apply to validation/test to avoid leakage.

**Normalization (min–max).** Rescale to a fixed range, usually [0, 1]:
`x' = (x − min) / (max − min)`. Bounded output, but highly sensitive to outliers (a single extreme value compresses everything else).

**Robust scaling.** Use median and IQR: `x' = (x − median) / IQR`. Far less sensitive to outliers than z-score or min–max.

**Vector normalization (L2/L1).** Scale each *sample* (row) to unit norm — common in text/TF-IDF and cosine-similarity settings — distinct from per-*feature* scaling above.

**Nonlinear transforms.** Log / Box–Cox / Yeo–Johnson (power transforms) and quantile transforms reshape skewed distributions toward Gaussianity or uniformity before or instead of linear scaling.

**Categorical encoding** maps discrete labels to numbers. Key axes: *nominal vs ordinal*, *low vs high cardinality*, and *unsupervised vs target-aware*.
- **One-hot**: one binary column per category — no false ordering, but explodes dimensionality on high cardinality.
- **Ordinal/label**: single integer column — correct only when a true order exists.
- **Target (mean) encoding**: replace a category with a (smoothed) statistic of the target for that category. Powerful for high-cardinality features but prone to **target leakage / overfitting** unless regularized.

**Target-encoding smoothing.** Blend the category mean with the global mean:
`enc(c) = (n_c · ȳ_c + m · ȳ) / (n_c + m)`, where `n_c` is the category count, `ȳ_c` the category target mean, `ȳ` the global mean, and `m` a smoothing prior. Rare categories shrink toward the global mean. Use **cross-fitting** (out-of-fold), leave-one-out, or CatBoost-style ordered encoding to prevent leakage.

**Where scaling does/doesn't matter.** Tree ensembles (Random Forest, gradient boosting) are invariant to monotonic feature scaling, so standardization is usually unnecessary for them — but **encoding** still matters everywhere.

## Variants

| Technique | Formula / idea | Output range | Outlier-robust | Best for |
|---|---|---|---|---|
| Standardization (z-score) | `(x−μ)/σ` | unbounded | No | linear models, NN, PCA, SVM |
| Min–max normalization | `(x−min)/(max−min)` | [0,1] | No | bounded inputs, image pixels |
| MaxAbs scaling | `x/max(|x|)` | [−1,1] | No | sparse data (keeps zeros) |
| Robust scaling | `(x−median)/IQR` | unbounded | Yes | heavy-tailed / outlier-rich |
| L2 / L1 normalization | scale row to unit norm | unit sphere | — | text, cosine similarity |
| Power transform (Box–Cox / Yeo–Johnson) | parametric power map | ≈Gaussian | partial | skewed positive (BC) / any-sign (YJ) |
| Quantile transform | map to uniform/normal via CDF | [0,1] or N(0,1) | Yes | nonlinear, robust reshaping |
| One-hot encoding | binary indicator per level | {0,1} | — | low-cardinality nominal |
| Ordinal / label encoding | integer per level | ℤ | — | genuinely ordered categories |
| Target / mean encoding | smoothed target stat per level | ℝ | — | high-cardinality nominal |
| Leave-one-out / CV target enc. | out-of-fold target stat | ℝ | — | leakage-safe target encoding |
| CatBoost (ordered) encoding | permutation-based running mean | ℝ | — | high-card. in boosting |
| Weight of Evidence (WoE) | `ln(P(x|1)/P(x|0))` | ℝ | — | binary classification, credit scoring |
| Hashing (feature hashing) | hash level → fixed buckets | sparse | — | very high cardinality, streaming |
| Frequency / count encoding | level → its frequency/count | ℕ/ℝ | — | cheap high-cardinality baseline |
| Binary / BaseN encoding | level → base-N digits as columns | {0,1} | — | mid-cardinality, dim reduction |

## Tools & libraries

| Tool | What it provides | URL |
|---|---|---|
| scikit-learn `preprocessing` | StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, PowerTransformer, QuantileTransformer, OneHotEncoder, OrdinalEncoder, TargetEncoder | https://scikit-learn.org/stable/modules/preprocessing.html |
| scikit-learn `ColumnTransformer` / `Pipeline` | Apply different transforms per column, leak-free inside CV | https://scikit-learn.org/stable/modules/compose.html |
| category_encoders (sklearn-contrib) | Target, LeaveOneOut, CatBoost, James-Stein, M-estimator, WoE, Hashing, BaseN, etc. | https://contrib.scikit-learn.org/category_encoders/ |
| CatBoost | Native ordered target encoding for categorical features | https://catboost.ai/ |
| LightGBM | Native categorical-feature handling (Fisher-style splits) | https://lightgbm.readthedocs.io/ |
| pandas | `get_dummies`, `Categorical`, `factorize` for quick encoding | https://pandas.pydata.org/docs/ |
| Feature-engine | Sklearn-compatible scalers & encoders with rich transformers | https://feature-engine.trainindata.com/ |
| TensorFlow Keras preprocessing layers | Normalization, StringLookup, IntegerLookup, CategoryEncoding | https://www.tensorflow.org/guide/keras/preprocessing_layers |

## Learning resources

| Resource | Type | Link |
|---|---|---|
| *An Introduction to Statistical Learning* (ISLR) | Free book — preprocessing, model basics | https://www.statlearning.com/ |
| *The Elements of Statistical Learning* (ESL) | Free book — feature transforms, basis expansions | https://hastie.su.domains/ElemStatLearn/ |
| scikit-learn "Preprocessing data" User Guide | Authoritative how-to with examples | https://scikit-learn.org/stable/modules/preprocessing.html |
| scikit-learn "Compare effect of scalers on data with outliers" | Visual intuition for each scaler | https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html |
| *Feature Engineering and Selection* (Kuhn & Johnson) | Free online book | https://bookdown.org/max/FES/ |
| *Feature Engineering A-Z* (Emil Hvitfeldt) | Free online reference (incl. target encoding) | https://feaz-book.com/ |
| Kaggle "Categorical Variables" micro-course | Hands-on tutorial | https://www.kaggle.com/code/alexisbcook/categorical-variables |
| category_encoders documentation | Per-encoder math & usage | https://contrib.scikit-learn.org/category_encoders/ |

## Key papers

- Micci-Barreca, D. (2001). *A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems.* ACM SIGKDD Explorations 3(1):27–32. — origin of smoothed target/impact encoding. https://dl.acm.org/doi/10.1145/507533.507538
- Prokhorenkova, L. et al. (2018). *CatBoost: unbiased boosting with categorical features.* NeurIPS 2018. — ordered target encoding that avoids target leakage. https://arxiv.org/abs/1706.09516
- Pargent, F. et al. (2022). *Regularized target encoding outperforms traditional methods in supervised machine learning with high cardinality features.* Computational Statistics. https://link.springer.com/article/10.1007/s00180-022-01207-6 (preprint: https://arxiv.org/abs/2104.00629)
- Box, G. E. P. & Cox, D. R. (1964). *An Analysis of Transformations.* JRSS-B 26(2):211–252. — the Box–Cox power transform. https://www.jstor.org/stable/2984418
- Yeo, I.-K. & Johnson, R. A. (2000). *A New Family of Power Transformations to Improve Normality or Symmetry.* Biometrika 87(4):954–959. https://doi.org/10.1093/biomet/87.4.954

## Cross-references in AIForge

- [../Feature_Engineering_Techniques/](../Feature_Engineering_Techniques/) — broader feature construction & transformation
- [../Feature_Selection/](../Feature_Selection/) — choosing which features to keep
- [../../Machine_Learning/](../../Machine_Learning/) — models that consume scaled/encoded features
- [../../Model_Evaluation/](../../Model_Evaluation/) — leakage-safe validation of preprocessing pipelines
- [../../Optimization_Algorithms/](../../Optimization_Algorithms/) — why scaling speeds gradient-based training

## Sources

- scikit-learn — Preprocessing data: https://scikit-learn.org/stable/modules/preprocessing.html
- scikit-learn — StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- scikit-learn — MinMaxScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
- scikit-learn — TargetEncoder: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html
- category_encoders documentation: https://contrib.scikit-learn.org/category_encoders/
- Micci-Barreca (2001), ACM SIGKDD Explorations: https://dl.acm.org/doi/10.1145/507533.507538
- Prokhorenkova et al. (2018), CatBoost: https://arxiv.org/abs/1706.09516
- Pargent et al. (2022), Regularized target encoding: https://link.springer.com/article/10.1007/s00180-022-01207-6
