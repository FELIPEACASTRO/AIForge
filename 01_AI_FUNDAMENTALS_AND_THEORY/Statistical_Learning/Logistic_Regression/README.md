# Logistic Regression and Classification

> A linear model that estimates the probability of class membership by passing a weighted sum of features through the logistic (sigmoid) or softmax function, fit by maximizing the (regularized) likelihood.

## Why it matters

Logistic regression is the canonical baseline for classification: it is fast, convex (so optimization is reliable), interpretable through coefficients/odds ratios, and produces probabilities rather than bare labels. It underpins large parts of applied statistics, credit scoring, epidemiology, and ad/recommendation ranking, and its softmax generalization is exactly the output layer used by virtually every modern deep classifier. Understanding its loss, regularization, and calibration transfers directly to neural networks.

## Core concepts

- **Logit / sigmoid (binary).** Model the log-odds as linear: `logit(p) = log(p / (1 - p)) = w·x + b`. Inverting gives `p = σ(z) = 1 / (1 + e^(-z))` with `z = w·x + b`. The sigmoid maps any real score to `(0, 1)`.
- **Odds ratio.** A unit increase in feature `x_j` multiplies the odds by `e^(w_j)`, giving direct interpretability of coefficients.
- **Softmax (multiclass).** For `K` classes, `p(y = k | x) = e^(z_k) / Σ_j e^(z_j)` where `z_k = w_k·x + b_k`. This is **multinomial / softmax regression**; binary logistic is the `K = 2` case.
- **Loss = negative log-likelihood (cross-entropy).** Binary: `L = -Σ [y log p + (1 - y) log(1 - p)]`. This is convex in the weights, so gradient methods reach the global optimum. Equivalent to maximum-likelihood estimation (MLE).
- **Optimization.** No closed form; solved with iteratively reweighted least squares (IRLS / Newton), L-BFGS, Newton-CG, SAG/SAGA, or SGD for large/streaming data.
- **Regularization.** Add a penalty to control overfitting and enable feature selection: **L2 / ridge** (`λ‖w‖²₂`, shrinks coefficients), **L1 / lasso** (`λ‖w‖₁`, induces sparsity), **Elastic-Net** (mix of both, handles correlated predictors). In scikit-learn the strength is `C = 1/λ` (smaller `C` = stronger regularization).
- **Decision boundary & threshold.** The boundary `w·x + b = 0` is linear; the 0.5 default threshold can be tuned for the precision/recall or cost trade-off.
- **Calibration.** Probabilities should match observed frequencies. Logistic regression trained by MLE is often well-calibrated, but regularization, class imbalance, or feature engineering can distort it. Assess with reliability diagrams, Brier score, and Expected Calibration Error (ECE); fix with **Platt scaling** (sigmoid), **isotonic regression**, or **temperature scaling**.
- **Assumptions / limits.** Linear in the log-odds, observations independent; sensitive to severe multicollinearity and complete separation (where MLE diverges — regularization or Firth's penalized likelihood helps).

## Variants

| Variant | Output | Key idea / when to use |
|---|---|---|
| Binary logistic regression | `P(y=1)` | Two-class problems; baseline classifier |
| Multinomial (softmax) regression | `P(y=k)`, mutually exclusive classes | Native multiclass; deep-net output layer |
| One-vs-Rest (OvR) | `K` binary models | Multiclass via independent binary fits; simple, parallel |
| Ridge (L2) logistic | shrunk weights | Stabilizes correlated features, default regularizer |
| Lasso (L1) logistic | sparse weights | Feature selection in high dimensions |
| Elastic-Net logistic | sparse + grouped | `p ≫ n`, correlated predictor groups (Zou & Hastie) |
| Ordinal logistic (proportional-odds) | ordered classes | Ratings, severity scales |
| Firth's penalized logistic | bias-reduced MLE | Rare events / separation / small samples |
| Conditional / mixed-effects logistic | grouped data | Matched case-control, hierarchical/clustered data |
| Bayesian logistic regression | posterior over `w` | Uncertainty quantification, priors, small data |

## Tools & libraries

| Tool | Use | URL |
|---|---|---|
| scikit-learn `LogisticRegression` | Penalized binary/multinomial fit, multiple solvers | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html |
| scikit-learn `LogisticRegressionCV` | Built-in CV over regularization strength | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html |
| scikit-learn `CalibratedClassifierCV` | Platt / isotonic post-hoc calibration | https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html |
| statsmodels `Logit` / GLM | Inference: p-values, CIs, odds ratios, diagnostics | https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.html |
| glmnet (R / `glmnet_python`) | Fast L1/L2/Elastic-Net regularization paths | https://glmnet.stanford.edu/ |
| PyMC | Bayesian logistic regression with priors/MCMC | https://www.pymc.io/ |
| Optuna | Hyperparameter search over `C`, penalty, solver | https://optuna.org/ |
| Vowpal Wabbit | Online / out-of-core logistic regression at scale | https://vowpalwabbit.org/ |
| XGBoost / LightGBM | `binary:logistic` / `multi:softprob` objectives (tree baselines) | https://xgboost.readthedocs.io/ |

## Learning resources

- **An Introduction to Statistical Learning (ISLR/ISLP)** — James, Witten, Hastie, Tibshirani. Chapter 4 covers logistic regression accessibly. Free PDF: https://www.statlearning.com/
- **The Elements of Statistical Learning (ESL)** — Hastie, Tibshirani, Friedman. Ch. 4 (linear classifiers) and Ch. 3/18 (regularization). Free PDF: https://hastie.su.domains/ElemStatLearn/
- **Mathematics for Machine Learning (MML book)** — Deisenroth, Faisal, Ong. Optimization and probabilistic foundations. Free: https://mml-book.github.io/
- **Convex Optimization** — Boyd & Vandenberghe. Why the logistic loss is convex and how it is solved. Free: https://web.stanford.edu/~boyd/cvxbook/
- **Probabilistic Machine Learning: An Introduction** — Kevin Murphy. Logistic/softmax regression in a probabilistic framing. Free draft: https://probml.github.io/pml-book/book1.html
- **StatQuest — Logistic Regression playlist** — Josh Starmer, intuitive video series: https://www.youtube.com/playlist?list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe
- **scikit-learn User Guide — Linear Models / Calibration** — https://scikit-learn.org/stable/modules/linear_model.html and https://scikit-learn.org/stable/modules/calibration.html
- **CS229 (Stanford) notes** — Andrew Ng's derivation of logistic regression and the GLM family: https://cs229.stanford.edu/main_notes.pdf

## Key papers

- Cox, D. R. (1958). *The Regression Analysis of Binary Sequences.* J. Royal Statistical Society B — foundational formulation. https://www.jstor.org/stable/2983890
- Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods.* — origin of "Platt scaling". https://www.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf
- Zou, H. & Hastie, T. (2005). *Regularization and Variable Selection via the Elastic Net.* JRSS B 67(2):301–320. https://doi.org/10.1111/j.1467-9868.2005.00503.x
- Niculescu-Mizil, A. & Caruana, R. (2005). *Predicting Good Probabilities with Supervised Learning.* ICML. https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf
- Friedman, J., Hastie, T. & Tibshirani, R. (2010). *Regularization Paths for Generalized Linear Models via Coordinate Descent.* J. Statistical Software 33(1) — the `glmnet` algorithm. https://doi.org/10.18637/jss.v033.i01
- Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. (2017). *On Calibration of Modern Neural Networks.* ICML — temperature scaling; connects softmax calibration to deep nets. https://arxiv.org/abs/1706.04599

## Cross-references in AIForge

- [Generalized Linear Models](../Generalized_Linear_Models/) — logistic regression as the Bernoulli/logit member of the GLM family
- [Regularization (Ridge, Lasso, Elastic-Net)](../Regularization_Ridge_Lasso_ElasticNet/) — penalties used to control overfitting and select features
- [Linear Regression](../Linear_Regression/) — the Gaussian-response counterpart in the same model family
- [Model Evaluation](../../Model_Evaluation/) — ROC/PR curves, log loss, Brier score, calibration metrics
- [Optimization Algorithms](../../Optimization_Algorithms/) — L-BFGS, Newton, SAG/SAGA, and SGD solvers
- [Bayesian and Probabilistic ML](../../Bayesian_and_Probabilistic_ML/) — Bayesian logistic regression and uncertainty
- [Deep Learning](../../Deep_Learning/) — softmax + cross-entropy as the standard classification head

## Sources

- scikit-learn LogisticRegression — https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- scikit-learn Linear Models user guide — https://scikit-learn.org/stable/modules/linear_model.html
- scikit-learn Calibration user guide — https://scikit-learn.org/stable/modules/calibration.html
- statsmodels Logit — https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.html
- Platt (1999), Probabilistic Outputs for SVMs — https://www.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf
- Zou & Hastie (2005), Elastic Net — https://doi.org/10.1111/j.1467-9868.2005.00503.x
- Niculescu-Mizil & Caruana (2005), Predicting Good Probabilities — https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf
- Guo et al. (2017), On Calibration of Modern Neural Networks — https://arxiv.org/abs/1706.04599
- ISLR/ISLP — https://www.statlearning.com/
- ESL — https://hastie.su.domains/ElemStatLearn/
