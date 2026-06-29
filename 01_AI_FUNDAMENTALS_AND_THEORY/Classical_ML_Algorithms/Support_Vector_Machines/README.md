# Support Vector Machines

> A maximum-margin classifier (and regressor) that finds the hyperplane separating classes with the widest gap, using kernels to handle non-linear boundaries and a soft margin to tolerate noise.

## Why it matters

Support Vector Machines (SVMs) were the dominant off-the-shelf classifier for nearly two decades and remain a strong, well-understood baseline on small-to-medium tabular and text data. Their convex optimization gives a unique global optimum (no local minima), and the "kernel trick" lets a linear algorithm learn highly non-linear decision boundaries without ever materializing the high-dimensional feature space. They are still the method of choice when data is scarce, dimensionality is high (e.g., genomics, text), and interpretable margin-based generalization guarantees matter.

## Core concepts

- **Maximum-margin hyperplane.** For linearly separable data, the SVM finds `w, b` defining `f(x) = w·x + b` that separates classes while maximizing the margin `2/‖w‖`. This is the **hard-margin** primal: minimize `½‖w‖²` subject to `yᵢ(w·xᵢ + b) ≥ 1` for all `i`.
- **Support vectors.** Only the training points lying on the margin (or violating it) have non-zero dual coefficients `αᵢ`; these are the *support vectors*. The solution depends only on them, which is why SVMs are sparse in the dual.
- **Soft margin (C parameter).** Real data is rarely separable. Slack variables `ξᵢ ≥ 0` allow margin violations, minimizing `½‖w‖² + C·Σξᵢ`. The hyperparameter `C` trades off margin width against training error: small `C` = wider margin, more regularization; large `C` = fewer violations, risk of overfitting. This is equivalent to **hinge-loss** minimization with L2 regularization: `Σ max(0, 1 − yᵢf(xᵢ)) + (1/2C)‖w‖²`.
- **The dual problem.** Lagrangian duality converts the primal into maximizing `Σαᵢ − ½ΣΣ αᵢαⱼyᵢyⱼ(xᵢ·xⱼ)` subject to `0 ≤ αᵢ ≤ C` and `Σαᵢyᵢ = 0`. Crucially the data enters only as inner products `xᵢ·xⱼ`.
- **The kernel trick.** Replace `xᵢ·xⱼ` with a kernel `K(xᵢ, xⱼ) = φ(xᵢ)·φ(xⱼ)`, implicitly computing inner products in a (possibly infinite-dimensional) feature space. Any positive semi-definite kernel (Mercer's condition) is valid. Common kernels:
  - Linear: `K(x,z) = x·z`
  - Polynomial: `K(x,z) = (γ x·z + r)^d`
  - RBF / Gaussian: `K(x,z) = exp(−γ‖x−z‖²)` — the default, with width `γ`
  - Sigmoid: `K(x,z) = tanh(γ x·z + r)`
- **KKT conditions.** At optimality the Karush-Kuhn-Tucker conditions partition points: `αᵢ = 0` (inside margin, correctly classified), `0 < αᵢ < C` (on the margin), `αᵢ = C` (margin violators). These conditions drive the SMO training algorithm.
- **Hyperparameters that matter.** `C` (regularization), `γ` (RBF kernel width), kernel choice, and class weights for imbalance. Typically tuned by grid/random search with cross-validation on a log scale.

## Algorithms / Methods

| Variant | Problem | Key idea |
|---|---|---|
| **Hard-margin SVM** | Linearly separable classification | Maximize margin with zero violations |
| **Soft-margin (C-SVM)** | Classification with noise | Slack variables + penalty `C·Σξᵢ` |
| **ν-SVM (Nu-SVM)** | Classification | Reparameterize `C` as `ν ∈ (0,1]` bounding the fraction of support vectors / margin errors |
| **SVR (ε-SVR / ν-SVR)** | Regression | ε-insensitive loss; penalize only errors larger than `ε` |
| **One-Class SVM** | Novelty / outlier detection | Separate data from the origin in feature space |
| **Kernel SVM** | Non-linear boundaries | Apply kernel trick to any of the above |
| **Linear SVM (LIBLINEAR)** | Large-scale, high-dim sparse data | Primal/dual coordinate descent, no kernel; scales to millions of samples |
| **SMO** | Training the dual QP | Decompose into 2-variable subproblems solved analytically |

## Tools & libraries

| Tool | Description | URL |
|---|---|---|
| scikit-learn (`SVC`, `SVR`, `LinearSVC`, `OneClassSVM`) | Python SVM API wrapping LIBSVM/LIBLINEAR | https://scikit-learn.org/stable/modules/svm.html |
| LIBSVM | Reference C/C++ kernel-SVM library (SMO-based) | https://www.csie.ntu.edu.tw/~cjlin/libsvm/ |
| LIBLINEAR | Large-scale linear SVM/logistic regression | https://www.csie.ntu.edu.tw/~cjlin/liblinear/ |
| ThunderSVM | GPU- and multicore-accelerated SVM | https://github.com/Xtra-Computing/thundersvm |
| Optuna | Hyperparameter optimization for `C`, `γ`, kernel | https://optuna.org/ |
| scikit-learn `GridSearchCV` | Built-in cross-validated hyperparameter search | https://scikit-learn.org/stable/modules/grid_search.html |

## Learning resources

| Resource | Type | Link |
|---|---|---|
| *An Introduction to Statistical Learning* (ISLR), Ch. 9 | Book (free PDF) | https://www.statlearning.com/ |
| *The Elements of Statistical Learning* (ESL), Ch. 12 | Book (free PDF) | https://hastie.su.domains/ElemStatLearn/ |
| *Mathematics for Machine Learning* (MML), Ch. 12 | Book (free PDF) | https://mml-book.github.io/ |
| *Convex Optimization* — Boyd & Vandenberghe | Book (free PDF) | https://web.stanford.edu/~boyd/cvxbook/ |
| *Pattern Recognition and ML* — Bishop, Ch. 7 | Book | https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/ |
| scikit-learn SVM user guide | Tutorial / docs | https://scikit-learn.org/stable/modules/svm.html |
| StatQuest — Support Vector Machines | Video series | https://www.youtube.com/watch?v=efR1C6CvhmE |
| MIT 6.034 — Learning: SVMs (Patrick Winston) | Lecture | https://www.youtube.com/watch?v=_PwhiWxHK8o |

## Key papers

| Year | Paper | Authors | Link |
|---|---|---|---|
| 1992 | A Training Algorithm for Optimal Margin Classifiers (COLT'92) — introduces the kernel-margin idea | Boser, Guyon, Vapnik | https://doi.org/10.1145/130385.130401 |
| 1995 | Support-Vector Networks (*Machine Learning* 20:273-297) — the soft-margin SVM | Cortes, Vapnik | https://doi.org/10.1007/BF00994018 |
| 1998 | Sequential Minimal Optimization (MSR-TR-98-14) — the SMO training algorithm | Platt | https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/ |
| 2004 | A Tutorial on Support Vector Regression | Smola, Schölkopf | https://doi.org/10.1023/B:STCO.0000035301.49549.88 |
| 2008 | LIBLINEAR: A Library for Large Linear Classification (*JMLR*) | Fan, Chang, Hsieh, Wang, Lin | https://www.jmlr.org/papers/v9/fan08a.html |
| 2011 | LIBSVM: A Library for Support Vector Machines (*ACM TIST* 2:27) | Chang, Lin | https://doi.org/10.1145/1961189.1961199 |

## Cross-references in AIForge

- [Machine Learning — broader supervised-learning context and the bias-variance framing of `C`/`γ`.
- [Optimization Algorithms — convex/quadratic programming and duality underpinning SVM training.
- [Model Evaluation — cross-validation and metrics for tuning SVM hyperparameters.
- [Bayesian and Probabilistic ML](../../Bayesian_and_Probabilistic_ML/README.md) — Gaussian processes as the kernelized probabilistic cousin of kernel SVMs.

## Sources

- Boser, Guyon, Vapnik (1992), *A Training Algorithm for Optimal Margin Classifiers* — https://doi.org/10.1145/130385.130401
- Cortes & Vapnik (1995), *Support-Vector Networks* — https://doi.org/10.1007/BF00994018
- Platt (1998), *Sequential Minimal Optimization* — https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/
- Chang & Lin (2011), *LIBSVM* — https://doi.org/10.1145/1961189.1961199 and https://www.csie.ntu.edu.tw/~cjlin/libsvm/
- scikit-learn SVM documentation — https://scikit-learn.org/stable/modules/svm.html
- *An Introduction to Statistical Learning* — https://www.statlearning.com/
- *The Elements of Statistical Learning* — https://hastie.su.domains/ElemStatLearn/
