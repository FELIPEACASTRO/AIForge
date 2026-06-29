# Gaussian Processes

> A Gaussian Process (GP) is a distribution over functions such that any finite set of function values is jointly Gaussian — a non-parametric, Bayesian tool for regression and classification with calibrated uncertainty.

## Why it matters

Gaussian Processes give you predictions *with* principled uncertainty estimates, which is exactly what you need in safety-critical, low-data, and active-learning settings (Bayesian optimization, geostatistics/kriging, time series, robotics). They are fully Bayesian, work well in the small-data regime, and let you encode prior structure (smoothness, periodicity, trends) directly through the kernel. The cost is poor scaling — exact inference is O(n³) — which motivates the large family of sparse and GPU-accelerated approximations.

## Core concepts

A GP is fully specified by a **mean function** m(x) and a **covariance (kernel) function** k(x, x′):

- f(x) ~ GP(m(x), k(x, x′)), with E[f(x)] = m(x) and Cov(f(x), f(x′)) = k(x, x′).
- For any finite inputs X = {x₁, …, xₙ}, the vector f = [f(x₁), …, f(xₙ)] is multivariate Gaussian: f ~ N(m, K), where Kᵢⱼ = k(xᵢ, xⱼ).

**GP regression (the closed form).** With Gaussian noise, y = f(x) + ε, ε ~ N(0, σ²), the predictive distribution at test points X* is again Gaussian:

- mean: μ* = K*ᵀ (K + σ²I)⁻¹ y
- covariance: Σ* = K** − K*ᵀ (K + σ²I)⁻¹ K*

where K = k(X, X), K* = k(X, X*), K** = k(X*, X*). The (K + σ²I)⁻¹ inversion is the O(n³) bottleneck (a Cholesky factorization in practice).

**Hyperparameter learning.** Kernel hyperparameters θ (length-scale, signal variance) and noise σ² are tuned by maximizing the **log marginal likelihood**:

- log p(y | X, θ) = −½ yᵀ(K+σ²I)⁻¹y − ½ log|K+σ²I| − (n/2) log 2π.

The data-fit term and the −½ log|K| complexity penalty implement an automatic Occam's razor.

**Kernels** determine the function class. Common choices:

- **RBF / squared-exponential**: k = σ² exp(−‖x−x′‖² / 2ℓ²) — infinitely smooth.
- **Matérn (ν = 1/2, 3/2, 5/2)**: controllable smoothness; ν→∞ recovers RBF; ν=1/2 is the exponential kernel.
- **Periodic**, **Linear (dot-product)**, **Rational Quadratic** — and sums/products of kernels to compose structure. **ARD** (one length-scale per dimension) enables automatic relevance determination.

**GP classification.** With a non-Gaussian likelihood (e.g. logistic/probit on a latent GP), the posterior is non-analytic and requires approximate inference: **Laplace approximation**, **Expectation Propagation (EP)**, or **variational inference**.

## Algorithms / Methods

| Method | Problem | Idea | Notes |
|---|---|---|---|
| Exact GP regression | Regression, Gaussian noise | Closed-form posterior via Cholesky | Gold standard; O(n³), O(n²) memory |
| Laplace approximation | Classification / non-Gaussian | Gaussian approx at posterior mode | Simple, fast, can be biased |
| Expectation Propagation (EP) | Classification | Moment-matching site approximations | Often more accurate than Laplace |
| SoR / FITC / DTC | Scalability | Inducing points, low-rank kernel approx | Classic sparse approximations |
| Variational sparse GP (SGPR) | Scalability | m≪n inducing variables, ELBO bound | Titsias (2009); inducing inputs as variational params |
| SVGP (stochastic variational) | Big data / non-conjugate | Mini-batch ELBO, arbitrary likelihoods | Hensman et al. (2013); scales to millions |
| Deep GPs | Expressiveness | Stack GPs into a hierarchy | Doubly-stochastic variational inference |
| SKI / KISS-GP | Scalability | Structured kernel interpolation + Toeplitz/Kronecker | Near-linear inference |
| BBMM (conjugate gradients) | Scalability / GPU | Matrix-matrix MVM, no explicit inverse | Core of GPyTorch |

## Tools & libraries

| Tool | Backend | URL |
|---|---|---|
| GPyTorch | PyTorch | https://gpytorch.ai/ · [docs](https://docs.gpytorch.ai/) · [GitHub](https://github.com/cornellius-gp/gpytorch) |
| GPflow | TensorFlow / TFP | https://www.gpflow.org/ · [GitHub](https://github.com/GPflow/GPflow) |
| scikit-learn (`gaussian_process`) | NumPy/SciPy | https://scikit-learn.org/stable/modules/gaussian_process.html |
| GPy | NumPy | https://github.com/SheffieldML/GPy |
| GPML toolbox (MATLAB/Octave) | MATLAB | https://gaussianprocess.org/gpml/code/matlab/doc/ |
| Stan | C++ / probabilistic | https://mc-stan.org/ |
| PyMC | PyTensor | https://www.pymc.io/ |
| BoTorch (BO on GPyTorch) | PyTorch | https://botorch.org/ |
| GPJax | JAX | https://github.com/JaxGaussianProcesses/GPJax |

## Learning resources

- **Rasmussen & Williams — *Gaussian Processes for Machine Learning* (MIT Press, 2006)** — the canonical reference, free online: https://gaussianprocess.org/gpml/chapters/RW.pdf · [MIT Press open access](https://direct.mit.edu/books/oa-monograph/2320/Gaussian-Processes-for-Machine-Learning)
- **Murphy — *Probabilistic Machine Learning: Advanced Topics* (MIT Press, 2023)** — modern GP chapters: https://probml.github.io/pml-book/book2.html
- **Deisenroth, Faisal & Ong — *Mathematics for Machine Learning*** — prerequisite linear algebra & probability: https://mml-book.github.io/
- **scikit-learn user guide on Gaussian Processes** — practical, hands-on: https://scikit-learn.org/stable/modules/gaussian_process.html
- **GPyTorch tutorials** (exact GPs, classification, deep GPs): https://docs.gpytorch.ai/en/stable/
- **"A Visual Exploration of Gaussian Processes" — distill.pub**: https://distill.pub/2019/visual-exploration-gaussian-processes/
- **Gaussian Process Summer School (GPSS) lectures & materials**: https://gpss.cc/

## Key papers

- Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press — book PDF: https://gaussianprocess.org/gpml/chapters/RW.pdf
- Titsias, M. (2009). *Variational Learning of Inducing Variables in Sparse Gaussian Processes*. AISTATS (PMLR 5:567–574): https://proceedings.mlr.press/v5/titsias09a.html
- Hensman, J., Fusi, N. & Lawrence, N. D. (2013). *Gaussian Processes for Big Data*. UAI — arXiv:1309.6835: https://arxiv.org/abs/1309.6835
- Gardner, J. R., Pleiss, G., Bindel, D., Weinberger, K. Q. & Wilson, A. G. (2018). *GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration*. NeurIPS — arXiv:1809.11165: https://arxiv.org/abs/1809.11165
- Matthews, A. G. de G. et al. (2017). *GPflow: A Gaussian Process Library using TensorFlow*. JMLR 18(40):1–6: https://jmlr.org/papers/v18/16-537.html
- Salimbeni, H. & Deisenroth, M. (2017). *Doubly Stochastic Variational Inference for Deep Gaussian Processes*. NeurIPS — arXiv:1705.08933: https://arxiv.org/abs/1705.08933

## Cross-references in AIForge

- [Bayesian and Probabilistic ML](../) — parent: Bayesian inference, priors, posteriors
- [Machine Learning](../../Machine_Learning/) — supervised learning foundations
- [Optimization Algorithms](../../Optimization_Algorithms/) — marginal-likelihood maximization, Bayesian optimization
- [Model Evaluation](../../Model_Evaluation/) — uncertainty calibration and validation

## Sources

- Rasmussen & Williams book — https://gaussianprocess.org/gpml/chapters/RW.pdf · https://direct.mit.edu/books/oa-monograph/2320/Gaussian-Processes-for-Machine-Learning
- GPyTorch — https://docs.gpytorch.ai/ · https://github.com/cornellius-gp/gpytorch · https://arxiv.org/abs/1809.11165
- GPflow — https://www.gpflow.org/ · https://jmlr.org/papers/v18/16-537.html · https://github.com/GPflow/GPflow
- scikit-learn — https://scikit-learn.org/stable/modules/gaussian_process.html
- Titsias (2009) — https://proceedings.mlr.press/v5/titsias09a.html
- Hensman et al. (2013) — https://arxiv.org/abs/1309.6835
- Salimbeni & Deisenroth (2017) — https://arxiv.org/abs/1705.08933
- distill.pub visual GP — https://distill.pub/2019/visual-exploration-gaussian-processes/
