# Probability Theory for ML

> The mathematics of uncertainty: random variables, distributions, expectation, and concentration inequalities that underpin learning, inference, and generalization in machine learning.

## Why it matters

Almost every ML model is a probabilistic statement about data: classifiers estimate `P(y | x)`, generative models learn `P(x)`, and loss functions are often negative log-likelihoods. Probability theory provides the language for quantifying uncertainty, the calculus (expectation, variance) for reasoning about randomized algorithms like SGD and dropout, and the concentration inequalities that justify *why* empirical risk on a finite sample approximates true risk — the foundation of generalization and sample-complexity bounds.

## Core concepts

- **Probability space** `(Ω, F, P)`: a sample space `Ω`, a σ-algebra of events `F`, and a probability measure `P` with `P(Ω)=1`. Events combine via the axioms of Kolmogorov.
- **Random variable (RV)**: a measurable map `X: Ω → ℝ`. Described by a CDF `F_X(x) = P(X ≤ x)`; a PMF `p(x)` (discrete) or PDF `f(x)` (continuous) where `∫ f = 1`.
- **Joint, marginal, conditional**: `p(x,y)`, `p(x) = Σ_y p(x,y)`, and `p(x|y) = p(x,y)/p(y)`. **Independence**: `p(x,y) = p(x)p(y)`.
- **Bayes' theorem**: `p(θ|D) = p(D|θ) p(θ) / p(D)` — posterior ∝ likelihood × prior. The engine of probabilistic ML and Bayesian inference.
- **Expectation**: `E[X] = Σ x p(x)` or `∫ x f(x) dx`. Linear: `E[aX + bY] = a E[X] + b E[Y]` (no independence needed). **LOTUS**: `E[g(X)] = ∫ g(x) f(x) dx`.
- **Variance & covariance**: `Var(X) = E[(X − E[X])²] = E[X²] − E[X]²`; `Cov(X,Y) = E[XY] − E[X]E[Y]`; correlation `ρ = Cov/(σ_X σ_Y) ∈ [−1,1]`.
- **Moments & MGF**: moment-generating function `M_X(t) = E[e^{tX}]`; used to derive tail (Chernoff) bounds and characterize distributions.
- **Limit theorems**: **Law of Large Numbers** — the sample mean `X̄_n → E[X]` (weak/strong). **Central Limit Theorem** — `√n (X̄_n − μ) / σ → N(0,1)`, justifying Gaussian approximations and confidence intervals.
- **Concentration inequalities** — quantify how fast `X̄_n` concentrates around its mean (non-asymptotic, finite-`n`):
  - **Markov**: for `X ≥ 0`, `P(X ≥ a) ≤ E[X]/a`.
  - **Chebyshev**: `P(|X − μ| ≥ kσ) ≤ 1/k²`.
  - **Chernoff**: `P(X ≥ a) ≤ min_t e^{−ta} M_X(t)` — exponential tail bounds via the MGF.
  - **Hoeffding**: for independent `X_i ∈ [a_i, b_i]`, `P(|X̄ − E[X̄]| ≥ t) ≤ 2 exp(−2n²t² / Σ(b_i−a_i)²)`. The workhorse for generalization bounds.
  - **Bernstein**: sharper than Hoeffding when the variance is small.
  - **McDiarmid (bounded differences)**: concentration for functions of independent variables that change little when one input changes — used for uniform-convergence / Rademacher-complexity bounds.
- **Information-theoretic quantities**: entropy `H(X) = −Σ p log p`, cross-entropy, and **KL divergence** `D_KL(p‖q) = Σ p log(p/q) ≥ 0` — the basis of cross-entropy loss, variational inference, and ELBO objectives.

## Distributions / methods

| Distribution | Support | Key parameters | Typical ML role |
|---|---|---|---|
| Bernoulli / Binomial | {0,1} / {0..n} | `p`, `n` | binary labels, coin-flip events |
| Categorical / Multinomial | classes | `p_1..p_k` | softmax outputs, class labels |
| Gaussian (Normal) | ℝ / ℝ^d | `μ, σ²` / `μ, Σ` | noise models, GMMs, weight priors |
| Poisson | ℕ | `λ` | count data, rare events |
| Exponential / Gamma | ℝ₊ | `λ` / `α, β` | waiting times, conjugate priors |
| Beta | [0,1] | `α, β` | conjugate prior for Bernoulli `p` |
| Dirichlet | simplex | `α` | conjugate prior for Categorical, topic models |
| Uniform | [a,b] | `a, b` | initialization, sampling baselines |
| Laplace | ℝ | `μ, b` | L1 / sparsity priors |
| Student-t | ℝ | `ν` | heavy-tailed / robust models |

| Inequality / tool | Bounds | When to use |
|---|---|---|
| Markov | `P(X≥a) ≤ E[X]/a` | only `E[X]` known, `X≥0` |
| Chebyshev | uses variance | polynomial tail, weak assumptions |
| Chernoff | MGF-based | exponential tails for sums of indep. RVs |
| Hoeffding | bounded RVs | generalization bounds, A/B test sizing |
| Bernstein | variance + bound | tighter than Hoeffding for low variance |
| McDiarmid | bounded differences | concentration of complex statistics |

## Tools & libraries

| Tool | What it does | URL |
|---|---|---|
| NumPy | RNG, sampling, basic stats | https://numpy.org/ |
| SciPy (`scipy.stats`) | distributions, PDFs/CDFs, tests | https://docs.scipy.org/doc/scipy/reference/stats.html |
| statsmodels | statistical models & inference | https://www.statsmodels.org/ |
| scikit-learn | ML estimators, model selection | https://scikit-learn.org/ |
| PyMC | Bayesian modeling / MCMC | https://www.pymc.io/ |
| Stan / CmdStanPy | probabilistic programming (HMC/NUTS) | https://mc-stan.org/ |
| TensorFlow Probability | distributions, bijectors, VI | https://www.tensorflow.org/probability |
| Pyro / NumPyro | deep probabilistic programming | https://pyro.ai/ |
| PyTorch `distributions` | differentiable distributions | https://pytorch.org/docs/stable/distributions.html |
| ArviZ | Bayesian diagnostics & visualization | https://www.arviz.org/ |

## Learning resources

- **Mathematics for Machine Learning** — Deisenroth, Faisal, Ong (Cambridge). Free PDF; Part I covers probability. https://mml-book.com/
- **Probabilistic Machine Learning: An Introduction** — Kevin P. Murphy (MIT Press). Free draft PDF + notebooks. https://probml.github.io/pml-book/book1.html
- **High-Dimensional Probability: An Introduction with Applications in Data Science** — Roman Vershynin (Cambridge). Free PDF; the standard concentration-of-measure reference. https://www.math.uci.edu/~rvershyn/papers/HDP-book/HDP-book.pdf
- **High-Dimensional Statistics: A Non-Asymptotic Viewpoint** — Martin J. Wainwright (Cambridge). Tail bounds, concentration, empirical processes. https://www.cambridge.org/9781108498029
- **Concentration of Measure** lecture notes — Larry Wasserman (CMU). https://www.stat.cmu.edu/~larry/=sml/Concentration.pdf
- **MIT 6.041 / 18.600 Probability** (OCW) — Bertsekas/Tsitsiklis curriculum. https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/
- **StatQuest** (Josh Starmer) — intuitive video explanations of probability & statistics. https://statquest.org/
- **Seeing Theory** — interactive visual introduction to probability (Brown). https://seeing-theory.brown.edu/

## Key papers

- W. Hoeffding, *Probability Inequalities for Sums of Bounded Random Variables*, J. Amer. Statist. Assoc., 1963. https://doi.org/10.1080/01621459.1963.10500830
- H. Chernoff, *A Measure of Asymptotic Efficiency for Tests of a Hypothesis Based on the Sum of Observations*, Ann. Math. Statist., 1952. https://doi.org/10.1214/aoms/1177729330
- C. McDiarmid, *On the Method of Bounded Differences*, Surveys in Combinatorics, 1989. https://doi.org/10.1017/CBO9781107359949.008
- S. Kullback, R. A. Leibler, *On Information and Sufficiency*, Ann. Math. Statist., 1951. https://doi.org/10.1214/aoms/1177729694
- S. Boucheron, G. Lugosi, P. Massart, *Concentration Inequalities: A Nonasymptotic Theory of Independence*, Oxford University Press, 2013. https://doi.org/10.1093/acprof:oso/9780199535255.001.0001

## Cross-references in AIForge

- [Mathematics for ML — overview](../) — sibling foundations (linear algebra, calculus, optimization).
- [Optimization Algorithms](../../Optimization_Algorithms/) — expectation & stochastic methods (SGD) build on probability.
- [Bayesian and Probabilistic ML](../../Bayesian_and_Probabilistic_ML/) — direct application of Bayes' theorem and distributions.
- [Machine Learning](../../Machine_Learning/) — concentration inequalities underpin generalization theory.
- [Model Evaluation](../../Model_Evaluation/) — confidence intervals, hypothesis testing, and CLT in practice.

## Sources

- Mathematics for Machine Learning — companion site. https://mml-book.com/
- Probabilistic Machine Learning (Murphy) — book page. https://probml.github.io/pml-book/book1.html
- High-Dimensional Probability (Vershynin) — free PDF. https://www.math.uci.edu/~rvershyn/papers/HDP-book/HDP-book.pdf
- High-Dimensional Statistics (Wainwright) — Cambridge. https://www.cambridge.org/9781108498029
- Concentration of Measure notes (Wasserman, CMU). https://www.stat.cmu.edu/~larry/=sml/Concentration.pdf
- Concentration Inequalities Quick Reference (UW CSE). https://courses.cs.washington.edu/courses/cse493s/25au/concentration-inequalities.pdf
