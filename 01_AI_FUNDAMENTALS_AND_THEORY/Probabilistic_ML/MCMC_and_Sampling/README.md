# MCMC and Sampling

> Markov Chain Monte Carlo (MCMC) draws correlated samples from a target distribution by simulating a Markov chain whose stationary distribution is that target — enabling approximate inference when the distribution is known only up to a normalizing constant.

## Why it matters

Most Bayesian models have posteriors `p(θ | data) ∝ p(data | θ) p(θ)` whose normalizing constant (the evidence) is an intractable high-dimensional integral, so we cannot sample or integrate directly. MCMC sidesteps this by requiring only the *unnormalized* density, turning expectations like posterior means, credible intervals, and predictive distributions into Monte Carlo averages over chain samples. It is the workhorse behind modern probabilistic programming (Stan, PyMC, NumPyro) and remains the gold-standard for asymptotically exact inference when variational methods are too biased.

## Core concepts

- **Target distribution**: we want samples from `π(θ)`, known only as `π(θ) = f(θ)/Z` with unknown `Z`. MCMC never needs `Z`.
- **Markov chain**: a sequence where `θ_{t+1}` depends only on `θ_t`. If the chain is *irreducible*, *aperiodic*, and admits `π` as its **stationary distribution**, then `θ_t → π` in distribution as `t → ∞`.
- **Detailed balance (reversibility)**: a sufficient condition for stationarity — `π(x) T(x→y) = π(y) T(y→x)` for transition kernel `T`. Metropolis–Hastings is constructed to satisfy this.
- **Acceptance ratio (Metropolis–Hastings)**: propose `θ'` from `q(θ'|θ)`, accept with probability `α = min(1, [f(θ') q(θ|θ')] / [f(θ) q(θ'|θ)])`. The unknown `Z` cancels. For symmetric `q`, this reduces to the original Metropolis rule `α = min(1, f(θ')/f(θ))`.
- **Burn-in / warmup**: discard early samples drawn before the chain reaches stationarity; samplers like NUTS also use warmup to adapt step size and the mass matrix.
- **Autocorrelation & effective sample size (ESS)**: consecutive samples are correlated, so `N` draws are worth `N_eff < N` independent draws. `ESS = N / (1 + 2 Σ_k ρ_k)` where `ρ_k` are autocorrelations.
- **Convergence diagnostics**: the **Gelman–Rubin statistic `R̂`** compares between- and within-chain variance across multiple chains (`R̂ ≈ 1.0`, e.g. `< 1.01`, suggests convergence); trace plots and ESS complement it.
- **Mixing**: how fast the chain explores the support. Random-walk proposals mix slowly in high dimensions; gradient-based methods (HMC/NUTS) mix far better by suppressing diffusive behavior.
- **Detailed-balance-free / gradient methods**: HMC augments `θ` with momentum `r`, simulates Hamiltonian dynamics `H(θ,r) = −log f(θ) + ½ rᵀM⁻¹r` via the leapfrog integrator, and uses a Metropolis correction for discretization error.

## Algorithms / Methods

| Method | Idea | Needs gradients? | Best for | Notes / weaknesses |
|---|---|---|---|---|
| **Metropolis** (1953) | Symmetric random-walk proposal + accept/reject | No | Low-dim, simple targets | Slow mixing in high dim; tune step size |
| **Metropolis–Hastings** (1970) | Generalizes to asymmetric proposals `q` | No | General-purpose | Performance hinges on proposal design |
| **Gibbs sampling** | Cycle through full conditionals `p(θ_i \| θ_{-i})` | No | Conditionally-conjugate models | Needs tractable conditionals; slow with strong correlation |
| **Metropolis-within-Gibbs** | MH step for conditionals lacking closed form | No | Hierarchical models | Combines both; still can mix poorly |
| **MALA** (Langevin) | Gradient-biased proposal `θ + ½ε²∇log f + ε·noise` | Yes | Continuous targets | Cheaper than HMC; sensitive to step size |
| **HMC** | Hamiltonian dynamics + leapfrog + MH correction | Yes | High-dim continuous | Must tune step size & path length |
| **NUTS** | HMC that auto-tunes trajectory length (no U-turn) | Yes | Default in Stan/PyMC | No path-length tuning; struggles w/ discrete params |
| **Slice sampling** | Sample uniformly under the density curve | No | Univariate / few params | Auto-adapts scale; awkward in high dim |
| **Reversible-jump MCMC** | Moves across models of differing dimension | No | Model selection / variable dim | Complex to design valid jumps |
| **Tempering / Parallel tempering** | Multiple chains at different "temperatures" | Optional | Multimodal targets | Helps escape modes; extra cost |
| **SMC / Particle MCMC** | Sequential resampling of weighted particles | No | State-space / sequential models | Combines well with MCMC (PMCMC) |

## Tools & libraries

| Tool | Language | Sampler(s) | URL |
|---|---|---|---|
| **Stan** | Stan DSL (R/Python/CLI) | NUTS, HMC, ADVI | https://mc-stan.org/ |
| **PyMC** | Python | NUTS, Metropolis, Gibbs-like, SMC | https://www.pymc.io/ |
| **NumPyro** | Python (JAX) | NUTS/HMC, SA, GPU-accelerated | https://num.pyro.ai/ |
| **Pyro** | Python (PyTorch) | NUTS/HMC, SVI | https://pyro.ai/ |
| **TensorFlow Probability** | Python | HMC, NUTS, RWM, MALA | https://www.tensorflow.org/probability |
| **emcee** | Python | Affine-invariant ensemble sampler | https://emcee.readthedocs.io/ |
| **BlackJAX** | Python (JAX) | HMC, NUTS, SMC, MALA | https://blackjax-devs.github.io/blackjax/ |
| **ArviZ** | Python | Diagnostics (R̂, ESS, plots) | https://www.arviz.org/ |
| **JAGS** | BUGS dialect | Gibbs, slice, Metropolis | https://mcmc-jags.sourceforge.io/ |
| **Turing.jl** | Julia | NUTS/HMC, particle, Gibbs | https://turinglang.org/ |
| **brms / rstanarm** | R (Stan front-ends) | NUTS via Stan | https://pas.craft.me/ (see https://mc-stan.org/users/interfaces/) |

## Learning resources

- **Bayesian Data Analysis (3rd ed.)** — Gelman, Carlin, Stern, Dunson, Vehtari, Rubin. Free PDF: https://www.stat.columbia.edu/~gelman/book/ — the canonical reference; chapters 11–12 cover MCMC and HMC.
- **Handbook of Markov Chain Monte Carlo** — Brooks, Gelman, Jones, Meng (eds.). Sample chapters: https://www.mcmchandbook.net/HandbookSampleChapters.html
- **Pattern Recognition and Machine Learning** — Bishop, Ch. 11 "Sampling Methods". Free PDF: https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/
- **Probabilistic Machine Learning: Advanced Topics** — Murphy (2023). Free: https://probml.github.io/pml-book/book2.html — Part on inference covers MCMC/HMC.
- **Information Theory, Inference, and Learning Algorithms** — MacKay, Ch. 29–30 (sampling). Free: https://www.inference.org.uk/mackay/itila/
- **A Conceptual Introduction to Hamiltonian Monte Carlo** — Betancourt (2017), an outstanding HMC explainer: https://arxiv.org/abs/1701.02434
- **Stan User's Guide & MCMC concepts**: https://mc-stan.org/docs/
- **Statistical Rethinking** — McElreath (course + book, uses Stan): https://xcelab.net/rm/ (lectures: https://www.youtube.com/playlist?list=PLDcUM9US4XdPz-KxHM4XHt7uUVGWWVSus)
- **MCMC Interactive Gallery** — Chi Feng's animations of samplers: https://chi-feng.github.io/mcmc-demo/
- **PyMC tutorials / example gallery**: https://www.pymc.io/projects/examples/

## Key papers

- Metropolis, Rosenbluth, Rosenbluth, Teller & Teller (1953). *Equation of State Calculations by Fast Computing Machines.* J. Chem. Phys. 21:1087. DOI: https://doi.org/10.1063/1.1699114
- Hastings, W.K. (1970). *Monte Carlo Sampling Methods Using Markov Chains and Their Applications.* Biometrika 57(1):97–109. DOI: https://doi.org/10.1093/biomet/57.1.97
- Geman, S. & Geman, D. (1984). *Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images.* IEEE TPAMI 6(6):721–741. DOI: https://doi.org/10.1109/TPAMI.1984.4767596
- Duane, Kennedy, Pendleton & Roweth (1987). *Hybrid Monte Carlo.* Physics Letters B 195(2):216–222. DOI: https://doi.org/10.1016/0370-2693(87)91197-X (origin of HMC)
- Gelman, A. & Rubin, D.B. (1992). *Inference from Iterative Simulation Using Multiple Sequences.* Statistical Science 7(4):457–472. DOI: https://doi.org/10.1214/ss/1177011136 (the `R̂` diagnostic)
- Neal, R.M. (2011). *MCMC Using Hamiltonian Dynamics.* In Handbook of MCMC. arXiv: https://arxiv.org/abs/1206.1901
- Hoffman, M.D. & Gelman, A. (2014). *The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo.* JMLR 15:1593–1623. URL: https://jmlr.org/papers/v15/hoffman14a.html — arXiv: https://arxiv.org/abs/1111.4246
- Carpenter et al. (2017). *Stan: A Probabilistic Programming Language.* Journal of Statistical Software 76(1). DOI: https://doi.org/10.18637/jss.v076.i01

## Cross-references in AIForge

- [Bayesian and Probabilistic ML](../../Bayesian_and_Probabilistic_ML/) — priors, posteriors, and the inference problems MCMC solves.
- [Optimization Algorithms](../../Optimization_Algorithms/) — variational inference and gradient methods that complement / contrast with sampling.
- [Machine Learning](../../Machine_Learning/) — broader modeling context for probabilistic methods.
- [Model Evaluation](../../Model_Evaluation/) — assessing predictive performance of fitted Bayesian models.

## Sources

- W.K. Hastings biography & 1970 paper context — https://probability.ca/hastings/ and https://academic.oup.com/biomet/article-abstract/107/1/1/5686745
- No-U-Turn Sampler (JMLR) — https://jmlr.org/papers/v15/hoffman14a.html ; arXiv https://arxiv.org/abs/1111.4246
- Neal, *MCMC using Hamiltonian dynamics* — https://arxiv.org/pdf/1206.1901 ; Handbook chapters https://www.mcmchandbook.net/HandbookSampleChapters.html
- Geman & Geman 1984 (Gibbs) — https://www.scirp.org/reference/referencespapers?referenceid=840838 (DOI 10.1109/TPAMI.1984.4767596)
- Stan / PyMC / NumPyro docs — https://mc-stan.org/ , https://www.pymc.io/ , https://num.pyro.ai/en/stable/getting_started.html
- Betancourt, *A Conceptual Introduction to HMC* — https://arxiv.org/abs/1701.02434
