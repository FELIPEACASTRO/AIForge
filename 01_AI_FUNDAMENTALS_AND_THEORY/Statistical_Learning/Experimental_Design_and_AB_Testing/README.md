# Experimental Design and A/B Testing

> The discipline of randomizing units across treatments to estimate causal effects of a change, and deciding — with statistical guarantees — whether the change is an improvement.

## Why it matters

A/B testing (online controlled experiments) is the gold standard for establishing causality in product, ML, and policy decisions, because randomization breaks the confounding that plagues observational data. Companies like Microsoft, Google, LinkedIn, Booking, and Airbnb run tens of thousands of experiments per year to ship changes with measurable, trustworthy impact. Getting the statistics right — power, peeking, multiple comparisons, variance reduction — separates reliable decisions from a flood of false positives.

## Core concepts

- **Randomized controlled trial (RCT).** Units (users, sessions, requests) are randomly assigned to **control** (A) and **treatment** (B). Randomization makes the groups exchangeable in expectation, so the difference in outcomes is an unbiased estimate of the **average treatment effect (ATE)**: `ATE = E[Y | T=1] - E[Y | T=0]`.
- **Hypothesis testing.** Null `H0: ATE = 0` vs alternative `H1: ATE ≠ 0`. Reject when the test statistic exceeds a threshold. **Type I error** `α` = false positive rate (reject true null); **Type II error** `β` = false negative; **power** `= 1 − β`.
- **Two-sample test.** For means, `z = (x̄_B − x̄_A) / sqrt(s²_A/n_A + s²_B/n_B)`; for conversion rates use the pooled-variance binomial/`z`-test or a chi-squared test.
- **Statistical power & sample size.** For a two-sided test at level `α`, power `1−β`, and standardized effect size (Cohen's) `d = Δ/σ`, the per-arm sample size is approximately `n ≈ 2·(z_{1−α/2} + z_{1−β})² · σ² / Δ²`. Halving the **minimum detectable effect (MDE)** roughly quadruples the required `n`.
- **Confidence interval.** `Δ̂ ± z_{1−α/2}·SE(Δ̂)`; an interval excluding 0 corresponds to a significant result at level `α`.
- **The peeking problem.** Repeatedly checking a fixed-horizon `p`-value and stopping when significant inflates Type I error far above `α` (can exceed 30–40%). Fixed-`n` tests are only valid if you look **once**, at the pre-planned sample size.
- **Sequential testing.** Methods (SPRT, mSPRT, group-sequential, confidence sequences) provide **anytime-valid** inference: error guarantees hold no matter when you stop, enabling continuous monitoring and early stopping.
- **Multiple comparisons.** Testing many metrics/variants inflates false discoveries; control the **family-wise error rate** (Bonferroni, Holm) or the **false discovery rate** (Benjamini–Hochberg).
- **Variance reduction.** **CUPED** (controlled-experiment using pre-experiment data) and regression adjustment use covariates to shrink `Var(Δ̂)`, increasing power without more traffic.
- **Multi-armed bandits (MAB).** Adaptively allocate traffic toward better arms to minimize **cumulative regret** `R_T = Σ_t (μ* − μ_{a_t})`, trading a clean effect estimate for higher in-experiment reward.
- **Common pitfalls.** Sample-ratio mismatch (SRM), network/SUTVA interference, novelty/primacy effects, Simpson's paradox, Twyman's law, and metric dilution.

## Methods

| Method | Idea | When to use |
|---|---|---|
| Fixed-horizon A/B test | Pre-compute `n`, test once at the end | Clean ATE estimate, sufficient traffic, ≤2–3 variants |
| A/B/n & factorial design | Multiple arms / multiple factors at once | Test several variants or interacting factors |
| Two-sample z / t-test | Compare means with normal approximation | Continuous metrics, large samples |
| Chi-squared / proportion z-test | Compare conversion rates | Binary outcomes |
| Welch's t-test | Unequal variances between arms | Heteroscedastic groups |
| SPRT (Wald) | Likelihood-ratio sequential test | Simple null vs simple alternative, early stopping |
| mSPRT / always-valid `p`-values | Mixture SPRT giving anytime-valid `p`/CI | Continuous monitoring without peeking penalty |
| Group-sequential (O'Brien–Fleming, Pocock) | Pre-planned interim analyses with alpha spending | Clinical-style staged looks |
| Confidence sequences | Time-uniform CIs via supermartingales | Nonparametric anytime-valid bounds |
| CUPED / regression adjustment | Covariate-based variance reduction | More power from the same traffic |
| ε-greedy / UCB / Thompson sampling | Adaptive allocation toward best arm | Many/short-lived variants, minimize regret |
| Contextual bandits | Per-context arm selection | Personalization, features per user |
| Switchback / cluster randomization | Randomize over time blocks or clusters | Marketplace / network interference |

## Tools & libraries

| Tool | Purpose | URL |
|---|---|---|
| statsmodels (`stats.power`) | Power/sample-size, t/z/proportion tests | https://www.statsmodels.org/stable/generated/statsmodels.stats.power.TTestIndPower.html |
| SciPy `scipy.stats` | t-test, chi-square, distributions | https://docs.scipy.org/doc/scipy/reference/stats.html |
| scikit-learn | ML metrics, model comparison | https://scikit-learn.org/ |
| PyMC | Bayesian A/B testing | https://www.pymc.io/ |
| pingouin | Power analysis & stats tests | https://pingouin-stats.org/ |
| Vowpal Wabbit | Contextual bandits at scale | https://vowpalwabbit.org/ |
| Optuna | Bandit-style / Bayesian optimization | https://optuna.org/ |
| PlanOut (Meta) | Experiment assignment framework | https://github.com/facebookarchive/planout |
| GrowthBook | Open-source experimentation platform | https://www.growthbook.io/ |
| confidence-sequences (`confseq`) | Anytime-valid CIs (Howard/Ramdas) | https://github.com/gostevehoward/confseq |
| R `pwr` | Power analysis (Cohen) | https://cran.r-project.org/package=pwr |

## Learning resources

- **Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing** — Kohavi, Tang, Xu (Cambridge, 2020). The definitive industry reference. https://www.cambridge.org/core/books/trustworthy-online-controlled-experiments/D97B26382EB0EB2DC2019A7A7B518F59 · companion: https://experimentguide.com/
- **Bandit Algorithms** — Lattimore & Szepesvári (Cambridge, 2020). Free PDF. https://tor-lattimore.com/downloads/book/book.pdf
- **Reinforcement Learning: An Introduction** — Sutton & Barto (2nd ed.); Ch. 2 covers bandits. http://incompleteideas.net/book/the-book-2nd.html
- **An Introduction to Statistical Learning (ISLR)** — James, Witten, Hastie, Tibshirani. https://www.statlearning.com/
- **The Elements of Statistical Learning (ESL)** — Hastie, Tibshirani, Friedman. https://hastie.su.domains/ElemStatLearn/
- **Statistical Rethinking** — McElreath (Bayesian view of inference). https://xcelab.net/rm/
- **Optimizely optimization glossary — A/B testing.** https://www.optimizely.com/optimization-glossary/ab-testing/
- **Microsoft ExP experimentation platform overview.** https://www.microsoft.com/en-us/research/group/experimentation-platform-exp/

## Key papers

- Johari, Pekelis, Koomen, Walsh — *Always Valid Inference: Continuous Monitoring of A/B Tests* (arXiv 2015; Operations Research 2022). https://arxiv.org/abs/1512.04922
- Howard, Ramdas, McAuliffe, Sekhon — *Time-uniform Chernoff bounds via nonnegative supermartingales* (Probability Surveys, 2020). https://arxiv.org/abs/1808.03204
- Howard, Ramdas, McAuliffe, Sekhon — *Time-uniform, nonparametric, nonasymptotic confidence sequences* (Annals of Statistics, 2021). https://arxiv.org/abs/1810.08240
- Auer, Cesa-Bianchi, Fischer — *Finite-time Analysis of the Multiarmed Bandit Problem* (Machine Learning 47, 2002). https://link.springer.com/article/10.1023/A:1013689704352
- Agrawal & Goyal — *Analysis of Thompson Sampling for the Multi-armed Bandit Problem* (COLT 2012). https://arxiv.org/abs/1111.1797
- Deng, Xu, Kohavi, Walker — *Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data (CUPED)* (WSDM 2013). https://dl.acm.org/doi/10.1145/2433396.2433413
- Kohavi, Longbotham, Sommerfield, Henne — *Controlled experiments on the web: survey and practical guide* (Data Mining and Knowledge Discovery, 2009). https://link.springer.com/article/10.1007/s10618-008-0114-1

## Cross-references in AIForge

- [Hypothesis Testing](../Hypothesis_Testing/) — `p`-values, Type I/II error, the foundations underlying A/B tests
- [Resampling and Cross-Validation](../Resampling_and_Cross_Validation/) — bootstrap CIs and permutation tests for effect estimation
- [Causal Inference](../../Causal_Inference/) — treatment effects beyond randomized settings
- [Bayesian and Probabilistic ML](../../Bayesian_and_Probabilistic_ML/) — Bayesian A/B testing and Thompson sampling
- [Reinforcement Learning](../../Reinforcement_Learning/) — bandits as the simplest RL setting

## Sources

- Kohavi, Tang, Xu — *Trustworthy Online Controlled Experiments* (Cambridge UP, 2020). https://www.cambridge.org/core/books/trustworthy-online-controlled-experiments/D97B26382EB0EB2DC2019A7A7B518F59
- Johari et al. — *Always Valid Inference* (arXiv:1512.04922). https://arxiv.org/abs/1512.04922 · INFORMS: https://pubsonline.informs.org/doi/10.1287/opre.2021.2135
- Howard & Ramdas et al. — arXiv:1808.03204 and arXiv:1810.08240
- Auer, Cesa-Bianchi, Fischer (2002). https://link.springer.com/article/10.1023/A:1013689704352
- Agrawal & Goyal (2012). https://arxiv.org/abs/1111.1797
- Deng et al., CUPED (WSDM 2013). https://dl.acm.org/doi/10.1145/2433396.2433413
- statsmodels power docs. https://www.statsmodels.org/stable/generated/statsmodels.stats.power.TTestIndPower.html
- Meta PlanOut. https://github.com/facebookarchive/planout
