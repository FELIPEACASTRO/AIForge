# Statistical Hypothesis Testing

> A framework for deciding whether observed data provide enough evidence to reject a default ("null") claim about a population, while quantifying and controlling the risk of being wrong.

## Why it matters

Hypothesis testing is the backbone of empirical claims in ML and science: it tells you whether a measured difference (model A vs. B, treatment vs. control, feature signal vs. noise) is real or just sampling variability. Getting it right — choosing the correct test, interpreting p-values honestly, correcting for many comparisons, and reporting effect sizes — separates reproducible results from false discoveries. The "replication crisis" in many fields traces directly to misuse of these tools.

## Core concepts

- **Null and alternative hypotheses.** H0 encodes "no effect"; H1 the effect of interest. A test produces a statistic whose distribution under H0 is known.
- **p-value.** P(observing a statistic at least as extreme as the data | H0 true). It is *not* P(H0 true) and *not* the probability the result is due to chance in the colloquial sense.
- **Significance level α and errors.** Reject H0 if p < α (commonly 0.05). **Type I error** = false positive (rate α); **Type II error** = false negative (rate β). **Power** = 1 − β.
- **Test statistic to p-value.** E.g. the one-sample t-statistic: `t = (x̄ − μ0) / (s / √n)`, compared to a Student-t distribution with `n − 1` degrees of freedom.
- **Confidence intervals.** A 95% CI is dual to a two-sided test at α = 0.05: values outside the CI are rejected. CIs convey magnitude + uncertainty, unlike a bare p-value.
- **Effect size.** Standardized magnitude independent of n — e.g. **Cohen's d** = `(μ1 − μ2)/σ_pooled`, **Pearson r**, **η² / ω²** for ANOVA, **Cramér's V** for χ². Statistical significance ≠ practical significance.
- **Parametric vs. nonparametric.** Parametric tests assume a distributional form (often normality); nonparametric (rank-based) tests relax this at some power cost.
- **Multiple testing.** Running m tests at α inflates the chance of at least one false positive. Control the **family-wise error rate (FWER)** (Bonferroni, Holm) or the **false discovery rate (FDR)** (Benjamini–Hochberg), trading stringency for power.
- **Bayesian alternative.** Bayes factors and posterior probabilities address some p-value pitfalls by comparing models directly; see the Bayesian section below.

## Algorithms / Methods

| Test | Question answered | Assumptions | Nonparametric analog |
|------|-------------------|-------------|----------------------|
| One-sample t-test | Is a mean equal to μ0? | Normal-ish, iid | Wilcoxon signed-rank |
| Two-sample (Welch) t-test | Do two group means differ? | Normal-ish, unequal var OK | Mann–Whitney U |
| Paired t-test | Do paired measurements differ? | Normal differences | Wilcoxon signed-rank |
| One-way ANOVA (F-test) | Do ≥3 group means differ? | Normal, equal variance | Kruskal–Wallis |
| Repeated-measures / two-way ANOVA | Effects of factors + interaction | Normal, sphericity | Friedman |
| Pearson χ² test | Association in a contingency table | Expected counts ≥ ~5 | Fisher's exact test |
| χ² goodness-of-fit | Does data match expected distribution? | Adequate expected counts | — |
| z-test (proportions) | Do proportions differ? | Large n | Exact binomial |
| Shapiro–Wilk / Kolmogorov–Smirnov | Is data normal / from a distribution? | — | — |
| Levene / Bartlett | Are variances equal? | (Bartlett assumes normality) | — |
| Likelihood-ratio / Wald / score | Nested model comparison | MLE regularity | — |
| Permutation / bootstrap test | Any statistic, minimal assumptions | Exchangeability | — |

**Multiple-testing corrections**

| Method | Controls | Notes |
|--------|----------|-------|
| Bonferroni | FWER | Reject if p < α/m; simple, conservative |
| Holm–Bonferroni | FWER | Step-down, uniformly more powerful than Bonferroni |
| Šidák | FWER | Assumes independence; slightly less conservative |
| Benjamini–Hochberg (BH) | FDR | Step-up; far more powerful for large m |
| Benjamini–Yekutieli | FDR | Valid under arbitrary dependence |
| Tukey HSD | FWER (pairwise) | Post-hoc after ANOVA |

## Tools & libraries

| Tool | What it offers | URL |
|------|----------------|-----|
| SciPy `scipy.stats` | t/χ²/ANOVA, normality, nonparametric tests, distributions | https://docs.scipy.org/doc/scipy/reference/stats.html |
| statsmodels | ANOVA, GLM, `multipletests` (FWER/FDR), Tukey HSD, power | https://www.statsmodels.org/stable/stats.html |
| statsmodels `multitest` | Bonferroni/Holm/BH/BY p-value correction | https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html |
| statsmodels `power` | Power & sample-size analysis | https://www.statsmodels.org/stable/stats.html#power-and-sample-size-calculations |
| pingouin | Pythonic stats: effect sizes, post-hoc, Bayes factors | https://pingouin-stats.org/ |
| Penn `statsmodels` GLM/AOV | Regression-based F-tests | https://www.statsmodels.org/stable/anova.html |
| R `stats` (base) | `t.test`, `aov`, `chisq.test`, `p.adjust` | https://www.r-project.org/ |
| R `pwr` | Power analysis (Cohen) | https://cran.r-project.org/package=pwr |
| G*Power | GUI power & sample-size calculator | https://www.psychologie.hhu.de/arbeitsgruppen/allgemeine-psychologie-und-arbeitspsychologie/gpower |
| JASP / jamovi | Open-source GUI for frequentist + Bayesian tests | https://jasp-stats.org/ |
| PyMC | Bayesian modeling / Bayes factors alternative | https://www.pymc.io/ |

## Learning resources

- **An Introduction to Statistical Learning (ISLR / ISLP)** — James, Witten, Hastie, Tibshirani; Ch. 13 covers multiple testing. Free PDF: https://www.statlearning.com/
- **The Elements of Statistical Learning (ESL)** — Hastie, Tibshirani, Friedman. Free PDF: https://hastie.su.domains/ElemStatLearn/
- **OpenIntro Statistics** — free, rigorous intro to inference. https://www.openintro.org/book/os/
- **Seeing Theory** — interactive visual intro to probability & inference. https://seeing-theory.brown.edu/
- **StatQuest (Josh Starmer)** — clear video explanations of t-tests, p-values, FDR. https://statquest.org/video-index/
- **Statistical Rethinking** — McElreath; Bayesian-first perspective on inference (lectures free). https://xcelab.net/rm/
- **Computer Age Statistical Inference** — Efron & Hastie; modern view incl. FDR, bootstrap. Free PDF: https://hastie.su.domains/CASI/
- **scikit-learn / SciPy hypothesis-test guide** — practical recipes. https://docs.scipy.org/doc/scipy/tutorial/stats.html

## Key papers

- Student (W. S. Gosset), "The Probable Error of a Mean," *Biometrika* 6(1):1–25, 1908 — origin of the t-test. https://doi.org/10.1093/biomet/6.1.1
- Neyman, J. & Pearson, E. S., "On the Problem of the Most Efficient Tests of Statistical Hypotheses," *Phil. Trans. R. Soc. A* 231:289–337, 1933 — the Neyman–Pearson lemma. https://doi.org/10.1098/rsta.1933.0009
- Benjamini, Y. & Hochberg, Y., "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing," *JRSS-B* 57(1):289–300, 1995 — FDR. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
- Benjamini, Y. & Yekutieli, D., "The Control of the False Discovery Rate in Multiple Testing under Dependency," *Annals of Statistics* 29(4):1165–1188, 2001. https://doi.org/10.1214/aos/1013699998
- Holm, S., "A Simple Sequentially Rejective Multiple Test Procedure," *Scand. J. Statistics* 6(2):65–70, 1979. https://www.jstor.org/stable/4615733
- Ioannidis, J. P. A., "Why Most Published Research Findings Are False," *PLoS Medicine* 2(8):e124, 2005. https://doi.org/10.1371/journal.pmed.0020124
- Wasserstein, R. L. & Lazar, N. A., "The ASA Statement on p-Values: Context, Process, and Purpose," *The American Statistician* 70(2):129–133, 2016. https://doi.org/10.1080/00031305.2016.1154108

## Cross-references in AIForge

- [Machine Learning](../../Machine_Learning/) — where tests gate feature selection and experiments
- [Model Evaluation](../../Model_Evaluation/) — significance testing of model comparisons, A/B tests
- [Bayesian and Probabilistic ML](../../Bayesian_and_Probabilistic_ML/) — Bayes factors as an alternative to p-values
- [Optimization Algorithms](../../Optimization_Algorithms/) — MLE underlying likelihood-ratio / Wald tests

## Sources

- SciPy stats reference — https://docs.scipy.org/doc/scipy/reference/stats.html
- statsmodels stats & multiple testing — https://www.statsmodels.org/stable/stats.html , https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
- ISLP multiple-testing lab — https://islp.readthedocs.io/en/latest/labs/Ch13-multiple-lab.html
- Benjamini & Hochberg 1995 — https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
- Ioannidis 2005 (PLoS Medicine) — https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0020124
- Student 1908, "The Probable Error of a Mean" (Biometrika) — https://www.york.ac.uk/depts/maths/histstat/student.pdf
- False discovery rate overview — https://en.wikipedia.org/wiki/False_discovery_rate
