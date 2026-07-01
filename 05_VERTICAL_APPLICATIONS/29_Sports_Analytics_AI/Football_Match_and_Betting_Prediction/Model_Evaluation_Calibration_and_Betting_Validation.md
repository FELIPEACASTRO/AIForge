# Model Evaluation, Calibration & Betting Validation

> The part almost everyone gets wrong: how to **evaluate, calibrate and validate** football (soccer / *futebol*) match & betting models honestly — proper scoring rules (RPS, log-loss, Brier), score decompositions, calibration diagnostics, Closing Line Value (CLV), backtest pitfalls, multiple-testing corrections, and time-aware validation — with real, checked sources and free-vs-paid tools, for **research & education only**, current 2024–2026. Sibling pages: [Statistics & Probability Foundations](./Statistics_and_Probability_Foundations.md) · [Odds, Betting Markets & Value Betting](./Odds_Betting_Markets_and_Value_Betting.md) · [Match Prediction Models & Techniques](./Match_Prediction_Models_and_Techniques.md).

> ⚠️ **Research & education only — NOT betting advice, NOT a system, NOT a tip.** Betting markets are **highly efficient**: sharp closing odds track real outcome frequencies almost perfectly. Across ~397,935 Pinnacle football games the closing line matched observed frequencies with **r² ≈ 0.997** ([Trademate Sports](https://tradematesports.medium.com/closing-line-the-most-important-metric-in-sports-trading-58e56cdb4458)); an independent check on ~87,960 Pinnacle 1X2 odds pairs found an almost exact **1:1** expected-vs-observed yield relationship ([football-data.co.uk — closing-line efficiency](https://www.football-data.co.uk/blog/pinnacle_efficiency.php)). **Most bettors lose money over time**, and after the bookmaker margin (*over-round / vig*) positive long-run ROI is rare, small and perishable. A model that "looks great" in a backtest is usually **overfit, leaking, or measured against the wrong metric** — this page is about detecting that. If gambling stops being fun, get help → [§13](#13-responsible-gambling-jogo-responsável--mandatory). 🇧🇷 **CVV 188** · [Autoexclusão SIGAP](https://www.gov.br/pt-br/servicos/plataforma-centralizada-de-autoexclusao-apostas).

---

## 1) The honest premise — evaluation is where models die

A match outcome is **random**; a model only outputs a **probability distribution**. Three facts frame everything below.

1. **Accuracy is the wrong metric.** Because outputs are probabilities, evaluate with **proper scoring rules** — rules an honest forecaster minimises by reporting its *true* beliefs ([Gneiting & Raftery 2007](https://www.tandfonline.com/doi/abs/10.1198/016214506000001437)). A model can be *more accurate* yet *worse-calibrated* (and less profitable) than another.
2. **The market is the benchmark.** De-vigged sharp closing odds are the best cheap estimate of true probability; consistently beating the **closing line (CLV)** is the practical test of skill, and almost nobody does it ([Pinnacle — What is CLV](https://www.pinnacle.com/betting-resources/en/educational/what-is-closing-line-value-clv-in-sports-betting)).
3. **Most "profitable" backtests are artefacts** of leakage, look-ahead, survivorship, ignoring the vig, or **multiple testing** (you tried 500 configs and kept the luckiest). This page's job is to make those artefacts visible before any money — real or notional — is at risk.

---

## 2) Proper scoring rules — log-loss, Brier, RPS

A scoring rule `S(p, y)` grades a forecast `p` against the realised outcome `y`. It is **proper** if truth-telling minimises the expected score, **strictly proper** if truth is the *unique* minimiser ([Gneiting & Raftery 2007, JASA 102(477):359–378](https://www.tandfonline.com/doi/abs/10.1198/016214506000001437); [PDF](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)). Report scores as an **average over matches**; lower is better.

| Rule | Formula (per match) | Properties | Football note | Reference |
|---|---|---|---|---|
| **Log-loss / Ignorance** | `−Σ_i y_i · log p_i` | Strictly proper; **local** (uses only `p` on the realised class); unbounded | Punishes confident wrong calls brutally; one `p→0` on a hit ⇒ ∞ | [Good 1952 / Gneiting & Raftery 2007](https://www.tandfonline.com/doi/abs/10.1198/016214506000001437) |
| **Brier score** | `Σ_i (p_i − y_i)²` (multiclass) | Strictly proper; **non-local**, distance-**insensitive**; bounded `[0,2]` | Decomposes into reliability + resolution + uncertainty (§4) | [Brier 1950, Mon. Weather Rev. 78(1):1–3](https://journals.ametsoc.org/view/journals/mwre/78/1/1520-0493_1950_078_0001_vofeit_2_0_co_2.xml) |
| **Ranked Probability Score (RPS)** | `1/(r−1) · Σ_{i=1}^{r−1} ( Σ_{j=1}^{i}(p_j − o_j) )²` | Strictly proper; **non-local**, distance-**sensitive** (ordinal) | The football default: penalises "home vs away" more than "home vs draw"; `r=3` ⇒ factor `½` | [Epstein 1969](https://journals.ametsoc.org/view/journals/apme/8/6/1520-0450_1969_008_0985_assfpf_2_0_co_2.xml) · [Constantinou & Fenton 2012](https://www.degruyterbrill.com/document/doi/10.1515/1559-0410.1418/html) |

- **Local vs non-local, distance-sensitive vs not** is the axis that matters. Log-loss is *local* (ignores probability mass on non-realised classes); Brier and RPS are *non-local*; only RPS is *distance-sensitive*, which is why it became the football standard for the ordered 1X2 scale.
- **scikit-learn** ships proper rules directly: [`log_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html) and [`brier_score_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html). RPS is a short NumPy one-liner (cumulative-sum of `p − o`, squared, summed) — see [`penaltyblog`](https://github.com/martineastwood/penaltyblog) for football-ready metric helpers.

---

## 3) The RPS-vs-Ignorance debate (don't pick a rule blindly)

RPS is the *de facto* football standard **because** it respects outcome ordering (Epstein 1969; popularised for football by [Constantinou & Fenton 2012, *Solving the Problem of Inadequate Scoring Rules…*, JQAS 8(1)](https://www.degruyterbrill.com/document/doi/10.1515/1559-0410.1418/html); [PDF](http://constantinou.info/downloads/papers/solvingtheproblem.pdf)). But it is **contested**:

- [**Wheatcroft (2021)**, *Evaluating probabilistic forecasts of football matches: the case against the ranked probability score*, JQAS 17(4):273–287](https://www.degruyterbrill.com/document/doi/10.1515/jqas-2019-0089/html) ([preprint arXiv:1908.08980](https://arxiv.org/abs/1908.08980); [LSE PDF](https://researchonline.lse.ac.uk/id/eprint/111494/3/Wheatcroft_evaluating_probabilistic_forecasts_published.pdf)) runs simulations comparing RPS (non-local, distance-sensitive), Brier (non-local, distance-insensitive) and **Ignorance/log-loss** (local, distance-insensitive), and argues distance-sensitivity can *mislead* forecast ranking — often favouring a plain log or Brier score.
- **Takeaway:** there is no universally "correct" rule. Pre-register your primary metric, report **at least two** (e.g. RPS + log-loss), and never switch metric after seeing which one flatters your model — that is metric-shopping, a form of multiple testing (§10).

---

## 4) Score decompositions — reliability, resolution, uncertainty

A single number hides *why* a model scores as it does. **Murphy's decomposition** splits the Brier score into three interpretable parts:

```text
Brier = Reliability − Resolution + Uncertainty
```

- **Reliability (calibration):** mean squared gap between forecast probability and observed frequency within bins — **lower is better** (0 = perfectly calibrated).
- **Resolution:** how far bin frequencies move from the base rate — **higher is better** (sharp, informative forecasts).
- **Uncertainty:** the base-rate variance, fixed by the data, not the model.

Reference: [Murphy 1973, *A New Vector Partition of the Probability Score*, J. Applied Meteorology 12:595–600](https://journals.ametsoc.org/view/journals/apme/12/4/1520-0450_1973_012_0595_anvpot_2_0_co_2.xml). For a modern, football-specific treatment — score decompositions plus reliability and discrimination diagnostics with several binning schemes (fixed thresholds, quantiles, logistic, isotonic) applied to 1X2 forecasts — see [**Foulley (2021)**, *More on verification of probability forecasts for football outcomes…*, arXiv:2106.14345](https://arxiv.org/abs/2106.14345). A rigorous stats-forecasting reference on reliability diagrams and decompositions is [Dimitriadis, Gneiting & Jordan 2021, *Stable reliability diagrams…* (arXiv:2008.03033)](https://arxiv.org/abs/2008.03033).

---

## 5) Calibration diagnostics — reliability diagrams, ECE / MCE / ACE

**Calibration** asks: when the model says 30%, does the event happen ~30% of the time? It is *necessary but not sufficient* — a model can be perfectly calibrated yet useless (always predicting the base rate). Diagnose it visually and numerically.

| Tool | What it shows | Gotcha |
|---|---|---|
| **Reliability diagram** | Predicted prob (x) vs observed frequency (y) per bin; 45° line = perfect | Bin count/edges change the picture; use enough data per bin |
| **Expected Calibration Error (ECE)** | Sample-weighted mean of `|confidence − accuracy|` across bins | Sensitive to binning; can hide offsetting errors |
| **Maximum Calibration Error (MCE)** | Worst bin gap (max, not mean) | Dominated by sparse bins |
| **Adaptive ECE (ACE)** | ECE with equal-count (quantile) bins | More stable when confidence is clustered |

- **scikit-learn:** [`calibration_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html) and [`CalibrationDisplay`](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibrationDisplay.html); guide at [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html).
- **net:cal (`netcal`):** a Python library dedicated to *measuring and mitigating* miscalibration — ships ECE, MCE, ACE and reliability plots ([GitHub: EFS-OpenSource/calibration-framework](https://github.com/EFS-OpenSource/calibration-framework) · [PyPI](https://pypi.org/project/netcal/) · [docs](https://efs-opensource.github.io/calibration-framework/)).
- For 3-class 1X2 forecasts, evaluate calibration **per class** (home/draw/away), not just on the top prediction — the **draw** is where football models are usually worst-calibrated.

---

## 6) Recalibration methods — Platt, temperature, isotonic, beta

If a model discriminates well but is mis-calibrated, **post-hoc recalibration** fits a small map from raw scores to calibrated probabilities on a **held-out** set (never the training or test fold).

| Method | Idea | Params | When it fits | Reference |
|---|---|---|---|---|
| **Platt / sigmoid scaling** | Logistic map `1/(1+exp(A·f+B))` on scores | 2 | Small data; sigmoidal distortion | [Platt 1999](https://www.researchgate.net/publication/2594015_Probabilistic_Outputs_for_Support_Vector_Machines_and_Comparisons_to_Regularized_Likelihood_Methods) · [scikit-learn](https://scikit-learn.org/stable/modules/calibration.html) |
| **Temperature scaling** | Divide logits by a single `T`, then softmax | 1 | Multiclass neural nets; keeps arg-max | [Guo et al. 2017, *On Calibration of Modern Neural Networks*, ICML (arXiv:1706.04599)](https://arxiv.org/abs/1706.04599) |
| **Isotonic regression** | Non-parametric non-decreasing step map | many | ≳1000 samples; flexible | [Zadrozny & Elkan 2002](https://dl.acm.org/doi/10.1145/775047.775151) · [scikit-learn](https://scikit-learn.org/stable/modules/calibration.html) |
| **Beta calibration** | Two-shape/one-location logistic-in-log-odds | 3 | Well-founded default for binary; fixes sigmoid's rigidity | [Kull, Silva Filho & Flach 2017, AISTATS (PMLR v54)](https://proceedings.mlr.press/v54/kull17a.html) · [betacal.github.io](https://betacal.github.io/) |

- **scikit-learn** wraps Platt (`method="sigmoid"`) and isotonic in [`CalibratedClassifierCV`](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html); **net:cal** adds temperature, logistic and **beta** scaling plus histogram binning.
- **Warning:** recalibration on a *time-shuffled* split leaks future information. Fit the calibrator on a **past** slice and validate on a strictly **later** one (see §11). Recalibration cannot rescue a model with no resolution — it only fixes the reliability term of §4.

---

## 7) Closing Line Value (CLV) — the gold-standard edge proof

The **closing line** is the last price before kickoff; it has absorbed sharp money and team news, so it is the market's best estimate of true probability. **CLV** asks whether you consistently bet at odds *better than the close*.

| Concept | Definition | Note |
|---|---|---|
| **CLV** | `your_odds / close_odds − 1` (positive = you beat the close) | Grade against the **de-vigged** close for a clean read |
| **Why it's the benchmark** | Consistent positive CLV is the most reliable signal of a real edge; it is what syndicates track | [Pinnacle — CLV](https://www.pinnacle.com/betting-resources/en/educational/what-is-closing-line-value-clv-in-sports-betting) |
| **Efficiency of the close** | Pinnacle closes ≈ outcomes, r² ≈ 0.997 (~397,935 games) | [Trademate Sports](https://tradematesports.medium.com/closing-line-the-most-important-metric-in-sports-trading-58e56cdb4458) · [football-data.co.uk](https://www.football-data.co.uk/blog/pinnacle_efficiency.php) |

- **The honest test:** a strategy that "profits" against opening/soft odds but **cannot beat the close** has found no edge — it front-ran information the market later priced. Report CLV on **out-of-sample, closing-line-graded** bets, or treat the profit as noise.
- **Operational reality:** even a genuine edge triggers **account limits/closure** at most books — a structural reason retail "systems" don't compound ([Kaunitz et al. 2017, arXiv:1710.02824](https://arxiv.org/abs/1710.02824)).

---

## 8) ROI, variance, risk-of-ruin & Monte-Carlo (understanding, not action)

> ⚠️ These formulas describe the **mechanics of risk**, not a licence to bet. Under an efficient market plus a bookmaker margin, expected value is usually **negative**; Kelly on a *mis-estimated* edge destroys bankrolls.

| Quantity | Formula / method | Why it matters for validation |
|---|---|---|
| **Yield / ROI** | `(returns − stakes) / stakes` | High variance; needs large `N` **and** CLV to mean anything |
| **EV per unit** | `EV = p·o − 1` (decimal odds `o`) | Value only if your `p·o > 1` vs the **de-vigged** market |
| **Single-bet variance** | `Var = p·(1−p)·o²` | Huge → short samples look "profitable" by luck |
| **Kelly fraction** | `f* = (o·p − 1)/(o − 1)` | Growth-optimal **only if `p` is exact**; use fractional `¼–½·f*` |
| **Risk of ruin** | rises steeply with full/over-Kelly and over-estimated `p` | Over-betting a wrong `p` ⇒ near-certain ruin |
| **Monte-Carlo bankroll sim** | simulate `N` bets × many paths → distribution of final bankroll, max drawdown, ruin % | The honest way to *see* variance and confidence bands before trusting a yield |

Report yields with **bootstrap / Monte-Carlo confidence intervals**, not point estimates. Football-Data's own note on Kelly's pitfalls (estimation error, variance, non-independent bets) is a good practitioner warning ([football-data.co.uk/blog/kelly_staking](https://www.football-data.co.uk/blog/kelly_staking.php)).

---

## 9) Backtest-pitfall catalogue

| Pitfall | Why it inflates results | Honest fix |
|---|---|---|
| **Ignoring the vig** | Break-even skill still loses to the margin | Grade vs **de-vigged** fair odds; report net of margin |
| **Grading vs opening/soft odds** | Front-runs info the close later prices | Grade against the **closing line**; report **CLV** (§7) |
| **Look-ahead / leakage** | Uses info unknown at bet time (final lineups, later odds, post-match xG) | Strict **point-in-time** features; time-respecting splits only (§11) |
| **Survivorship / selection** | Dropped void/postponed matches or delisted teams | Keep the full as-scheduled universe |
| **Overfitting** | Many parameters fit historical noise | Few justified parameters; out-of-sample + rolling windows |
| **Multiple testing** | Tried many configs, kept the luckiest | Correct for it: White RC / Hansen SPA / DSR / PBO (§10) |
| **Unrealistic execution** | Assumes free stakes at shown odds | Model stake caps, min odds, limits, slippage, latency |
| **Too few bets** | Short samples look profitable by luck | Thousands of graded bets + confidence bands (§8) |

**Golden rule:** if a strategy cannot show **positive CLV on out-of-sample, closing-line-graded** data, treat the "profit" as noise.

---

## 10) Multiple-testing corrections (the step bettors skip)

Backtesting hundreds of models/features and reporting the best is **data snooping**: the winner's performance is inflated by selection. Borrow the fixes from quantitative finance.

| Method | What it does | Reference |
|---|---|---|
| **White's Reality Check (RC)** | Bootstrap test: is the *best* strategy better than a benchmark once you account for the whole search? | [White 2000, *A Reality Check for Data Snooping*, Econometrica 68(5):1097–1126](https://onlinelibrary.wiley.com/doi/abs/10.1111/1468-0262.00152) |
| **Hansen's SPA test** | Studentised, more powerful RC; robust to poor/irrelevant alternatives | [Hansen 2005, *A Test for Superior Predictive Ability*, JBES 23(4):365–380](https://www.tandfonline.com/doi/abs/10.1198/073500105000000063) |
| **Deflated Sharpe Ratio (DSR)** | Adjusts a Sharpe/edge for **number of trials**, sample length, skew & kurtosis | [Bailey & López de Prado 2014, *The Deflated Sharpe Ratio*, J. Portfolio Mgmt 40(5):94–107](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) |
| **Probability of Backtest Overfitting (PBO)** | Via **CSCV**: probability the selected config underperforms out-of-sample | [Bailey, Borwein, López de Prado & Zhu, *The Probability of Backtest Overfitting*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253) |

If you compared *k* strategies, the honest headline is not "the best returned X%" but "after correcting for *k* trials, is any edge still significant?" — usually the answer is no. Pairwise forecast comparison uses the [Diebold–Mariano test](https://www.tandfonline.com/doi/abs/10.1080/07350015.1995.10524599) (Diebold & Mariano 1995).

---

## 11) Time-aware validation — walk-forward, purged & embargoed CV, CPCV

Football data is a **time series with overlapping information**; plain k-fold CV leaks the future into the past.

| Scheme | Idea | Football fit |
|---|---|---|
| **Walk-forward / expanding window** | Train on ≤ t, test on t+1…; roll forward | The default: mirrors how you'd actually deploy across gameweeks/seasons |
| **Purged K-fold CV** | Drop training samples whose label window **overlaps** the test window | Removes leakage from rolling/form features spanning matches |
| **Embargo** | Also drop a buffer of samples **right after** each test block | Kills serial-correlation leakage across the fold boundary |
| **Combinatorial Purged CV (CPCV)** | Many purged+embargoed train/test recombinations → a **distribution** of out-of-sample scores | Robust estimate + feeds PBO/DSR (§10) |

Reference for purging, embargoing and CPCV: **López de Prado (2018), *Advances in Financial Machine Learning*, Wiley** ([purged cross-validation overview](https://en.wikipedia.org/wiki/Purged_cross-validation)). Football walk-forward walkthroughs: [Betfair Data Scientists — backtesting tutorial](https://betfair-datascientists.github.io/tutorials/backtestingRatingsTutorial/). Season-aware splitting matters: never let a model "know" a later gameweek's results, injuries, or transfer-window changes when scoring an earlier one.

---

## 12) Data, APIs & tools to practise validation (free vs paid)

| Resource | What / scope | Free? | Link |
|---|---|---|---|
| **football-data.co.uk** | 30+ yrs results **+ opening/closing odds** (many leagues incl. 🇧🇷) — build CLV/backtests | ✅ Free CSV | [football-data.co.uk](https://www.football-data.co.uk/data.php) |
| **2017 Soccer Prediction Challenge / Open International Soccer DB** | 216,743 matches, 52 leagues, 35 countries; benchmark 206-match test set (CC0) | ✅ Free (OSF) | [OSF osf.io/ftuva](https://osf.io/ftuva/) · [dataset paper (ML 108, DOI 10.1007/s10994-018-5726-0)](https://link.springer.com/article/10.1007/s10994-018-5726-0) · [challenge results (Berrar, Lopes & Dubitzky, ML 108:97–126)](https://link.springer.com/article/10.1007/s10994-018-5747-8) |
| **FPL API — `bootstrap-static`** | Players/teams/gameweeks JSON (no auth) — clean data for calibration practice | ✅ Free | [fantasy.premierleague.com/api/bootstrap-static/](https://fantasy.premierleague.com/api/bootstrap-static/) · [endpoint guide](https://medium.com/@frenzelts/fantasy-premier-league-api-endpoints-a-detailed-guide-acbd5598eb19) |
| **Betfair Exchange API** | Back/lay traded prices = market-implied probabilities; JSON-RPC | ✅ Free dev use (funded acct for live) | [developer.betfair.com/exchange-api](https://developer.betfair.com/exchange-api/) · [samples](https://github.com/betfair-datascientists/API) |
| **StatsBomb Open Data** | Free event data (JSON), selected competitions — feature building | ✅ Free (research) | [github.com/statsbomb/open-data](https://github.com/statsbomb/open-data) |
| **Metrica Sports sample-data** | Anonymised tracking + event sample games (EPTS/CSV/JSON) | ✅ Free | [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) |
| **SkillCorner Open Data** | 10 broadcast-tracking matches (2024/25 A-League) + derived events, w/ PySport | ✅ Free | [github.com/SkillCorner/opendata](https://github.com/SkillCorner/opendata) |
| **scikit-learn** | `log_loss`, `brier_score_loss`, `calibration_curve`, `CalibratedClassifierCV` | ✅ Open source | [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html) |
| **net:cal (`netcal`)** | ECE/MCE/ACE + temperature/logistic/beta scaling + reliability plots | ✅ Open source | [GitHub](https://github.com/EFS-OpenSource/calibration-framework) · [PyPI](https://pypi.org/project/netcal/) |
| **penaltyblog** (Python) | Poisson/Dixon–Coles, de-vig, ratings, evaluation helpers | ✅ Open source | [GitHub](https://github.com/martineastwood/penaltyblog) · [PyPI](https://pypi.org/project/penaltyblog/) |

🇧🇷 **Brazil note:** Brasileirão results + odds live in football-data.co.uk (`BRA.csv`) for CLV/backtest practice; the OSF Soccer DB includes Brazilian leagues. See the [Global Datasets & Data APIs sibling page](./Global_Datasets_and_Data_APIs.md) for full Brazilian coverage.

---

## Canonical references (checked)

| # | Work | Contribution | Venue / year |
|---|---|---|---|
| 1 | **Gneiting & Raftery** — *Strictly Proper Scoring Rules, Prediction, and Estimation* | Theory of (strictly) proper scoring rules | [JASA 102(477):359–378, 2007](https://www.tandfonline.com/doi/abs/10.1198/016214506000001437) |
| 2 | **Brier** — *Verification of forecasts expressed in terms of probability* | Brier score | [Mon. Weather Rev. 78(1):1–3, 1950](https://journals.ametsoc.org/view/journals/mwre/78/1/1520-0493_1950_078_0001_vofeit_2_0_co_2.xml) |
| 3 | **Epstein** — *A Scoring System for Probability Forecasts of Ranked Categories* | Origin of the RPS | [J. Appl. Meteor. 8:985–987, 1969](https://journals.ametsoc.org/view/journals/apme/8/6/1520-0450_1969_008_0985_assfpf_2_0_co_2.xml) |
| 4 | **Constantinou & Fenton** — *Solving the Problem of Inadequate Scoring Rules…* | RPS for football | [JQAS 8(1), 2012](https://www.degruyterbrill.com/document/doi/10.1515/1559-0410.1418/html) |
| 5 | **Wheatcroft** — *…the case against the ranked probability score* | Critique of RPS | [JQAS 17(4):273–287, 2021](https://www.degruyterbrill.com/document/doi/10.1515/jqas-2019-0089/html) · [arXiv:1908.08980](https://arxiv.org/abs/1908.08980) |
| 6 | **Murphy** — *A New Vector Partition of the Probability Score* | Reliability/resolution/uncertainty | [J. Appl. Meteor. 12:595–600, 1973](https://journals.ametsoc.org/view/journals/apme/12/4/1520-0450_1973_012_0595_anvpot_2_0_co_2.xml) |
| 7 | **Foulley** — *More on verification of probability forecasts for football outcomes…* | Football score decompositions | [arXiv:2106.14345, 2021](https://arxiv.org/abs/2106.14345) |
| 8 | **Guo, Pleiss, Sun & Weinberger** — *On Calibration of Modern Neural Networks* | Temperature scaling | [ICML 2017 / arXiv:1706.04599](https://arxiv.org/abs/1706.04599) |
| 9 | **Kull, Silva Filho & Flach** — *Beta calibration* | Beta recalibration | [AISTATS 2017, PMLR v54](https://proceedings.mlr.press/v54/kull17a.html) |
| 10 | **White** — *A Reality Check for Data Snooping* | Multiple-testing correction | [Econometrica 68(5):1097–1126, 2000](https://onlinelibrary.wiley.com/doi/abs/10.1111/1468-0262.00152) |
| 11 | **Hansen** — *A Test for Superior Predictive Ability* | Studentised SPA test | [JBES 23(4):365–380, 2005](https://www.tandfonline.com/doi/abs/10.1198/073500105000000063) |
| 12 | **Bailey & López de Prado** — *The Deflated Sharpe Ratio* | Selection-bias–corrected Sharpe | [J. Portfolio Mgmt 40(5):94–107, 2014](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) |
| 13 | **Bailey, Borwein, López de Prado & Zhu** — *The Probability of Backtest Overfitting* | PBO via CSCV | [J. Comp. Finance / SSRN 2326253](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253) |
| 14 | **López de Prado** — *Advances in Financial Machine Learning* | Purged/embargoed CV, CPCV | [Wiley, 2018](https://en.wikipedia.org/wiki/Purged_cross-validation) |

---

## 13) Responsible gambling (jogo responsável) — mandatory

Betting is entertainment with a **negative expected value** for almost everyone; everything above is for **understanding and research**, not to encourage wagering. Set limits, never chase losses (*não persiga o prejuízo*), never bet borrowed money. If it stops being fun, stop.

| Region | Resource | Contact / tool |
|---|---|---|
| 🌍 Global | [Gambling Therapy](https://www.gamblingtherapy.org/) (Gordon Moody) | Free multilingual online chat & forum |
| 🇬🇧 UK | [GambleAware](https://www.gambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) | National Gambling Helpline **0808 8020 133** (24/7) |
| 🇺🇸 USA | [NCPG](https://www.ncpgambling.org/) | **1-800-MY-RESET** — call/text/chat · [1800myreset.org](https://www.1800myreset.org/) |
| 🇧🇷 Brasil | [Jogo Responsável — SPA/Ministério da Fazenda](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel) · [Jogadores Anônimos](https://jogadoresanonimos.com.br/) | Apoio emocional **CVV 188** (24h) |
| 🇧🇷 Brasil — autoexclusão | [Plataforma centralizada de autoexclusão (SIGAP)](https://www.gov.br/pt-br/servicos/plataforma-centralizada-de-autoexclusao-apostas) | Bloqueia o CPF em **todas** as casas autorizadas |

Under the Brazilian **Lei das Bets** (Lei nº 14.790/2023), regulated fixed-odds operation began **1 Jan 2025** under the `.bet.br` domain, with mandatory deposit/loss/time limits, reality checks and self-exclusion ([Lei das Bets — Wikipédia](https://pt.wikipedia.org/wiki/Lei_das_Bets)).

---

## Related in AIForge
- [Statistics & Probability Foundations](./Statistics_and_Probability_Foundations.md) · [Odds, Betting Markets & Value Betting](./Odds_Betting_Markets_and_Value_Betting.md) · [Match Prediction Models & Techniques](./Match_Prediction_Models_and_Techniques.md) · [Betting Exchanges, Trading & Microstructure](./Betting_Exchanges_Trading_and_Microstructure.md) · [Fantasy Football & FPL Analytics](./Fantasy_Football_and_FPL_Analytics.md) · Parent vertical: [`../`](../) (Sports Analytics AI)

**Sources:** tandfonline.com (Gneiting & Raftery 2007 · Hansen 2005) · journals.ametsoc.org (Brier 1950 · Epstein 1969 · Murphy 1973) · degruyterbrill.com (Constantinou & Fenton 2012 · Wheatcroft 2021) · arxiv.org (Wheatcroft 1908.08980 · Foulley 2106.14345 · Guo 1706.04599 · Kaunitz 1710.02824 · Dimitriadis 2008.03033) · researchonline.lse.ac.uk (Wheatcroft PDF) · proceedings.mlr.press (Kull et al. 2017) · betacal.github.io · scikit-learn.org/stable/modules/calibration.html · github.com/EFS-OpenSource/calibration-framework · pypi.org/project/netcal · github.com/martineastwood/penaltyblog · onlinelibrary.wiley.com (White 2000) · davidhbailey.com (Deflated Sharpe) · papers.ssrn.com (PBO 2326253) · en.wikipedia.org/wiki/Purged_cross-validation (López de Prado 2018) · tandfonline.com (Diebold–Mariano 1995) · pinnacle.com (CLV) · tradematesports.medium.com · football-data.co.uk (pinnacle_efficiency · kelly_staking · data.php) · osf.io/ftuva · link.springer.com (10.1007/s10994-018-5726-0 · 10.1007/s10994-018-5747-8) · fantasy.premierleague.com/api/bootstrap-static · developer.betfair.com/exchange-api · github.com/betfair-datascientists/API · github.com/statsbomb/open-data · github.com/metrica-sports/sample-data · github.com/SkillCorner/opendata · gamblingtherapy.org · gambleaware.org · gamcare.org.uk · ncpgambling.org · 1800myreset.org · gov.br/fazenda (SPA jogo responsável) · gov.br (autoexclusão SIGAP) · jogadoresanonimos.com.br · pt.wikipedia.org/wiki/Lei_das_Bets

**Keywords:** model evaluation, avaliação de modelos, calibration, calibração, proper scoring rule, regra de pontuação própria, strictly proper, log-loss, cross-entropy, Brier score, ranked probability score, RPS, ignorance score, Gneiting Raftery, Epstein 1969, Constantinou Fenton, Wheatcroft, Murphy decomposition, reliability resolution uncertainty, score decomposition, Foulley, reliability diagram, diagrama de confiabilidade, expected calibration error, ECE, MCE, ACE, Platt scaling, temperature scaling, isotonic regression, beta calibration, scikit-learn, netcal, closing line value, CLV, linha de fechamento, market efficiency, eficiência de mercado, Pinnacle, backtest overfitting, sobreajuste, data snooping, White reality check, Hansen SPA, deflated Sharpe ratio, probability of backtest overfitting, PBO, CSCV, walk-forward validation, purged cross-validation, embargo, CPCV, Lopez de Prado, time series validation, validação temporal, risk of ruin, risco de ruína, Monte Carlo bankroll, Kelly criterion, ROI, yield, expected value, valor esperado, football prediction, futebol previsão, soccer prediction challenge, Open International Soccer Database, FPL API, Betfair, StatsBomb, Metrica, SkillCorner, penaltyblog, responsible gambling, jogo responsável, CVV 188, autoexclusão, Lei das Bets.
