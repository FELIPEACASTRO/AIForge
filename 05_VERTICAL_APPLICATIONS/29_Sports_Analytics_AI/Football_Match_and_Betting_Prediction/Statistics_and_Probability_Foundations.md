# Statistics & Probability Foundations

> The mathematical core of football (soccer / futebol) match forecasting — the **Poisson goals process**, **team-rating systems**, **Bayesian inference**, **odds↔probability conversion**, **proper scoring rules**, and **betting math (matemática de apostas)** — every formula tied to a real, checked reference. **Research & education only (pesquisa e educação); this is not betting advice (não é dica de aposta).** Current 2024–2026.

> ⚠️ **Read this first.** Football betting markets are **highly efficient (altamente eficientes)**: sharp books (e.g. Pinnacle) and the **closing line (linha de fechamento)** already encode almost all public information, so a positive long-run edge is **rare and hard to sustain**. **Most bettors lose money.** The math below explains *how the market is priced and evaluated* — it is **not** a system to make money. If gambling is a problem, get help: [BeGambleAware](https://www.begambleaware.org/) · [GamCare / National Gambling Helpline 0808 8020 133](https://www.gamcare.org.uk/) · [Gambling Therapy (multilingual)](https://www.gamblingtherapy.org/) · 🇧🇷 [Jogo Responsável — Secretaria de Prêmios e Apostas / MF](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel) · 🇧🇷 CVV **188**.

**Sibling pages:** [Open-Source Tools & Libraries](./Open_Source_Tools_and_Libraries.md) · [Global Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md) · [Kaggle Datasets](./Kaggle_Football_Datasets_and_Competitions.md) · [Innovative Models & Deep Learning](./Innovative_Models_and_Deep_Learning.md).

> **Notation.** Home/away expected goals `λ` (home) and `μ` (away); scorelines `(x, y)`; forecast probabilities `p`; decimal odds `o`; bankroll `B`. Formulas use inline `code`; verify each against its cited source before use.

---

## 1) The honest premise (a premissa honesta)

Modelling exists because a match outcome is **random** — even a "correct" model only outputs a **probability distribution**, never a certainty. Two facts frame everything below:

1. **The market is the benchmark.** Devigged odds from a sharp book are the best cheap estimate of true probability; beating the **closing line value (CLV)** consistently is the practical test of skill, and almost nobody does it. See [Pinnacle — beating the bookies / CLV](https://www.pinnacle.com/betting-resources/en/betting-strategy/all-you-need-to-know-to-beat-the-bookies/mcqj75njb6l6rjzp) and the market-efficiency literature.
2. **A good model ≠ profit.** After the bookmaker margin (**overround / vig / suja**), a model must be not just *accurate* but *better-calibrated than the market on the specific bets you place*. Constantinou & Fenton's pi-ratings study is notable precisely because demonstrating profit vs. market odds is so unusual ([JQAS 2013](https://ideas.repec.org/a/bpj/jqsprt/v9y2013i1p37-50n4.html)).

---

## 2) The goals-generating process: the Poisson family (o processo de gols)

Goals are rare, discrete, roughly independent-in-time events → the natural baseline is a **Poisson process**. `P(X = k) = e^(−λ) · λ^k / k!`, with `E[X] = Var[X] = λ`.

**Maher (1982)** introduced team **attack/defence** parameters; the standard log-linear means are `log λ = μ0 + home + att_home + def_away` and `log μ = μ0 + att_away + def_home` ([Statistica Neerlandica 36:109–118](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9574.1982.tb00782.x); [PDF](http://www.90minut.pl/misc/maher.pdf)).

| Model | Core formula / idea | What it fixes | Reference |
|---|---|---|---|
| **Independent Poisson** | `P(x,y)=Pois(x;λ)·Pois(y;μ)` | Baseline; assumes home & away goals independent | [Maher 1982](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9574.1982.tb00782.x) |
| **Bivariate Poisson** | `X=W1+W3, Y=W2+W3`, `W3~Pois(λ3)` → `Cov(X,Y)=λ3≥0` | Adds positive score **correlation** | [Karlis & Ntzoufras 2003, JRSS-D 52:381–393](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/1467-9884.00366) · [PDF](http://www2.stat-athens.aueb.gr/~karlis/Bivariate%20Poisson%20Regression.pdf) |
| **Dixon–Coles** | Independent Poisson × low-score correction `τ(x,y)` + time decay | Under-counts of **draws / 0-0, 1-1** | [Dixon & Coles 1997, JRSS-C 46(2):265–280](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/1467-9876.00065) |
| **Diagonal-inflated bivariate Poisson** | Mixture inflating `(0,0),(1,1),(2,2)…` | Excess **draws** + overdispersion | [Karlis & Ntzoufras 2003](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/1467-9884.00366) |
| **Zero-inflated Poisson (ZIP/ZINB)** | Mixture: point mass at 0 + Poisson/NB | Excess **0-goal** counts | [SAS ZIP/ZINB ref](https://support.sas.com/resources/papers/sgf2008/countreg.pdf) |
| **Negative binomial** | `Var = μ + μ²/r` (r = dispersion) | **Overdispersion** (variance > mean) | [Pollard 1985 — Goal-scoring & the NB distribution](https://www.researchgate.net/publication/270311475_699_Goal-Scoring_and_the_Negative_Binomial_Distribution) |
| **Skellam (goal difference)** | `D=X−Y`, `P(D=z)=e^(−(λ+μ))·(λ/μ)^(z/2)·I_|z|(2√(λμ))` | Models **margin** directly; captures dependence | [Karlis & Ntzoufras 2009, IMA JMM 20(2):133–145](https://academic.oup.com/imaman/article-abstract/20/2/133/716512) |
| **Bivariate Weibull count** | Weibull inter-arrival → flexible dispersion | Non-Poisson goal timing | [Boshnakov, Kharrat & McHale 2017, IJF 33(2):458–466](https://www.sciencedirect.com/science/article/abs/pii/S0169207017300018) |

Note: the shared-component bivariate Poisson only models **positive** correlation; real football correlation is small and sometimes negative, which is why Dixon–Coles' local correction (below) and the Skellam approach remain popular.

---

## 3) Dixon–Coles: low-score correction (τ) & time-decay weighting

Dixon & Coles keep independent Poisson marginals but multiply the joint by a **local correction** `τ` on the four lowest scorelines, governed by one parameter `ρ`:

```text
τ(0,0) = 1 − λ·μ·ρ      τ(0,1) = 1 + λ·ρ
τ(1,0) = 1 + μ·ρ        τ(1,1) = 1 − ρ
τ(x,y) = 1              otherwise
```

`ρ < 0` lifts 0-0 and 1-1 while trimming 1-0/0-1, matching observed draw frequencies. **Time decay:** recent matches matter more, so the log-likelihood is weighted `φ(t) = exp(−ξ·t)` where `t` is match age (days/weeks) and `ξ` the decay rate — fit by maximum likelihood. See the [Dixon–Coles paper](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/1467-9876.00065), a worked derivation at [opisthokonta.net](https://opisthokonta.net/?p=890) and [dashee87's time-weighting post](https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/); reference implementations in [`penaltyblog`](https://github.com/martineastwood/penaltyblog) and [`regista`](https://torvaney.github.io/regista/reference/dixoncoles.html).

---

## 4) From a goals model to market probabilities (a matriz de placares)

Any goals model yields a **score matrix (matriz de placares)** `M[x,y] = P(home = x, away = y)` for `x, y = 0…n` (n≈10 captures ≈100%). Every football market is a **sum over cells**:

| Market (mercado) | Probability from the score matrix | Note |
|---|---|---|
| **1X2** (home/draw/away) | `P(home)=Σ_{x>y} M`, `P(draw)=Σ_{x=y} M`, `P(away)=Σ_{x<y} M` | The classic three-way (três resultados) |
| **Over/Under 2.5** (mais/menos) | `P(Over 2.5)=Σ_{x+y≥3} M` | Any line by changing the threshold |
| **Correct score** (placar exato) | `M[x,y]` (single cell) | Highest-margin, hardest market |
| **Both Teams To Score** (ambas marcam) | `P(BTTS)=Σ_{x≥1, y≥1} M` | = `1 − P(x=0) − P(y=0) + M[0,0]` |
| **Asian handicap** (handicap asiático) | Σ cells where `(x − y + h) > 0`; **push (anula)** when `x − y + h = 0`; split lines (e.g. −0.25) = half-stake each side | Quarter lines split the stake |
| **Draw No Bet / DNB** | Renormalise 1X2 removing the draw: `P'(home)=P(home)/(P(home)+P(away))` | Stake returned on draw |

Marginal totals (over/under) can also be read straight from the **Skellam** distribution of `D = x − y` for handicaps and DNB, avoiding the full matrix.

---

## 5) Rating math (classificações / ratings)

Ratings compress team strength into a number updated after each match — a cheap, strong baseline that feeds a probability via a link function.

| System | Update / expected-score formula | Note | Reference |
|---|---|---|---|
| **Elo** | `We = 1 / (1 + 10^(−dr/400))`; `Rn = Ro + K·G·(W − We)` | `dr` = rating diff (+100 home); `W∈{1,0.5,0} `; logistic expected score | [World Football Elo — eloratings.net](https://www.eloratings.net/about) · [Wikipedia](https://en.wikipedia.org/wiki/World_Football_Elo_Ratings) |
| **Elo K-factor** | `K` = 60 (World Cup), 50 (continental), 40 (qualifiers), 30 (other), 20 (friendly) | Higher `K` = faster adaptation, noisier | [eloratings.net](https://www.eloratings.net/about) |
| **Elo goal multiplier** | `G=1` (≤1-goal), `1.5` (2), `(11+N)/8` (`N≥3`) | Rewards margin of victory (goleada) | [Wikipedia](https://en.wikipedia.org/wiki/World_Football_Elo_Ratings) |
| **pi-ratings** | Separate **home/away** ratings; error `e=|predicted GD − actual GD|`; update via diminishing `ψ(e)=c·log₁₀(1+e)`, learn-rates `λ` (own) & `γ` (cross-venue) | Beat Elo & showed profit vs odds over 5 EPL seasons | [Constantinou & Fenton, JQAS 9(1):37–50, 2013](https://ideas.repec.org/a/bpj/jqsprt/v9y2013i1p37-50n4.html) · [PDF](http://www.constantinou.info/downloads/papers/pi-ratings.pdf) · [`piratings` R pkg](https://cran.r-project.org/web/packages/piratings/vignettes/README.html) |
| **TrueSkill** | Skill `s~N(μ,σ²)`, performance `~N(s,β²)`; Bayesian update by **message passing on a factor graph**; models draws & teams | Generalises Elo; tracks **uncertainty** | [Herbrich, Minka & Graepel, NeurIPS 2006](https://papers.nips.cc/paper/3079-trueskilltm-a-bayesian-skill-rating-system) · [TrueSkill 2](https://www.microsoft.com/en-us/research/publication/trueskill-2-improved-bayesian-skill-rating-system/) |

Elo/Glicko/TrueSkill output a **win/draw** probability directly; to reach 1X2 you still need a draw model (e.g. a rating gap → draw-probability curve) or a goals model conditioned on the rating difference.

---

## 6) Inference: MLE, Bayesian, hierarchical shrinkage (inferência)

- **Maximum likelihood (MLE).** Dixon–Coles fit attack/defence/home/`ρ`/`ξ` by maximising the (time-weighted) log-likelihood `ℓ(θ)=Σ_t φ(t)·log P(x_t,y_t | θ)` — a standard numerical optimisation ([Dixon & Coles 1997](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/1467-9876.00065)).
- **Bayesian inference.** Put **priors** on parameters, sample the **posterior** with **MCMC** (Gibbs/HMC/NUTS). Baio & Blangiardo's **hierarchical model** shares strength across teams via **shrinkage (encolhimento)** toward league means — and warns about **over-shrinkage** of extreme teams, fixed with a mixture prior ([Baio & Blangiardo, J. Applied Statistics 37(2):253–264, 2010](https://www.tandfonline.com/doi/full/10.1080/02664760802684177); [author page + code](https://gianluca.statistica.it/research/football/)).
- **Constraint.** Attack/defence are identifiable only up to a constant, so impose `Σ att = 0` (or a sum-to-zero / corner constraint).
- **Tooling.** MCMC via Stan / PyMC / OpenBUGS; football-specific: [`footBayes`](https://github.com/LeoEgidi/footBayes) (R package — Bayesian & MLE double-Poisson, bivariate-Poisson, Skellam & Student-t models via Stan), plus PyMC tutorials on [penaltyblog](https://pena.lt/y/2021/08/25/predicting-football-results-using-bayesian-statistics-with-python-and-pymc3/).

---

## 7) Odds ↔ probability: overround, vig & margin removal (devig / remoção da margem)

Decimal odds `o` imply a raw probability `1/o`. Across all outcomes the **booksum** `Σ 1/o_i = 1 + margin > 1`; the excess is the **overround / vig / margem (suja)**. To recover fair probabilities you must **remove the margin** — and the method matters, because the vig is *not* spread evenly (favourite–longshot bias).

| Method | Fair probability `p_i` | Note | Reference |
|---|---|---|---|
| **Basic / multiplicative** | `p_i = (1/o_i) / Σ(1/o_j)` | Simple; spreads vig proportionally, ignores longshot bias | [Pinnacle devig guide](https://www.pinnacleoddsdropper.com/guides/how-to-devig-pinnacle-s-odds-for-betting-on-soft-books) |
| **Additive** | `p_i = 1/o_i − margin/n` | Equal absolute vig per outcome | [WinnerOdds true-odds](https://winnerodds.com/true-odds-calculator/) |
| **Power** | find `n` s.t. `Σ (1/o_i)^n = 1`; `p_i=(1/o_i)^n` | Corrects **favourite–longshot bias** | [Wisdom-of-the-Crowd PDF](https://www.football-data.co.uk/The_Wisdom_of_the_Crowd_updated.pdf) |
| **Shin** | solve insider-fraction `z`: `p_i = [√(z² + 4(1−z)·(1/o_i)²/Σ(1/o_j)) − z] / (2(1−z))` | Models **insider trading**; often most accurate across sports | [Shin 1993](https://en.wikipedia.org/wiki/Favourite-longshot_bias) · [`mberk/shin` Python](https://github.com/mberk/shin) |
| **Wisdom of the Crowd (Pinnacle)** | devigged **closing** odds of a sharp book = best public proxy for true `p` | Crowd of sharp money ≈ efficient | [football-data.co.uk WotC](https://www.football-data.co.uk/The_Wisdom_of_the_Crowd_updated.pdf) |

Empirically, Shin- and power-normalised probabilities beat naive division in several team sports — margins rise with the number of outcomes, exactly as Shin's insider model predicts.

---

## 8) Evaluation & calibration (avaliação e calibração)

Because outputs are **probabilities**, evaluate with **proper scoring rules (regras de pontuação próprias)** — rules minimised by honest probabilities — not by accuracy.

| Metric | Formula (per match, then average) | Football note | Reference |
|---|---|---|---|
| **Log-loss (cross-entropy)** | `−Σ_i y_i · log p_i` | Punishes confident wrong calls harshly; unbounded | standard |
| **Brier score** | `Σ_i (p_i − y_i)²` (multiclass) | Bounded [0,2]; decomposes into calibration + refinement | [Brier 1950] |
| **Ranked Probability Score (RPS)** | `RPS = 1/(r−1) · Σ_{i=1}^{r−1} ( Σ_{j=1}^{i}(p_j − o_j) )²` | **The football standard**: ordinal-aware (home closer to draw than to away); `r=3` → factor `1/2` | [Constantinou & Fenton, JQAS 8(1), 2012](https://www.degruyter.com/document/doi/10.1515/1559-0410.1418/html) · [PDF](http://constantinou.info/downloads/papers/solvingtheproblem.pdf) |
| **Reliability diagram** | plot predicted vs. observed frequency in bins | Visual **calibration** check (45° line = perfect) | standard |
| **ROC / AUC** | rank-order discrimination | **Limited**: ignores calibration & the draw; poor for 3-way betting | — |
| **ROI / yield** | `(returns − stakes) / stakes` | Business metric, but **high variance**; needs large `N` + CLV to be meaningful | [CLV explainer](https://www.pinnacleoddsdropper.com/blog/closing-line-value) |

RPS is the de-facto standard *because* it is sensitive to the ordering of outcomes (Epstein 1969; popularised for football by Constantinou & Fenton). It is **contested**: Wheatcroft argues distance-sensitivity can mislead and that a plain (multiclass Brier / log) score is preferable — read [*"the case against the RPS"* (Wheatcroft, arXiv:1908.08980)](https://arxiv.org/abs/1908.08980) before committing to one rule. **Accuracy is not enough**: a model can be more accurate yet worse-calibrated (and thus less profitable) than another.

---

## 9) Betting math (matemática de apostas) — for understanding, not action

> ⚠️ These formulas describe the **mechanics of risk**, not a licence to bet. Under an efficient market and a bookmaker margin, **expected value is usually negative**; Kelly on a *mis-estimated* edge destroys bankrolls.

| Concept | Formula | Note |
|---|---|---|
| **Expected value (EV / valor esperado)** | `EV = p·(o−1) − (1−p) = p·o − 1` (unit stake, decimal odds `o`) | **Value bet** only if `p·o > 1`, i.e. your `p` beats the market's `1/o` |
| **Variance of a single bet** | `Var = p·(1−p)·o²` | Single-match variance is huge → long horizons needed |
| **Kelly criterion** | `f* = (b·p − q)/b = (o·p − 1)/(o − 1)`, `b=o−1`, `q=1−p` | Fraction of bankroll maximising **long-run log-growth** ([Kelly 1956, Bell System Tech. J.](https://en.wikipedia.org/wiki/Kelly_criterion)) |
| **Log-growth rate** | `g(f) = p·ln(1+f·b) + q·ln(1−f)`; maximise → Kelly | Kelly = argmax of expected `log` wealth |
| **Fractional Kelly** | stake `c·f*` with `c ≈ 0.25–0.5` | Cuts variance & drawdowns; robust to `p` **estimation error** (the real-world killer) |
| **Risk of ruin** | ↑ steeply with full/over-Kelly and with over-estimated edge | Over-betting a wrong `p` ⇒ near-certain ruin |
| **Monte-Carlo bankroll sim** | simulate `N` bets ×many paths → distribution of `B_final`, drawdown, ruin % | The honest way to *see* variance before risking anything |
| **Favourite–longshot bias** | longshots systematically **over-bet** (returns below implied); favourites slightly under-bet | [Wikipedia](https://en.wikipedia.org/wiki/Favourite-longshot_bias); Shin explains it via insiders |

Key intuition: Kelly's edge term `o·p − 1` **is exactly the EV**; if your `p` merely equals the devigged market `p`, EV ≤ 0 after vig and Kelly says **bet nothing**. Fractional Kelly exists because your `p` is an *estimate* with error, and full Kelly assumes it is exact.

---

## 10) Data, APIs & libraries to practise the math (free vs paid)

| Resource | What / scope | Free? | Link |
|---|---|---|---|
| **football-data.co.uk** | 30+ yrs results **+ closing/opening odds** (many leagues, incl. 🇧🇷) | ✅ Free (CSV) | [football-data.co.uk](https://www.football-data.co.uk/) |
| **StatsBomb Open Data** | Event data (JSON), selected competitions | ✅ Free (research) | [github.com/statsbomb/open-data](https://github.com/statsbomb/open-data) |
| **Understat** | xG / shot data, top-5 leagues (+ RFPL) | ✅ Free (scrape) | [understat.com](https://understat.com/) |
| **football-data.org** | Results API, 12 comps incl. **Brasileirão Série A** | ✅ Free tier (10 req/min) · 💳 paid | [football-data.org/pricing](https://www.football-data.org/pricing) |
| **API-Football** | 1,200+ leagues, odds, live | 💳 Free 100 req/day · paid from ~$19/mo | [api-football.com/pricing](https://www.api-football.com/pricing) |
| **penaltyblog** (Python) | Poisson, Bivariate, **Dixon–Coles**, ratings, devig | ✅ Open source | [PyPI](https://pypi.org/project/penaltyblog/) · [GitHub](https://github.com/martineastwood/penaltyblog) |
| **footBayes** (R) | Bayesian & MLE double/bivariate-Poisson, Skellam, Student-t (Stan) | ✅ Open source | [GitHub](https://github.com/LeoEgidi/footBayes) · [CRAN](https://cran.r-project.org/web/packages/footBayes/) |
| **mberk/shin** (Python) | Shin margin-removal implementation | ✅ Open source | [github.com/mberk/shin](https://github.com/mberk/shin) |
| **piratings** (R) | pi-ratings recurrence | ✅ Open source | [CRAN](https://cran.r-project.org/web/packages/piratings/vignettes/README.html) |

🇧🇷 **Brazil note (nota Brasil):** Brasileirão results/odds are in football-data.co.uk (`BRA.csv`) and football-data.org's free tier; xG for the top-5 leagues (not Brazil) via Understat/FBref. See the [Datasets & APIs sibling page](./Global_Datasets_and_Data_APIs.md) for the full Brazilian coverage list.

---

## 11) Canonical references (checked, most-cited first)

| # | Work | Contribution | Venue / year |
|---|---|---|---|
| 1 | **Maher** — *Modelling association football scores* | Attack/defence Poisson | [Statistica Neerlandica 36:109–118, 1982](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9574.1982.tb00782.x) |
| 2 | **Dixon & Coles** — *Modelling … and inefficiencies in the football betting market* | `τ` low-score correction + time decay | [JRSS-C 46(2):265–280, 1997](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/1467-9876.00065) |
| 3 | **Karlis & Ntzoufras** — *Analysis of sports data by using bivariate Poisson models* | Bivariate & diagonal-inflated Poisson | [JRSS-D 52:381–393, 2003](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/1467-9884.00366) |
| 4 | **Karlis & Ntzoufras** — *Bayesian modelling … Skellam's distribution* | Goal-difference (Skellam) model | [IMA JMM 20(2):133–145, 2009](https://academic.oup.com/imaman/article-abstract/20/2/133/716512) |
| 5 | **Baio & Blangiardo** — *Bayesian hierarchical model for … football results* | Hierarchical shrinkage + MCMC | [J. Appl. Stat. 37(2):253–264, 2010](https://www.tandfonline.com/doi/full/10.1080/02664760802684177) |
| 6 | **Constantinou & Fenton** — *Solving the problem of inadequate scoring rules* | **RPS** for football | [JQAS 8(1), 2012](https://www.degruyter.com/document/doi/10.1515/1559-0410.1418/html) |
| 7 | **Constantinou & Fenton** — *…dynamic ratings based on relative discrepancies…* | **pi-ratings** | [JQAS 9(1):37–50, 2013](https://ideas.repec.org/a/bpj/jqsprt/v9y2013i1p37-50n4.html) |
| 8 | **Herbrich, Minka & Graepel** — *TrueSkill™* | Bayesian skill rating | [NeurIPS 2006](https://papers.nips.cc/paper/3079-trueskilltm-a-bayesian-skill-rating-system) |
| 9 | **Boshnakov, Kharrat & McHale** — *A bivariate Weibull count model…* | Non-Poisson goal timing | [IJF 33(2):458–466, 2017](https://www.sciencedirect.com/science/article/abs/pii/S0169207017300018) |
| 10 | **Wheatcroft** — *…the case against the ranked probability score* | Critique of RPS | [JQAS 2021 / arXiv:1908.08980](https://arxiv.org/abs/1908.08980) |

---

## 🛟 Responsible gambling (jogo responsável) — mandatory

Betting is entertainment with a **negative expected value** for almost everyone; the math on this page is for **understanding and research**, not to encourage wagering. Set limits, never chase losses, never bet borrowed money. If it stops being fun, stop.

| Region | Resource | Contact |
|---|---|---|
| 🌍 Global | [Gambling Therapy](https://www.gamblingtherapy.org/) (Gordon Moody) | Free online support, multiple languages |
| 🇬🇧 UK | [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) | National Gambling Helpline **0808 8020 133** (24/7) |
| 🇧🇷 Brasil | [Jogo Responsável — SPA/Ministério da Fazenda](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel) · [Secretaria de Prêmios e Apostas](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) | Autoexclusão nacional; apoio emocional **CVV 188** (24h) |

Under the Brazilian **Lei das Bets** (Lei 14.790/2023), regulated operation began **1 Jan 2025** under the `.bet.br` domain with mandatory tools: deposit/loss/time limits, reality checks, and self-exclusion ([Ministério da Fazenda](https://www.gov.br/fazenda/pt-br/assuntos/noticias/2024/dezembro/regulamentacao-feita-pela-secretaria-de-premios-e-apostas-coloca-brasil-em-mercado-regulado-de-apostas-em-2025) · [Lei das Bets — Wikipédia](https://pt.wikipedia.org/wiki/Lei_das_Bets)).

---

**Keywords:** football/soccer prediction, futebol previsão de resultados, Poisson goals model, modelo de Poisson para gols, bivariate Poisson, Poisson bivariado, Dixon-Coles, correção de placares baixos, time-decay weighting, ponderação por decaimento temporal, negative binomial overdispersion, superdispersão, Skellam distribution, diferença de gols, zero-inflated Poisson, Elo rating, classificação Elo, pi-ratings, TrueSkill, Bayesian hierarchical model, modelo hierárquico bayesiano, MCMC, maximum likelihood, máxima verossimilhança, score matrix, matriz de placares, 1X2, over/under, mais/menos gols, Asian handicap, handicap asiático, correct score, placar exato, overround, vig, margem, devig, remoção de margem, Shin method, método de Shin, power method, wisdom of the crowd, log-loss, Brier score, ranked probability score, RPS, calibration, calibração, reliability diagram, expected value, valor esperado, Kelly criterion, critério de Kelly, fractional Kelly, risk of ruin, risco de ruína, Monte Carlo bankroll, favourite-longshot bias, viés favorito-azarão, closing line value, CLV, market efficiency, eficiência de mercado, responsible gambling, jogo responsável, Brasileirão, Lei das Bets.
