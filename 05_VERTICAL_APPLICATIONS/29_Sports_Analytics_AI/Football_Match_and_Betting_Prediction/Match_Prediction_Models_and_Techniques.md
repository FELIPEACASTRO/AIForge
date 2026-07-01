# Football Match Prediction Models & Techniques

> A dense, sourced map of the models, ratings, datasets, odds feeds and evaluation methods used to forecast football (soccer / *futebol*) match outcomes worldwide — for **research and education only**, not betting advice. Betting markets are highly efficient, beating the closing line is extremely hard, and **most bettors lose money**.

---

## ⚠️ Read this first — honest expectations & responsible gambling (*jogo responsável*)

- **The market is the benchmark, and it is very good.** The closing odds at a sharp book (e.g. Pinnacle) are near-perfectly calibrated; empirically the closing line explains observed outcome frequencies almost exactly, so it is the *hardest* baseline to beat. A model that merely "predicts winners" but cannot beat the **closing line** (Closing Line Value, CLV) has no demonstrable edge — see [Pinnacle: What is CLV](https://www.pinnacle.com/betting-resources/en/educational/what-is-closing-line-value-clv-in-sports-betting) and Constantinou & Fenton, *Profiting from an inefficient association football gambling market* ([evidence of inefficiency, 2013 PDF](http://constantinou.info/downloads/papers/evidenceofinefficiency.pdf)).
- **The ceiling is low and the draw is the hard class.** On commonly available data, models typically reach only ~**50–55% 1X2 accuracy** / RPS ≈ **0.19–0.21**. The strongest published benchmark to date (CatBoost on pi-ratings) reaches ≈**0.558 accuracy** / ≈**0.193 RPS** — still close to what de-vigged market odds already deliver. Draws (~26% of matches) are barely "predictable" as the modal class. See the review by Bunker, Yeung & Fujii 2024 ([arXiv:2403.07669](https://arxiv.org/abs/2403.07669)).
- **This page is for building forecasting skill, calibration and reproducible pipelines — not for gambling.** After a bookmaker's margin (*over-round / vig*), positive long-run ROI is rare and fragile. If you or someone you know is at risk, jump to [Responsible-gambling resources](#-responsible-gambling-resources-jogo-responsável).

---

## 1. Prediction targets (the markets / *mercados*)

| Market | Description (PT) | Outcome space | Typical model output |
|---|---|---|---|
| **1X2** (Match Odds) | Casa / Empate / Fora | 3-way categorical | P(home), P(draw), P(away) |
| **Double Chance** | Chance dupla (1X, 12, X2) | Pairs of 1X2 | Sum of 1X2 probs |
| **Over/Under goals** | Mais/Menos gols (2.5 etc.) | Binary at a line | From goal distribution |
| **BTTS** | Ambos marcam | Binary | 1 − P(0-x) − P(x-0) |
| **Correct Score** | Placar exato | ~ full score grid | Joint P(i,j) matrix |
| **Asian Handicap** | Handicap asiático | Line with push/quarter | From goal-difference dist. |
| **HT/FT, first goal, cards, corners** | Intervalo/Final etc. | various | Derived / separate models |

Goal-based statistical models are attractive precisely because a **single fitted score matrix P(home=i, away=j)** derives *all* of the above (1X2, O/U, BTTS, correct score, AH) consistently.

---

## 2. Evaluation — score the probabilities, not the tips

| Metric | What it measures | Proper? | Note |
|---|---|---|---|
| **Log-loss / Ignorance score** | Information / surprise | Strictly proper, *local* | Harsh on confident errors; favoured by Wheatcroft (below) |
| **Brier score** | Mean squared prob error | Strictly proper | Insensitive to outcome ordering |
| **RPS (Ranked Probability Score)** | Squared error on *cumulative* probs | Strictly proper, distance-sensitive | The de-facto football standard — Constantinou & Fenton 2012 |
| **Calibration / reliability** | Do "60%" events happen ~60%? | diagnostic | Reliability diagrams, ECE |
| **Accuracy / F1** | Top-1 hit rate | improper | Ignores confidence — avoid as primary metric |
| **ROI / yield & CLV** | Money & edge vs. market | economic | The only test that matters for "beating the market" |

- **RPS** respects ordering (an away win is "closer" to a draw than to a home win): Constantinou & Fenton, *Solving the Problem of Inadequate Scoring Rules for Assessing Probabilistic Football Forecast Models*, JQAS 8(1) 2012, [doi:10.1515/1559-0410.1418](https://doi.org/10.1515/1559-0410.1418) — [PDF](http://constantinou.info/downloads/papers/solvingtheproblem.pdf) · [repec](https://ideas.repec.org/a/bpj/jqsprt/v8y2012i1n12.html).
- **The case against RPS**: Wheatcroft (LSE) argues the Ignorance (log) score is preferable for comparing forecasters — *Evaluating probabilistic forecasts of football matches: the case against the RPS*, JQAS 17(4):273–287, 2021, [doi:10.1515/jqas-2019-0089](https://doi.org/10.1515/jqas-2019-0089) — [arXiv:1908.08980](https://arxiv.org/abs/1908.08980).
- **Validation must respect time**: use walk-forward / expanding-window backtests, never random k-fold, because matches are time-ordered.

---

## 3. Goal-based statistical models (the Poisson family)

Core idea: goals arrive ~ as a Poisson process; each team has **attack** and **defence** strengths, plus a **home advantage** (*fator casa*). The expected goals for home team `i` vs away `j`: `λ_home = α_i · β_j · γ`, `λ_away = α_j · β_i`.

| Model | Key idea | Strength | Reference |
|---|---|---|---|
| **Independent Poisson (Maher 1982)** | Separate attack/defence params; two independent Poissons | First rigorous team-strength model; still a solid baseline | Maher, *Modelling association football scores*, Statistica Neerlandica 36(3):109–118, [doi:10.1111/j.1467-9574.1982.tb00782.x](https://doi.org/10.1111/j.1467-9574.1982.tb00782.x) · [PDF](http://www.90minut.pl/misc/maher.pdf) |
| **Dixon–Coles (1997)** | Adds low-score dependence correction **τ (ρ)** for 0-0/1-0/0-1/1-1, plus **exponential time-decay** weighting of past matches | The reference model; fixes Poisson's under-count of draws/low scores; reported a positive betting return on 1995–96 odds | Dixon & Coles, *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*, JRSS-C 46(2):265–280, [doi:10.1111/1467-9876.00065](https://doi.org/10.1111/1467-9876.00065) |
| **Rue–Salvesen (2000)** | **Bayesian dynamic** GLM; attack/defence follow Brownian-motion random walk over time, MCMC inference | Principled time-variation of team form; retrospective + predictive | Rue & Salvesen, *Prediction and Retrospective Analysis of Soccer Matches in a League*, The Statistician 49(3):399–418, [doi:10.1111/1467-9884.00243](https://doi.org/10.1111/1467-9884.00243) |
| **Crowder et al. (2002)** | Fast deterministic approximation to Rue–Salvesen for EFL betting | Practical, near-real-time dynamic ratings | Crowder, Dixon, Ledford & Robinson, The Statistician 51(2):157–168, [doi:10.1111/1467-9884.00308](https://doi.org/10.1111/1467-9884.00308) |
| **Karlis–Ntzoufras bivariate & diagonal-inflated (2003)** | **Bivariate Poisson** models score correlation; **diagonal inflation** boosts all draw scores | Cleaner treatment of draws & correlation than DC's τ | Karlis & Ntzoufras, *Analysis of sports data by using bivariate Poisson models*, The Statistician 52(3):381–393, [doi:10.1111/1467-9884.00366](https://doi.org/10.1111/1467-9884.00366) · [PDF](http://www2.stat-athens.aueb.gr/~karlis/Bivariate%20Poisson%20Regression.pdf) |
| **Skellam / goal-difference (Karlis–Ntzoufras 2009)** | Model the goal *difference* directly via the Skellam distribution (difference of two Poissons) | Sidesteps needing both scores; natural for handicaps | Karlis & Ntzoufras, *Bayesian modelling of football outcomes: using the Skellam's distribution for the goal difference*, IMA J. Management Mathematics 20(2):133–145, [doi:10.1093/imaman/dpn026](https://doi.org/10.1093/imaman/dpn026) |
| **Bayesian hierarchical (Baio–Blangiardo 2010)** | Team params drawn from common hyper-priors (partial pooling); mixture prior fixes **overshrinkage** of extreme teams | Full posterior → honest predictive uncertainty; Serie A case study | Baio & Blangiardo, *Bayesian hierarchical model for the prediction of football results*, J. Applied Statistics 37(2):253–264, [doi:10.1080/02664760802684177](https://doi.org/10.1080/02664760802684177) · [author page](https://gianluca.statistica.it/research/football/) |

---

## 4. Rating systems (team-strength → win probability)

| System | Idea | Free data / access | Reference |
|---|---|---|---|
| **Elo** | Zero-sum rating updated by result vs. expectation | — | Elo (chess); adapted widely |
| **Club Elo** | Elo for clubs (data back to ~1939); **Fixtures endpoint gives goal-difference & score probabilities** | **Free CSV API** — [clubelo.com/API](http://clubelo.com/API) (`api.clubelo.com/YYYY-MM-DD`, `/CLUBNAME`, `/Fixtures`) | [clubelo.com](http://clubelo.com/) |
| **World Football Elo** | Elo for **national teams**, all "A" internationals; K weighted by margin, match importance, home edge | **Free site** — [eloratings.net](https://eloratings.net/) · [about](https://www.eloratings.net/about) | [Wikipedia](https://en.wikipedia.org/wiki/World_Football_Elo_Ratings) |
| **pi-ratings (Constantinou–Fenton 2013)** | Dynamic team ratings from **relative goal discrepancies**; separate home/away form | Method PDF | Reportedly outperformed Elo and was profitable over 5 EPL seasons — JQAS 9(1):37–50, [doi:10.1515/jqas-2012-0036](https://doi.org/10.1515/jqas-2012-0036) · [pi-ratings PDF](http://www.constantinou.info/downloads/papers/pi-ratings.pdf) |
| **FiveThirtyEight SPI** | Off/def ratings → Poisson match forecast (Nate Silver, SPI introduced 2009) | **Free CSV, but frozen** — updates stopped **21 June 2023** | [github/fivethirtyeight/data soccer-spi](https://github.com/fivethirtyeight/data/tree/master/soccer-spi) · [method](https://fivethirtyeight.com/features/how-our-club-soccer-projections-work/) |
| **Opta Power Rankings / Supercomputer** | Elo-derived, **0–100 scale**, **~13,000+ clubs / 180+ countries / 400+ leagues**; season sims blend rankings + market odds, run thousands of times | Editorial free; data paid (Stats Perform) | [Opta football predictions](https://theanalyst.com/articles/opta-football-predictions) · [Power Rankings explainer](https://theanalyst.com/articles/power-rankings-your-club-ranked) |
| **Massey** | Least-squares on point (goal) differentials | via `penaltyblog` | [Massey, network-science view (arXiv:1701.03363)](https://arxiv.org/abs/1701.03363) |
| **Colley** | Regularised least-squares on wins/losses only (no scores) | via `penaltyblog` | [Network diffusion family: Markov/Massey/Colley (JQAS 2018)](https://doi.org/10.1515/jqas-2017-0098) |
| **TrueSkill (Microsoft)** | Bayesian skill = mean ± uncertainty (μ,σ); handles teams & draws | `trueskill` (Python) | [Microsoft Research](https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/) · [Wikipedia](https://en.wikipedia.org/wiki/TrueSkill) |

Ratings are cheap, robust and make excellent **features** for ML models (Elo/pi-ratings consistently rank among the most important inputs).

---

## 5. Expected goals (xG) & xG-based prediction

- **xG** assigns each shot a scoring probability (0–1) from features like distance, angle, body part, assist type, and (in richer models) goalkeeper/defender positions. Team xG is **more predictive of future results than actual goals or shot counts** — [Opta: What is xG](https://theanalyst.com/articles/what-is-expected-goals-xg) · [Wikipedia: Expected goals](https://en.wikipedia.org/wiki/Expected_goals).
- Providers differ (no single "true" xG): **StatsBomb**, **Opta/Stats Perform**, **Understat** each use different features and event definitions.
- **Prediction use:** replace noisy goals with **xG-based attack/defence strengths**, or feed rolling xG/xGA into ML; simulate matches from shot-level xG. xG-informed Dixon–Coles-style models tend to de-noise short-run variance.

---

## 6. Machine-learning approaches

| Family | Typical use | Strengths / notes |
|---|---|---|
| **Logistic / multinomial / ordinal regression** | 1X2, O/U | Interpretable, well-calibrated baseline; ordinal respects H>D>A ordering |
| **Random Forest** | 1X2, score | Robust, low-tuning; competitive with limited features (used for FIFA WC outcome models) |
| **Gradient-boosted trees — XGBoost / LightGBM / CatBoost** | 1X2, O/U, ratings-features | **Usual top performers**; CatBoost + pi-ratings is the best-reported combo in benchmarks |
| **SVM** | 1X2 | Historically common; generally trails GBTs |
| **k-NN on engineered ratings** | 1X2 | **Best single model at the 2017 Soccer Prediction Challenge** (k-NN on rating features, RPS ≈ 0.215) |
| **Neural nets (MLP)** | 1X2, goals | Need lots of data; rarely beat GBTs on tabular match data |
| **Ensembles / stacking** | all | Blend statistical + rating + ML models; often best calibrated |

Evidence base: Bunker, Yeung & Fujii, *Machine Learning for Soccer Match Result Prediction*, 2024 — best model **CatBoost + pi-ratings** (acc ≈ **0.558**, RPS ≈ **0.193**), beating the 2017-challenge best (k-NN on ratings, ≈0.505 acc / ≈0.215 RPS) — [arXiv:2403.07669](https://arxiv.org/abs/2403.07669) · [book chapter doi:10.1007/978-3-031-76047-1_2](https://doi.org/10.1007/978-3-031-76047-1_2). Systematic review: Galekwa et al., *A Systematic Review of Machine Learning in Sports Betting*, 2024 — [arXiv:2410.21484](https://arxiv.org/abs/2410.21484). **Feature engineering (ratings, recent form, rest days, xG) matters more than the algorithm.**

---

## 7. Deep learning, sequence & graph models

| Approach | Idea | Reference |
|---|---|---|
| **Rating/recency feature learning + GBT/DL** | Engineer temporal features, then boost/deep-net | Berrar, Lopes & Dubitzky, *Incorporating domain knowledge in machine learning for soccer outcome prediction*, ML 108(1):97–126, 2019 — [doi:10.1007/s10994-018-5747-8](https://doi.org/10.1007/s10994-018-5747-8) |
| **Deep learning + GBT feature optimization** | DL feature selection feeding GBTs (CatBoost + pi-ratings) | Yeung, Bunker, Umemoto & Fujii, *Evaluating Soccer Match Prediction Models*, 2024 — [arXiv:2309.14807](https://arxiv.org/abs/2309.14807) |
| **Graph / passing-network models** | Use the passing network directly as a graph (GNN) | Lee, Park & del Pobil, *We know who wins: graph-oriented approaches of passing networks for predictive football match outcomes*, J. Big Data 12(1), 2025 — [doi:10.1186/s40537-025-01203-9](https://doi.org/10.1186/s40537-025-01203-9) |
| **Player–team interaction graph transformer** | Heterogeneous graph transformer over players & teams (HIGFormer, Wyscout data) | [arXiv:2507.10626](https://arxiv.org/abs/2507.10626) |
| **Axial transformer, in-game forecasting** | Live match/team/player forecasting (~75k live predictions per game, low latency) | Horton & Lucey, [arXiv:2511.18730](https://arxiv.org/abs/2511.18730) |
| **Distributional forecasting** | Predict full score distribution (shot quantity & quality) | Mendes-Neves et al., *Forecasting Soccer Matches through Distributions*, 2025 — [arXiv:2501.05873](https://arxiv.org/abs/2501.05873) |

Reality check: fancier architectures rarely beat well-tuned Dixon–Coles / GBT-on-ratings on **pre-match** 1X2 with commonly available data; DL/graph gains show up mainly with **rich event/tracking data** and **in-play** settings.

---

## 8. Odds-based & hybrid models (market as feature and benchmark)

- **De-vig the odds** into implied probabilities and treat them as a **strong prior / feature** — this is often the single most predictive input. Blending model + market (e.g. logistic stacking) usually beats either alone.
- **Two honest baselines every project needs:** (1) a naive "always predict market favourite", and (2) the **de-vigged closing odds** themselves. If your model can't beat (2) out-of-sample on log-loss/RPS, it has no edge.
- Benchmark competitions: **2017 Soccer Prediction Challenge** (Open International Soccer DB: 216,743 matches from 52 leagues / 35 countries; predict 206 future matches) — Berrar/Lopes/Dubitzky, and the **2023 Soccer Prediction Challenge** (Task 1 exact score, Task 2 1X2 probs; 736 matches from 44 leagues) — [Springer ML 2024, doi:10.1007/s10994-024-06625-9](https://doi.org/10.1007/s10994-024-06625-9).

---

## 9. Datasets (free vs paid, worldwide coverage)

| Dataset | Coverage | Content | Cost | Link |
|---|---|---|---|---|
| **football-data.co.uk** | ~22 European divisions + extra leagues, 2000/01→now | Results **+ bookmaker odds** (Bet365, Pinnacle, WH…) CSV | **Free** | [football-data.co.uk](https://www.football-data.co.uk/data.php) |
| **Kaggle European Soccer DB** (Hugo Mathien) | 11 countries, 2008–2016, 25k+ matches | Results, **odds (up to 10 books)**, FIFA attributes, some events | **Free** | [Kaggle](https://www.kaggle.com/datasets/hugomathien/soccer) |
| **StatsBomb Open Data** | Selected comps (WC, women's, historic) | **Event data** (JSON), incl. xG | **Free (attribution + non-commercial licence)** | [github/statsbomb/open-data](https://github.com/statsbomb/open-data) |
| **Understat** | EPL, La Liga, Bundesliga, Serie A, Ligue 1, RFPL; 2014/15→ | **Shot-level xG** | **Free (scrape)** | [understat.com](https://understat.com/) · [understatapi](https://pypi.org/project/understatapi/) |
| **openfootball** | **Worldwide** incl. N. America, Asia, Africa, Australia; World Cups in a sister repo | Fixtures/results, JSON/TXT | **Free (public domain, CC0)** | [github/openfootball/football.json](https://github.com/openfootball/football.json) · [world](https://github.com/openfootball/world) · [worldcup.json](https://github.com/openfootball/worldcup.json) |
| **Open International Soccer DB** | 216,743 matches, 52 leagues / 35 countries | Results for ML challenges | **Free** | Dubitzky et al. 2019 — [doi:10.1007/s10994-018-5726-0](https://doi.org/10.1007/s10994-018-5726-0) |
| **Campeonato Brasileiro (adaoduque)** 🇧🇷 | Brasileirão Série A, pontos-corridos era | Match results, lineups | **Free** | [Kaggle](https://www.kaggle.com/datasets/adaoduque/campeonato-brasileiro-de-futebol) |
| **Brazilian Soccer DB (ricardomattos05)** 🇧🇷 | Jogos do Brasileirão | Results | **Free** | [Kaggle](https://www.kaggle.com/datasets/ricardomattos05/jogos-do-campeonato-brasileiro) |
| **StatsBomb / Opta / Wyscout (full)** | Global, event + tracking | Pro event & tracking | **Paid** | [Stats Perform / Opta](https://www.statsperform.com/products/opta-data/) |

---

## 10. Odds data & prediction APIs (free vs paid)

| Service | Coverage | Free tier | Paid | Link |
|---|---|---|---|---|
| **football-data.org** | 12 comps incl. **Brazilian Série A** + **World Cup**, EPL, top-5 EU | **Free forever** (10 req/min, results/standings) | Paid tiers for more competitions/history | [football-data.org](https://www.football-data.org/) |
| **API-Football** (API-Sports) | **1,200+ leagues/cups**, live, **odds** | Free 100 req/day | ~$19–29/mo | [api-football.com](https://www.api-football.com/) |
| **The Odds API** | Many soccer leagues incl. **Campeonato Brasileiro Série A**, EPL, UCL | Free 500 credits/mo | tiered | [the-odds-api.com](https://the-odds-api.com/) |
| **Club Elo API** | Club Elo + fixture probabilities | **Free** CSV | — | [clubelo.com/API](http://clubelo.com/API) |
| **Pinnacle** | Sharp, low-margin closing lines (the efficiency benchmark) | odds feed | account/affiliate | [pinnacle.com resources](https://www.pinnacle.com/betting-resources/) |

> Historical **closing odds** (needed for honest CLV backtests) are the scarce resource: football-data.co.uk and the Kaggle European DB carry historic bookmaker odds; most live APIs only give current prices.

---

## 11. Tools & libraries (open source)

| Tool | Language | What it does | Link |
|---|---|---|---|
| **penaltyblog** | Python | Poisson, Bivariate Poisson, **Dixon–Coles** (+ time-decay), Bayesian/hierarchical (MCMC), **Elo/Massey/Colley/pi-ratings**, scrapers. MIT, v1.11.0 (2026) | [github](https://github.com/martineastwood/penaltyblog) · [PyPI](https://pypi.org/project/penaltyblog/) |
| **soccerdata** | Python | Scrapers: Club Elo, ESPN, FBref, football-data.co.uk, Sofascore, SoFIFA, Understat, WhoScored → tidy DataFrames | [github](https://github.com/probberechts/soccerdata) |
| **worldfootballR** | R | FBref, Transfermarkt, Understat extraction (**archived Sept 2025**; still installable) | [github](https://github.com/JaseZiv/worldfootballR) |
| **StatsBombPy** | Python | Stream StatsBomb open + API event data | [statsbombpy](https://github.com/statsbomb/statsbombpy) |
| **socceraction** | Python | Value actions (VAEP/xT) from event data | [docs](https://socceraction.readthedocs.io/) |
| **regista** | R | Dixon–Coles team strengths | [regista](https://torvaney.github.io/regista/) |
| **trueskill** | Python | TrueSkill ratings | [Microsoft Research](https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/) |

---

## 12. Regional coverage notes (*cobertura por região*)

- **Europe (top leagues):** best-served — football-data.co.uk (results + odds), Understat (xG top-5), FBref via worldfootballR/soccerdata.
- **South America incl. 🇧🇷 Brasileirão:** football-data.org (free Série A + World Cup), The Odds API (Campeonato Brasileiro Série A odds), Kaggle Brasileirão datasets (adaoduque, ricardomattos05), FBref (Série A/B) via scrapers. *Terminologia:* 1X2 = Casa/Empate/Fora, over/under = mais/menos gols, handicap asiático, ambos marcam.
- **Africa & Asia:** thinnest odds/xG coverage; **openfootball/world** and API-Football give fixtures/results for many African & Asian leagues; expect sparser markets and larger model uncertainty.
- **North America (MLS):** FBref/StatsBomb coverage is good; American Soccer Analysis publishes open xG-style metrics; API-Football/football-data cover fixtures.
- **International / World Cup:** eloratings.net (national-team Elo), openfootball/worldcup.json (incl. 2026 Canada/USA/Mexico), StatsBomb open WC data.

---

## 13. Key research & further reading

- Maher 1982 — independent Poisson foundations — [doi:10.1111/j.1467-9574.1982.tb00782.x](https://doi.org/10.1111/j.1467-9574.1982.tb00782.x)
- Dixon & Coles 1997 — the reference model — [doi:10.1111/1467-9876.00065](https://doi.org/10.1111/1467-9876.00065)
- Rue & Salvesen 2000 — Bayesian dynamic — [doi:10.1111/1467-9884.00243](https://doi.org/10.1111/1467-9884.00243)
- Karlis & Ntzoufras 2003 — bivariate/diagonal-inflated Poisson — [doi:10.1111/1467-9884.00366](https://doi.org/10.1111/1467-9884.00366); 2009 — Skellam goal-difference — [doi:10.1093/imaman/dpn026](https://doi.org/10.1093/imaman/dpn026)
- Baio & Blangiardo 2010 — Bayesian hierarchical — [doi:10.1080/02664760802684177](https://doi.org/10.1080/02664760802684177)
- Constantinou & Fenton 2013 — pi-ratings — [doi:10.1515/jqas-2012-0036](https://doi.org/10.1515/jqas-2012-0036); 2012 — RPS scoring rules — [doi:10.1515/1559-0410.1418](https://doi.org/10.1515/1559-0410.1418)
- Wheatcroft 2021 — case against RPS — [doi:10.1515/jqas-2019-0089](https://doi.org/10.1515/jqas-2019-0089) · [arXiv:1908.08980](https://arxiv.org/abs/1908.08980)
- Bunker, Yeung & Fujii 2024 — ML review — [arXiv:2403.07669](https://arxiv.org/abs/2403.07669); Galekwa et al. 2024 — betting-ML review — [arXiv:2410.21484](https://arxiv.org/abs/2410.21484)

---

## 🆘 Responsible-gambling resources (*jogo responsável*)

**This material is for research and education. It is not betting advice. Betting can cause serious financial and personal harm; markets are efficient and most bettors lose. Never bet to solve financial problems, never chase losses, and never stake money you cannot afford to lose.**

| Region | Resource | Contact |
|---|---|---|
| 🌍 International | **Gambling Therapy** (multilingual online support, run by the Gordon Moody charity) | [gamblingtherapy.org](https://www.gamblingtherapy.org/) |
| 🌍 International | **Gamblers Anonymous** | [gamblersanonymous.org](https://www.gamblersanonymous.org/) |
| 🇬🇧 UK | **GamCare — National Gambling Helpline** (free, 24/7) | [gamcare.org.uk](https://www.gamcare.org.uk/) · 0808 8020 133 |
| 🇬🇧 UK | **GambleAware** (info & self-assessment; funding transitioning to a statutory NHS-led levy) | [gambleaware.org](https://www.gambleaware.org/) |
| 🇧🇷 Brasil | **CVV** — apoio emocional / prevenção (24h) | [cvv.org.br](https://www.cvv.org.br/) · 188 |
| 🇧🇷 Brasil | **Instituto Brasileiro de Jogo Responsável (IBJR)** | [ibjr.org.br](https://ibjr.org.br/en/responsible-gaming/) |

*Contexto regulatório 🇧🇷:* apostas de quota fixa ("bets") são reguladas pela **Lei nº 14.790/2023** e pela Secretaria de Prêmios e Apostas (SPA) do Ministério da Fazenda, com regras obrigatórias de jogo responsável e autoexclusão.

---

**Keywords:** football match prediction, soccer betting models, previsão de resultados de futebol, apostas esportivas, Poisson model, Dixon-Coles, bivariate Poisson, Bayesian hierarchical, Elo rating, Club Elo, pi-ratings, SPI, Opta Power Rankings, expected goals (xG), gols esperados, XGBoost, CatBoost, random forest, ranked probability score (RPS), Brier score, log-loss, calibração, 1X2 casa empate fora, over/under gols, ambos marcam (BTTS), handicap asiático, placar exato, closing line value (CLV), market efficiency, eficiência de mercado, Brasileirão, Campeonato Brasileiro, jogo responsável, responsible gambling, football-data.co.uk, StatsBomb, Understat, penaltyblog, The Odds API.
