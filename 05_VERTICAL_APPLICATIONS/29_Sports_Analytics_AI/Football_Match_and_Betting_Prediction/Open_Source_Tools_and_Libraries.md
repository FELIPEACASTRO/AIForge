# Open-Source Tools & Libraries for Football Prediction

> Authoritative, worldwide index of **open-source (código aberto) Python & R libraries** for modelling football (soccer / futebol) matches — goals models (Poisson / Dixon-Coles), team ratings (classificações), event/tracking data, action valuation (VAEP/xT), scrapers (raspagem), visualization, and Monte-Carlo tournament simulation. Every repo below was checked for existence + maintenance status. Built for **research & education (pesquisa e educação — data science / ML)**, current 2024–2026.

> ⚠️ **Research & education only — not betting advice (não é dica de aposta).** Football betting markets are **highly efficient**: sharp books (e.g. Pinnacle) price near the true probability, and the **closing line** is very hard to beat. Most bettors lose money over time. These tools teach modelling; they are **not a system to make money**. If gambling is a problem, get help: [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) · [Gambling Therapy](https://www.gamblingtherapy.org/) · 🇧🇷 [Jogo Responsável (SPA/MF)](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) · 🇧🇷 CVV **188**.

**Where the data lives:** this page is about *tools*; for datasets, odds and APIs see the sibling pages → [Global Football Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md) · [Kaggle Football Datasets](./Kaggle_Football_Datasets_and_Competitions.md).

**A typical open-source stack (pilha típica):** `soccerdata`/`ScraperFC` (get data) → `penaltyblog`/`goalmodel` (fit a Dixon-Coles goals model) → `elote`/ClubElo (ratings) → `kloppy` + `socceraction` (event data → VAEP features) → `mplsoccer` (visualize) → Monte-Carlo loop (simulate the league/cup). Then **back-test against closing odds** to see how efficient the market is.

---

## 0) Pick a tool by task (escolha rápida)

| I want to… | Best open-source pick | Language |
|---|---|---|
| Predict 1X2 / correct score / over-under from goals | **penaltyblog** (`DixonColesGoalModel`) or **goalmodel** | Python / R |
| Do it the rigorous Bayesian way (uncertainty) | **footBayes** (Stan) | R |
| Rank teams / cheap strong baseline | **elote** (Elo/Glicko) · **ClubElo** API · penaltyblog ratings | Python |
| Scrape results + xG (Brazil/global) | **soccerdata** or **ScraperFC** (FBref) | Python |
| Load & standardize event/tracking data | **kloppy** (+ **statsbombpy** for StatsBomb) | Python |
| Value on-ball actions (VAEP / xT) | **socceraction** | Python |
| Sync event ↔ tracking, compute physical/pressure | **databallpy** · **floodlight** | Python |
| Draw pitches, shot maps, pass networks | **mplsoccer** | Python |
| Simulate a league or the World Cup | **fbRanks** / **footBayes** (built-in) or a Monte-Carlo loop on penaltyblog | R / Python |

---

## 1) Match-outcome & goals modelling (Python)

The core of match prediction: **bivariate-Poisson / Dixon-Coles** goal models that output P(home/draw/away), correct-score grids, over/under and Asian-handicap probabilities. `penaltyblog` is the most complete and actively maintained.

| Library | Purpose | Key models | Maintained? | Link |
|---|---|---|---|---|
| **penaltyblog** | End-to-end football analytics: models + scrapers + ratings + backtest | Poisson, Bivariate Poisson, **Dixon-Coles**, Zero-Inflated Poisson, Negative Binomial, Weibull-Copula, Bayesian; Elo/Massey/Colley/Pi ratings; odds decoding | ✅ **Active** — v1.11.0 (Mar 2025), Python 3.10–3.13, Cython | [github](https://github.com/martineastwood/penaltyblog) · [PyPI](https://pypi.org/project/penaltyblog/) · [docs](https://docs.pena.lt/y/) |
| **socceraction** (SPADL/VAEP/xT) | Convert event streams → SPADL / atomic-SPADL; value actions with **VAEP** & Expected Threat | ML action-valuation, xT Markov model | ⚠️ **Maintenance-only** (research reproducibility) — v1.5.3 (Aug 2024) | [github](https://github.com/ML-KULeuven/socceraction) · [VAEP paper (KDD'19)](https://arxiv.org/abs/1802.07127) |

> ⚠️ **Note on non-existent packages (avoid hallucinated repos):** community lists sometimes cite `footballprob`, `PySpados`, or a standalone `atomic-spadl` repo — **none exists** on PyPI/GitHub. Use `penaltyblog` (its `FootballProbabilityGrid` gives all outcome probabilities); for SPADL **and** atomic-SPADL use `socceraction` — both action representations ship **inside** socceraction (`socceraction.atomic.spadl`), there is no separate `atomic-spadl` package.

---

## 2) Team & player rating systems (Elo, Glicko, WHR)

Ratings (classificações de força) are the cheapest, strongest baseline for match prediction — feed the rating difference into a logistic/ordered model, or convert directly to win/draw/loss probabilities.

| Library | Lang | Systems | Maintained? | Link |
|---|---|---|---|---|
| **elote** | Python | Elo, Glicko-1, **Glicko-2**, TrueSkill, ECF, DWZ; head-to-head arena/benchmarking | ✅ Active — v1.1.0 (Apr 2025) | [github](https://github.com/wdm0006/elote) |
| **skelo** | Python | Elo + Glicko-2 with a **scikit-learn** estimator interface (pipelines/backtest) | ~ Stable | [github](https://github.com/mbhynes/skelo) · [PyPI](https://pypi.org/project/skelo/) |
| **whole_history_rating (whr)** | Python | Rémi Coulom **Whole-History Rating** (time-varying strengths) | ~ Stable | [pfmonville](https://github.com/pfmonville/whole_history_rating) · [wind23 (C++/py)](https://github.com/wind23/whole_history_rating) |
| **penaltyblog.ratings** | Python | Elo, Massey, Colley, **Pi-ratings** in one package | ✅ Active | [github](https://github.com/martineastwood/penaltyblog) |
| **ClubElo API** | (data) | Ready-made daily club Elo + win/draw/loss probs — no fitting needed | ✅ | [clubelo.com/API](http://clubelo.com/API) |

---

## 3) R packages for goals models & Bayesian football

R has the richest **statistical** football-modelling ecosystem (Dixon-Coles is originally an R/statistics literature model). Great for rigorous, well-documented modelling and simulation.

| Package | Purpose | Models | Maintained? | Link |
|---|---|---|---|---|
| **goalmodel** | Flexible goals models with time-decay weighting | Poisson, **Dixon-Coles**, Negative Binomial, Conway-Maxwell-Poisson; xG-as-input | ✅ Maintained — v0.6.4 | [github](https://github.com/opisthokonta/goalmodel) |
| **regista** | Tidyverse-style, extensible Dixon-Coles (`dixoncoles`, `dixoncoles_ext`) | Dixon-Coles (arbitrarily extended, e.g. + xG) | ~ Stable | [github](https://github.com/Torvaney/regista) · [docs](https://torvaney.github.io/regista/) |
| **footBayes** | **Bayesian** football via Stan/CmdStan; MLE too; forecast + viz | Double Poisson, Bivariate Poisson, Skellam, Student-t, (zero/diagonal-)inflated | ✅ Active — CRAN v2.0.0 | [CRAN](https://cran.r-project.org/web/packages/footBayes/index.html) · [vignette](https://cran.r-project.org/web/packages/footBayes/vignettes/footBayes_a_rapid_guide.html) |
| **fbRanks** | Time-dependent Poisson-regression ranking + **tournament/league simulation** | Dixon-Coles-style Poisson; glm/glmnet fitting | ⚠️ **Archived from CRAN** 2022-06-15; last v2.0 (Eli Holmes) — install via CRAN archive | [rdrr docs](https://rdrr.io/cran/fbRanks/) · [CRAN mirror](https://github.com/cran/fbRanks) |
| **engsoccerdata** | Historical results **dataset-as-package** (England + Europe 1871–2022) | data (for training/backtests) | ✅ Community-maintained | [github](https://github.com/jalapic/engsoccerdata) |
| **worldfootballR** | Scraper: FBref, Transfermarkt, Understat, FotMob (xG, market values) | data acquisition | ⚠️ GitHub repo **archived** (2026); still published on **CRAN** & usable | [github](https://github.com/JaseZiv/worldfootballR) · [docs](https://jaseziv.github.io/worldfootballR/) |
| **StatsBombR** | Official R client for StatsBomb free + licensed event/360 data | data acquisition | ✅ Official | [github](https://github.com/statsbomb/StatsBombR) |

---

## 4) Event & tracking data — load, standardize, value actions (Python)

Modern features (xG, xT, VAEP, pressing, pitch control) come from **event data** (every on-ball action with x/y) and **tracking data** (all 22 players + ball at 10–25 Hz). These libraries ingest many providers and normalize coordinates/definitions.

| Library | Purpose | Providers unified | Maintained? | Link |
|---|---|---|---|---|
| **kloppy** (PySport) | Vendor-independent model for **event + tracking**; export Pandas/Polars, SportsCode XML | StatsBomb, Opta, Wyscout, Sportec, Metrica, TRACAB, SecondSpectrum, SkillCorner, DataFactory, PFF | ✅ **Very active** — v3.19.x | [github](https://github.com/PySport/kloppy) · [docs](https://kloppy.pysport.org/) |
| **socceraction** | Event → SPADL → **VAEP / xT** action values (ML) | StatsBomb, Opta, Wyscout, Stats Perform, WhoScored | ⚠️ Maintenance-only | [github](https://github.com/ML-KULeuven/socceraction) |
| **statsbombpy** | Official Python client for StatsBomb open + licensed data (events, 360) | StatsBomb | ✅ Official/active | [github](https://github.com/statsbomb/statsbombpy) · [open-data](https://github.com/statsbomb/open-data) |
| **floodlight** | Provider-independent load/process/model of tracking, event & code data; space control, metabolic power, entropy | DFL/Sportec, Opta, StatsBomb, TRACAB, SecondSpectrum, etc. | ~ Semi-active — v1.2.0 (Py 3.13) | [github](https://github.com/floodlight-sports/floodlight) · [paper](https://arxiv.org/abs/2206.02562) |
| **databallpy** | Load + **synchronise** event↔tracking (Needleman-Wunsch), pressure, covered distance, viz | Metrica, Tracab, Opta, SciSports, Inmotio, StatsBomb | ✅ Active — JOSS-published | [github](https://github.com/Alek050/databallpy) · [docs](https://databallpy.readthedocs.io/) · [JOSS](https://doi.org/10.21105/joss.10223) |
| **codeball** | Metrica's tactical/EPV/pass-network/pitch-control toolkit on Pandas subclasses | Metrica formats | ❌ **Stale** — last release v0.3.1 (2021) | [github](https://github.com/metrica-sports/codeball) |

---

## 5) Data acquisition — scrapers & odds tools (Python)

Get results, xG, standings, market values and odds (cotações) into DataFrames. Respect each site's Terms of Service (raspagem responsável) and rate limits.

| Library | Scrapes | Maintained? | Link |
|---|---|---|---|
| **soccerdata** | Club Elo, ESPN, **FBref**, Football-Data.co.uk, Sofascore, SoFIFA, **Understat**, WhoScored — unified column names, cached | ✅ **Very active** — v1.9.0 (Apr 2026) | [github](https://github.com/probberechts/soccerdata) · [docs](https://soccerdata.readthedocs.io/) |
| **ScraperFC** | FBref, Sofascore, Understat, Club Elo, Capology, Transfermarkt (Europe + Americas + Asia) | ✅ Active | [github](https://github.com/oseymour/ScraperFC) · [PyPI](https://pypi.org/project/ScraperFC/) |
| **understat** (amosbastian) | Async wrapper for Understat shot-level **xG/xA** (top-5 EU + Russia) | ~ Stable | [github](https://github.com/amosbastian/understat) · [understatapi](https://pypi.org/project/understatapi/) |
| **soccerapi** | Bookmaker **odds** scraper (888sport, bet365, Unibet) → 1X2 / O-U / Asian handicap | ⚠️ Fragile (books block bots; ToS/VPN issues) | [github](https://github.com/S1M0N38/soccerapi) |

> 🇧🇷 **Brazil (futebol brasileiro):** for **Brasileirão Série A/B, Libertadores, Sudamericana, Copa do Brasil**, `soccerdata` and `ScraperFC` both reach **FBref** (StatsBomb-powered xG for Brazil) and **Understat** does not cover Brazil — use FBref via `soccerdata`. Odds history for Brazil is on football-data.co.uk (extra leagues). See the [datasets page](./Global_Datasets_and_Data_APIs.md#-brazil-brasileirão--futebol-brasileiro).

---

## 6) Visualization (visualização)

| Library | Lang | Purpose | Maintained? | Link |
|---|---|---|---|---|
| **mplsoccer** | Python | Matplotlib **pitches** (campo), shot maps, pass networks, heatmaps, radar/pizza charts; loads StatsBomb open data; provider `Standardizer` | ✅ Active — v1.6.x | [github](https://github.com/andrewRowlinson/mplsoccer) · [docs](https://mplsoccer.readthedocs.io/) |
| **floodlight** | Python | Space-control / pitch-control plots from tracking | ~ Semi-active | [github](https://github.com/floodlight-sports/floodlight) |
| **soccermatics** (R) | R | ggplot2 pitch plots, StatsBomb helpers (companion to the *Soccermatics* course) | ~ Stable | [rdrr](https://rdrr.io/github/JoGall/soccermatics/) |

---

## 7) Tournament & Monte-Carlo simulation

Once you have a goals model or ratings, **simulate the fixture list thousands of times** to get league-title / promotion / knockout / World-Cup probabilities. Most of this is a short loop on top of the libraries above; a few packages/repos ship it ready.

| Tool / approach | What it does | Link |
|---|---|---|
| **fbRanks** (R) | Built-in `simulate.*` for leagues & knockout cups from fitted Poisson model (archived from CRAN 2022 — install via CRAN archive) | [rdrr](https://rdrr.io/cran/fbRanks/) |
| **footBayes** (R) | Bayesian posterior-predictive match & season simulation | [CRAN](https://cran.r-project.org/web/packages/footBayes/index.html) |
| **penaltyblog** (Py) | Fit Dixon-Coles → sample score grids → loop the calendar for a Monte-Carlo table | [github](https://github.com/martineastwood/penaltyblog) |
| **World-Cup 2026 community models** | Open repos combining **Elo + Dixon-Coles + Monte-Carlo** (educational; verify before reuse) | [Hicruben/world-cup-2026-prediction-model](https://github.com/Hicruben/world-cup-2026-prediction-model) · [0xNadr/wc2026 (Bayesian MC)](https://github.com/0xNadr/wc2026) |

**Recipe (receita):** `expected goals λ_home, λ_away` from your model → draw score from bivariate-Poisson/Dixon-Coles → apply cup rules (extra time, penalties) → repeat 10k–50k times → aggregate advancement/title odds.

---

## 8) Minimal verified example — Dixon-Coles with `penaltyblog`

Verified against the official docs ([docs.pena.lt/y](https://docs.pena.lt/y/models/dixon_coles.html), penaltyblog ≥ 1.x). Install: `pip install penaltyblog`.

```python
import penaltyblog as pb

# 1) Get data (free football-data.co.uk via the built-in scraper)
fb = pb.scrapers.FootballData("ENG Premier League", "2019-2020")
df = fb.get_fixtures()

# 2) Fit a Dixon-Coles goals model (modelo de gols)
clf = pb.models.DixonColesGoalModel(
    df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
)
clf.fit()

# 3) Predict a fixture -> FootballProbabilityGrid
probs = clf.predict("Liverpool", "Wolves")

probs.home_draw_away          # [P(home win), P(draw), P(away win)]
probs.home_win                # P(home win)
probs.total_goals("over", 1.5)     # P(over 1.5 goals)  -> mercado over/under
probs.asian_handicap("home", 1.5)  # Asian handicap prob -> handicap asiático
probs.both_teams_to_score          # BTTS / ambas marcam
```

For **time-decay weighting** (recent matches matter more, per Dixon & Coles 1997), pass `weights=pb.models.dixon_coles_weights(df["date"], xi=0.0018)` to `.fit()`. `neutral_venue=True` in `.predict()` removes home advantage (useful for World-Cup neutral sites).

---

## 9) Reality check — why open-source models rarely beat the market

- **The closing line is brutally efficient.** By kickoff, sharp money and late team news are fully priced in. A model that only ties the closing line has **zero edge after the margin (vig / overround)**. The honest skill metric is **Closing Line Value (CLV)** — do your prices *repeatedly* beat the close? ([Pinnacle: CLV](https://www.pinnacle.com/betting-resources/en/educational/what-is-closing-line-value-clv-in-sports-betting)).
- **Public models = public information.** penaltyblog, goalmodel, footBayes and ClubElo are excellent, but every serious bookmaker uses the same math (and much better data). An off-the-shelf Dixon-Coles will not systematically beat a sharp book.
- **The Kaunitz cautionary tale.** Kaunitz et al. (2017) built a strategy that beat *published* odds in back-test — but in live betting **bookmakers limited/closed the winning accounts**, erasing the edge ([arXiv 1710.02824](https://arxiv.org/abs/1710.02824)).
- **Use these tools to learn**, to build features, to study efficiency, to teach ML — **not as an income plan**. Back-test with proper time-splits, log-loss/Brier scoring, and realistic margins.

---

## 10) Responsible gambling (Jogo Responsável) — mandatory

Gambling can cause serious harm. This index exists for **data science / ML research and education only (somente pesquisa e educação)**. Set limits, never chase losses (nunca persiga o prejuízo), and seek help if betting stops being fun.

| Region | Resource | Contact |
|---|---|---|
| 🌍 Global | [Gambling Therapy](https://www.gamblingtherapy.org/) | Online chat/forum, multilingual |
| 🇬🇧 UK | [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) | National Gambling Helpline **0808 8020 133** (24/7) |
| 🇧🇷 Brazil | [Jogo Responsável — SPA/Ministério da Fazenda](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) (Portaria SPA/MF nº 1.231/2024) | Apoio emocional **CVV 188** |

---

## 11) Evaluation, baselines & curated lists (avaliação e aprendizado)

Model **quality** is judged with proper probabilistic scores, not accuracy. Always compare against a **market baseline** (implied probabilities from de-margined closing odds) and a **rating baseline** (Elo/ClubElo).

| Need | Tool / metric | Note |
|---|---|---|
| Score probabilistic forecasts | **log-loss** & **Brier score** (via `scikit-learn`) | Lower is better; the market is a very strong benchmark |
| De-margin odds → fair probabilities | `penaltyblog` odds/implied-probability utilities | Removes overround (vig) before comparison |
| Backtest with time-splits | `penaltyblog` backtest · scikit-learn `TimeSeriesSplit` | Never shuffle across time (evita vazamento / leakage) |
| Curated catalog of soccer-analytics OSS | **awesome-soccer-analytics** · **PySport open-source overview** | Living directories of libraries & datasets |

- **awesome-soccer-analytics** (curated list): [github.com/matiasmascioto/awesome-soccer-analytics](https://github.com/matiasmascioto/awesome-soccer-analytics)
- **PySport open-source overview** (filter by sport = Soccer): [opensource.pysport.org](https://opensource.pysport.org/)

**Foundational papers (exact citations, verified):**
- **Maher, M.J. (1982)** — "Modelling association football scores." *Statistica Neerlandica* **36**(3):109–118. [DOI:10.1111/j.1467-9574.1982.tb00782.x](https://doi.org/10.1111/j.1467-9574.1982.tb00782.x) — the independent-Poisson attack/defence model that started it all.
- **Dixon, M.J. & Coles, S.G. (1997)** — "Modelling Association Football Scores and Inefficiencies in the Football Betting Market." *Journal of the Royal Statistical Society: Series C (Applied Statistics)* **46**(2):265–280. [DOI:10.1111/1467-9876.00065](https://doi.org/10.1111/1467-9876.00065) — low-score dependence correction (τ) + exponential time-decay.
- **Decroos, Bransen, Van Haaren & Davis (2019)** — "Actions Speak Louder than Goals: Valuing Player Actions in Soccer" (VAEP), *KDD '19*. [arXiv:1802.07127](https://arxiv.org/abs/1802.07127).
- **Karun Singh (2018)** — "Introducing Expected Threat (xT)." [karun.in/blog/expected-threat.html](https://karun.in/blog/expected-threat.html).

---

## Related in AIForge
- [Global Football Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md) · [Kaggle Football Datasets & Competitions](./Kaggle_Football_Datasets_and_Competitions.md) · Parent vertical: [`../`](../) (Sports Analytics AI)

**Sources (all verified to exist with working URLs, Jul 2026):** github.com/martineastwood/penaltyblog · pypi.org/project/penaltyblog · docs.pena.lt/y · github.com/ML-KULeuven/socceraction · arxiv.org/abs/1802.07127 · github.com/wdm0006/elote · github.com/mbhynes/skelo · github.com/pfmonville/whole_history_rating · github.com/wind23/whole_history_rating · github.com/opisthokonta/goalmodel · github.com/Torvaney/regista · cran.r-project.org/web/packages/footBayes · rdrr.io/cran/fbRanks · github.com/jalapic/engsoccerdata · github.com/JaseZiv/worldfootballR · github.com/statsbomb/StatsBombR · github.com/statsbomb/statsbombpy · github.com/PySport/kloppy · github.com/floodlight-sports/floodlight · arxiv.org/abs/2206.02562 · github.com/Alek050/databallpy · doi.org/10.21105/joss.10223 · github.com/metrica-sports/codeball · github.com/probberechts/soccerdata · github.com/oseymour/ScraperFC · github.com/amosbastian/understat · github.com/S1M0N38/soccerapi · github.com/andrewRowlinson/mplsoccer · github.com/matiasmascioto/awesome-soccer-analytics · github.com/Hicruben/world-cup-2026-prediction-model · github.com/0xNadr/wc2026 · clubelo.com/API · karun.in/blog/expected-threat.html · doi.org/10.1111/1467-9876.00065 · doi.org/10.1111/j.1467-9574.1982.tb00782.x · pinnacle.com · arxiv.org/abs/1710.02824 · begambleaware.org · gamcare.org.uk · gamblingtherapy.org · gov.br/fazenda (SPA)

**Keywords:** football prediction open source, Dixon-Coles Python, Poisson goals model, penaltyblog, socceraction VAEP, expected threat xT, kloppy tracking data, soccerdata FBref scraper, mplsoccer, Elo ratings football, elote, whole history rating, footBayes, goalmodel R, fbRanks, Monte Carlo tournament simulation, modelo de previsão de futebol, modelo de gols Poisson, classificações Elo, raspagem de dados de futebol, xG, jogo responsável, eficiência do mercado de apostas, closing line value.
