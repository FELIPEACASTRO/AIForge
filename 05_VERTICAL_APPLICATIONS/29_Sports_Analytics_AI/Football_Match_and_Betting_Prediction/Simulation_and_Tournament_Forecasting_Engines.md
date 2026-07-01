# Simulation & Tournament Forecasting Engines

> **Monte Carlo simulation** for football (soccer / *futebol*) forecasting at the **season and tournament** level — a distinct discipline from single-match models. Instead of predicting one game, you take a per-match engine (Poisson / Dixon–Coles / Elo) and **simulate every remaining fixture thousands of times**, then count how often each team wins the title (*título*), qualifies (*classifica*), or is relegated (*rebaixamento*). This page indexes only items whose existence and URL were **verified** at the time of writing. Research & education only.

> ⚠️ **RESPONSIBLE GAMBLING — READ FIRST. This is a research & education page, NOT betting advice (não é aconselhamento de apostas).** A tournament simulator produces confident-looking percentages, but each one is **only as good as the match model underneath**. Two verified recent examples: at **Euro 2024**, Opta's model made England the favourite and **France second at 19.1%** — neither won; **Spain, rated ≈9.6%, did**. Before the **2025-26 Premier League** season the same model made **Liverpool the favourite at 28.5%** — **Arsenal won the title** (85 pts). High simulation percentages are not certainties. Markets are broadly efficient and no public simulator has shown a durable edge over closing odds. 🇧🇷 If gambling is causing harm, contact **CVV 188** (Centro de Valorização da Vida, Brazil) or your local support service.

---

## What this actually is

A tournament/season forecaster is a **two-layer** system:

1. **Base match model** — estimates each fixture's goal/result distribution (e.g. Dixon–Coles bivariate Poisson, or an Elo/SPI-style rating that outputs win/draw/loss probabilities).
2. **Monte Carlo shell** — draws a scoreline for every remaining fixture from that distribution, resolves the table or bracket, and repeats **N times** (Opta publicly reports **10,000 runs**). Counting outcomes across the N simulated worlds gives title / qualification / relegation percentages.

The percentages are only a *summary of the base model's assumptions* — they add no information the base model didn't already contain.

## Uncertainty: simulations inherit the base model's error

- **Correlated error, not averaged-away error.** Running 10,000 sims narrows *sampling* noise but does nothing about a *biased* base model. If the model overrates a team, every one of the 10,000 worlds overrates it.
- **Garbage in, garbage out.** Injuries, suspensions, rotation, motivation and tactical mismatches are usually absent from the goal model, yet they move real outcomes.
- **Overconfidence in the tails.** Long knockout paths multiply small per-match errors; single-elimination formats are dominated by variance, so even a "correct" model spreads probability widely.

## Live tournament & season forecasters (verified)

| Forecaster | URL | What it publishes | Status |
|---|---|---|---|
| **Opta "supercomputer"** (Opta Analyst / Stats Perform) | https://theanalyst.com/ | 10,000-run Monte Carlo sims for the World Cup, Euros and domestic leagues — title, top-4 and relegation % | Live, actively updated (World Cup 2026 and Premier League 2025-26 outputs verified) |
| **FiveThirtyEight SPI** (Soccer Power Index) | https://github.com/fivethirtyeight/data/tree/master/soccer-spi | Historical SPI ratings + match forecast CSVs (`spi_matches.csv`, `spi_global_rankings.csv`) | **Discontinued.** Sports forecasts ended ~June 2023; the site was shut down in 2025. The dataset archive is **frozen** — useful for backtesting only, not new forecasts. |

## Data sources you feed the engine (verified)

| Source | URL | Use |
|---|---|---|
| **Club Elo** | http://clubelo.com/ (API: http://api.clubelo.com/) | Club Elo ratings with full history and a plain-CSV API — strength priors for the base model |
| **Football-Data.co.uk** | https://www.football-data.co.uk/ | Historical results + opening/closing bookmaker odds as CSV/Excel — backtesting and probability calibration |
| **Understat** | https://understat.com/ | Expected-goals (xG) for the five big leagues + RPL, exportable as CSV/JSON/XLSX — a live xG feed for goal models |
| **FBref** (Sports Reference) | https://fbref.com/ | Broad team/player stats. Note: FBref's **advanced (xG) stats were discontinued in January 2026**, so use Understat for current xG feeds. |

## Open-source simulators & modelling libraries (verified)

| Tool | URL | What it gives you |
|---|---|---|
| **penaltyblog** (Martin Eastwood) | https://github.com/martineastwood/penaltyblog · https://pypi.org/project/penaltyblog/ | Poisson, Bivariate Poisson, Dixon–Coles and Bayesian goal models plus Elo/Massey/Colley/Pi ratings and data scrapers — the building blocks to roll your own season/tournament sim |
| **soccerdata** (Pieter Robberechts) | https://github.com/probberechts/soccerdata | Python scraper returning tidy pandas DataFrames from Club Elo, FBref, Understat, Football-Data.co.uk and others — feeds the base model |

## Peer-reviewed & preprint tournament-simulation studies (verified)

| Study | Link | Contribution |
|---|---|---|
| **Dixon & Coles (1997)**, *JRSS Series C (Applied Statistics)* 46(2):265–280 | https://doi.org/10.1111/1467-9876.00065 | The bivariate-Poisson base model with a low-score correction and time-weighting that most goal-based sims still build on |
| **Groll, Ley & Schauberger (2018)** — random-forest World Cup 2018 forecast | https://arxiv.org/abs/1806.03208 | Combines a random forest with ranking-based team-ability parameters, then simulates the tournament repeatedly for winning probabilities |
| **Zeileis, Leitner & Hornik** — bookmaker-consensus tournament forecasts | https://www.zeileis.org/news/fifa2022/ · https://ideas.repec.org/p/inn/wpaper/2018-09.html | "Inverse" approach: average de-margined bookmaker odds into a consensus, back out team abilities, then simulate all pairwise matches |

## Minimal Monte Carlo recipe

1. Fit a base match model (start with Dixon–Coles via **penaltyblog**) on recent results/xG.
2. For every remaining fixture, draw a scoreline from the model's goal distribution.
3. Apply the competition rules (points, tie-breakers, or the knockout bracket) to get a final table/winner for that simulated world.
4. Repeat steps 2–3 for **N ≥ 10,000** iterations.
5. Report each outcome's frequency **with the model's own uncertainty**, and **calibrate/backtest** against historical odds (Football-Data.co.uk) before trusting any number.

## Failure modes to expect

- **Stale or mis-specified strength inputs** → confidently wrong percentages.
- **Missing context** (injuries, rotation, weather, fixture congestion) the goal model never saw.
- **Format variance** — short knockouts are luck-heavy; treat single-digit-percent favourites as genuinely uncertain.
- **Reading precision as accuracy** — "28.5%" looks exact but was wrong (see the header examples).

## Honest limits

Public simulators are excellent *explanatory and educational* tools and a fair way to structure your own uncertainty. They are **not** a reliable route to beating efficient betting markets. The examples in the header are the point: top-rated favourites routinely fail to win, and the closing line already prices in most public models.

## Responsible gambling / não é aconselhamento

- **This page is research & education only — it is not betting advice (não é aconselhamento de apostas).**
- The average bettor loses over time; simulation percentages do not change that.
- Brazil: **CVV — 188** (free, 24h) · https://www.cvv.org.br/ · Set limits, treat any stake as entertainment spend, and seek help if betting stops being fun.

---

**Keywords (EN):** football/soccer tournament simulation, Monte Carlo forecasting, season simulation, title/relegation probabilities, Dixon–Coles, Poisson, Elo, SPI, Opta supercomputer, bookmaker consensus model, responsible gambling.
**Palavras-chave (PT):** simulação de campeonato de futebol, Monte Carlo, probabilidade de título e rebaixamento, modelo de Poisson/Dixon–Coles, Elo, previsão de Copa do Mundo, jogo responsável, não é aconselhamento de apostas.
