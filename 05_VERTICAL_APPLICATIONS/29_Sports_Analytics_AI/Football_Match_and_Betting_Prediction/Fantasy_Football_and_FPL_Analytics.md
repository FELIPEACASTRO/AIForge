# Fantasy Football (FPL) Analytics & Prediction

> Authoritative, worldwide index of **Fantasy Premier League (FPL)** and adjacent **fantasy football (futebol fantasy)** prediction — the official FPL API, community wrappers, historical datasets, **expected-points (xP / pontos esperados)** models, mixed-integer **optimizers/solvers**, live-rank & price-change tools, and other ecosystems (UEFA, Dream11, Sorare, DFS, 🇧🇷 **Cartola FC**). Fantasy is **prediction-heavy but mostly a skill game**, not bookmaker betting — it overlaps match prediction (minutes, goals, xG) without a house margin. Every repo/API/tool below was checked for existence + status. Built for **research & education (pesquisa e educação — data science / ML)**, current 2024–2026.

> ⚠️ **Mostly skill, not gambling — but money can be involved.** Core **FPL and UEFA Fantasy are free** season-long games (no stake, no house edge), so they are excellent, *low-harm* sandboxes for prediction/ML. **Daily Fantasy Sports (DFS — DraftKings/FanDuel), paid Dream11 contests, and Sorare (crypto cards)** *do* involve real money and can cause harm; treat those like betting. Research & education only — **not investment or betting advice (não é dica de aposta).** If gambling stops being fun, get help: [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) · [Gambling Therapy](https://www.gamblingtherapy.org/) · 🇧🇷 [Jogo Responsável (SPA/MF)](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) · 🇧🇷 CVV **188**.

**Where this fits:** for match-outcome models (Poisson/Dixon-Coles/Elo/xG/ML/DL), odds/value/CLV, datasets & scrapers, and the classic papers, see the sibling pages → [Open-Source Tools & Libraries](./Open_Source_Tools_and_Libraries.md) · [Global Football Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md) · [Innovative Models & Deep Learning](./Innovative_Models_and_Deep_Learning.md). This page is the **fantasy-specific** layer built on top of that stack.

---

## 0) Why fantasy is a clean prediction problem

Unlike 1X2 betting, FPL has **no bookmaker overround (sem margem da casa)** and a public, machine-readable scoring rule — so your only adversary is *other managers* and *your own forecast error*. The core quantity is **expected points (xP)** for each player over the next 1–N gameweeks (GW), which decomposes into predictable sub-problems:

| Sub-problem | What you predict | Typical signal / feature |
|---|---|---|
| **Minutes / rotation (minutagem)** | P(starts), P(plays ≥60′) | line-ups, injuries, congestion, manager tendencies |
| **Attacking returns** | goals, assists | **xG / xA** (Understat, FBref/StatsBomb), shots, xGChain |
| **Clean sheets / defence** | team goals conceded | team strength, opponent xG, **Dixon-Coles / Poisson λ** |
| **Defensive contribution (2025/26)** | CBIT/CBIRT threshold hits | tackles+interceptions+clearances+blocks (+recoveries) |
| **Bonus (BPS)** | 0/1/2/3 bonus points | FPL's Bonus Points System components |
| **Fixture difficulty (FDR)** | ease of the run | team strength ratings, Elo, upcoming schedule |
| **Price change (variação de preço)** | rise/fall tonight | net transfers vs FPL's ownership-based threshold |

`xP = P(play) × [ minutes pts + goals·xG-weight + assists·xA-weight + CS prob·CS-pts + E(bonus) + E(defcon) − E(cards) ]`. Captaincy is then simply **argmax xP × 2**; transfers/chips maximise multi-GW xP **minus** the 4-pt hit for extra transfers.

---

## 1) The official FPL API (undocumented but open)

Base URL `https://fantasy.premierleague.com/api/`. Returns JSON; **no official documentation** (community reverse-engineered) and **no published rate limit** — but aggressive polling of per-manager endpoints across millions of IDs can trip **Cloudflare / temporary IP blocks**, so **cache `bootstrap-static`, throttle, and send a browser `User-Agent`**. Most endpoints are public (GET, no auth); `my-team/` requires login.

| Endpoint (path after `/api/`) | Returns | Auth? |
|---|---|---|
| `bootstrap-static/` | **Master dump**: all players (`elements`, incl. `ep_this`/`ep_next` = FPL's own xP), teams (with FDR strength), positions, gameweeks (`events`), game settings | no |
| `fixtures/` (`?event=` , `?future=1`) | Every fixture, kickoff, FDR difficulty, stats once played | no |
| `element-summary/{player_id}/` | One player's per-GW **history**, past seasons, upcoming fixtures | no |
| `event/{gw}/live/` | Live per-player points & stat breakdown for a GW | no |
| `entry/{manager_id}/` · `.../history/` | Manager profile; season & past-season history | no |
| `entry/{manager_id}/event/{gw}/picks/` | A manager's XI, bench, captain, chip for a GW | no |
| `leagues-classic/{league_id}/standings/` (`?page_standings=`) | Mini-league table (paginated) | no |
| `dream-team/{gw}/` · `event-status/` · `team/set-piece-notes/` | GW dream team; data-update status; set-piece takers | no |
| `my-team/{manager_id}/` | Your live squad, bank, free transfers, chips | **yes (login)** |

Minimal, verified pull (no wrapper needed):

```python
import requests
BASE = "https://fantasy.premierleague.com/api"
H = {"User-Agent": "Mozilla/5.0"}                       # be polite; cache this call
boot = requests.get(f"{BASE}/bootstrap-static/", headers=H).json()
players, teams, gws = boot["elements"], boot["teams"], boot["events"]
# FPL's own naive expected points (baseline to beat):
naive_xp = {p["web_name"]: float(p["ep_next"]) for p in players}
pid = players[0]["id"]
hist = requests.get(f"{BASE}/element-summary/{pid}/", headers=H).json()  # per-GW form + fixtures
```

---

## 2) Community API wrappers & historical datasets

Don't hand-roll everything — mature wrappers and ready-made historical CSVs exist. **Verify freshness**: `vaastav`'s live weekly updates stopped after 2024-25 (now ~3 releases/yr), so pair it with the live API or a maintained fork for the current season.

| Tool / dataset | Purpose | Free/Paid | Link |
|---|---|---|---|
| **amosbastian/fpl** | Async Python wrapper (aiohttp) for the whole FPL API; login support | Free (OSS) | [github](https://github.com/amosbastian/fpl) · [docs](https://fpl.readthedocs.io/) |
| **jeppe-smith/fpl-api** | TypeScript/JS wrapper for the same endpoints | Free (OSS) | [github](https://github.com/jeppe-smith/fpl-api) |
| **pandas-fpl (177arc)** | Loads the FPL API straight into pandas DataFrames (built on `amosbastian/fpl`) | Free (OSS) | [PyPI](https://pypi.org/project/pandas-fpl/) · [github](https://github.com/177arc/pandas-fpl) |
| **amosbastian/FPLbot** | Reddit bot for /r/FantasyPL: price changes + player-vs-player/team comparisons (FPL API + Understat) | Free (OSS, MIT) | [github](https://github.com/amosbastian/FPLbot) |
| **vaastav/Fantasy-Premier-League** | **The** historical FPL dataset: `cleaned_players.csv`, per-GW `gws/`, per-player folders, per-season **`understat/` xG/xA**, 2016-17 → 2024-25 | Free (OSS) | [github](https://github.com/vaastav/Fantasy-Premier-League) · [data dictionary](https://github.com/vaastav/Fantasy-Premier-League/blob/master/DATA_DICTIONARY.md) |
| **olbauday/FPL-Core-Insights** | Newer combined dataset incl. **2025/26**: fuses FPL API + match stats + team **Elo** + cups/friendlies/Euro coverage, aligned by FPL IDs | Free (OSS) | [github](https://github.com/olbauday/FPL-Core-Insights) |

> ⚠️ **`ep_this`/`ep_next` caveat:** these FPL-provided "expected points" and `vaastav`'s scraped `xP` are weak, partly *post-hoc* baselines — fine as a floor, not a target. Serious models rebuild xP from minutes + Understat xG/xA (see §4).

---

## 3) Predicted-points (xP) services — commercial & free

The "big four" projection services (FPL Review, Hub, Scout, Fix) below are what most competitive managers use; treat their numbers as **strong baselines to benchmark your own model against** (they roughly rival open ML, see §4). Most are **freemium**: free tables, paid deep tools/multi-GW planners.

| Service | What it gives | Free/Paid | Link |
|---|---|---|---|
| **FPL Review** (*Massive Data*) | ML+stats EV up to **14 GW ahead**; exports a projections CSV that feeds solvers (§5) | Freemium (free "Free Planner"; paid Massive Data) | [fplreview.com](https://fplreview.com/) · [model docs](https://docs.fplreview.com/the-model/projections/massive-data-model/) |
| **Fantasy Football Hub** | AI player projections, My-Team rater, fixture analyser | Paid (premium) | [fantasyfootballhub.co.uk/predictions](https://www.fantasyfootballhub.co.uk/predictions) |
| **Fantasy Football Scout** | Opta-powered stats tables, predicted line-ups, price predictor, **ICT Index** | Freemium (paid *Members*) | [fantasyfootballscout.co.uk](https://www.fantasyfootballscout.co.uk/) |
| **Fantasy Football Fix** | AI predictions, algorithmic team optimiser, price predictor | Freemium | [fantasyfootballfix.com](https://www.fantasyfootballfix.com/) |
| **FPL Form** | Free predicted-points (xFPL) table + **CSV export**, planner, price tool | **Free** | [fplform.com/fpl-predicted-points](https://fplform.com/fpl-predicted-points) |
| **FPL Copilot** | Free xPts table + explainer content | Free | [fplcopilot.com/expected-points](https://fplcopilot.com/expected-points) |
| **Fantasy Football Pundit** | Free points predictor | Free | [fantasyfootballpundit.com/fpl-points-predictor](https://www.fantasyfootballpundit.com/fpl-points-predictor/) |

> **Benchmarking claim, sourced:** the 2025 **OpenFPL** paper (position-specific ensembles trained only on public FPL + Understat data, 2020-21→2023-24) shows that, tested prospectively on 2024-25, an open-source model **matches a leading commercial service** and **surpasses it for high-return players (>2 pts)** — holding across one-, two-, and three-GW horizons — i.e. the commercial edge is real but *reproducible with open ML* ([arXiv:2508.09992](https://arxiv.org/abs/2508.09992)).

---

## 4) Open-source ML xP models (build your own)

These are the transparent, reproducible alternatives to the paid services — great for learning feature engineering, minutes classification, and position-specific regression.

| Repo / paper | Approach | Free/Paid | Link |
|---|---|---|---|
| **daniegr/OpenFPL** | **Position-specific ensemble models** on public FPL+Understat (2020-21→2023-24); matches a leading commercial service out-of-sample on 2024-25, and beats it for high-return (>2-pt) players | Free (OSS, Jupyter/MIT) | [github](https://github.com/daniegr/OpenFPL) · [paper](https://arxiv.org/abs/2508.09992) |
| **saheedniyi02/fpl-ai** | Two-stage: classifier for *starts* + regressor for *points*; FPL API + `vaastav` historical data | Free (OSS) | [github](https://github.com/saheedniyi02/fpl-ai) |
| **daniel-mehta/FPL-Expected-Points** | Side-by-side **Random Forest xP** vs a hand-built statistical xP model | Free (OSS) | [github](https://github.com/daniel-mehta/FPL-Expected-Points) |
| **ritviyer/FPL-Team-Prediction** | XGBoost / linear / RF points models + LP squad selection | Free (OSS) | [github](https://github.com/ritviyer/FPL-Team-Prediction) |
| **GitHub topic hubs** | Browse living lists of FPL analytics projects | Free | [topics/fpl-analysis](https://github.com/topics/fpl-analysis) · [topics/fantasy-premier-league](https://github.com/topics/fantasy-premier-league) |

**Modelling notes.** Minutes is the highest-leverage prediction (a benched star scores ~0) — model it as a classifier and multiply through. Use **Understat xG/xA** (already in `vaastav`'s per-season `data/<season>/understat/`) rather than raw goals to cut variance. Score forecasts with **RMSE / MAE on points** and, for captaincy, **hit-rate of argmax**. Never train on future GWs — split by time (evita vazamento / leakage), exactly as in the [match-model page](./Innovative_Models_and_Deep_Learning.md).

---

## 5) Optimizers & solvers (squad = an integer program)

Once you have an xP grid, **picking the best 15 within £100m, 3-per-club, and formation rules is a mixed-integer linear program (MILP)**; multi-GW transfer/chip planning adds the 4-pt hit and free-transfer carry as constraints. This is the single biggest reason FPL is a genuine optimization/OR teaching case.

| Tool | What it does | Free/Paid | Link |
|---|---|---|---|
| **solioanalytics/open-fpl-solver** (was *sertalpbilal/FPL-Optimization-Tools*) | The reference **MILP** solver: `pandas` + `sasoptpy` + **HiGHS** (`highspy`); multi-GW, chips, sensitivity; **browser/Colab** version; consumes a projections CSV (e.g. FPL Review) | Free (Apache-2.0) | [github](https://github.com/solioanalytics/open-fpl-solver) · [author/tutorials](https://sertalpbilal.com/) |
| **ewenme/fplinear** | Lightweight **R** LP optimiser for FPL points | Free (OSS) | [github](https://github.com/ewenme/fplinear) |
| **FPL Optimized** | Web xP calculator + optimizer (no code) | Free | [fploptimized.com](https://fploptimized.com/) |

**Recipe (receita):** FPL Review / OpenFPL xP CSV → MILP: maximise `Σ xP[player,gw]·pick − 4·(extra transfers)` s.t. budget, 2 GK/5 DEF/5 MID/3 FWD, ≤3 per club, valid XI+bench, chip logic → inspect solution + **sensitivity** (which picks are robust to xP noise). This is `socceraction`-grade OR, but for team selection instead of action value.

---

## 6) Live rank, effective ownership & price-change models

| Tool | Purpose | Free/Paid | Link |
|---|---|---|---|
| **LiveFPL** | **Live overall rank**, predicted bonus, **effective ownership (EO)**, auto-sub projection, tier averages | **Free** | [livefpl.net](https://www.livefpl.net/) · [rank](https://www.livefpl.net/rank) |
| **LiveFPL Price Predictor** | Overnight rise/fall predictions (net-transfer thresholds) | Free | [livefpl.net/prices](https://www.livefpl.net/prices) |
| **FPL Statistics** | Classic **price-change predictor** + ownership %, refreshed ~½-hourly | Free | [fplstatistics.com](https://www.fplstatistics.com/) |
| **FPL Form / FFScout price tools** | Alternative price-change predictors (cross-check them) | Freemium | [fplform.com/fpl-price-change](https://www.fplform.com/fpl-price-change) · [FFScout](https://www.fantasyfootballscout.co.uk/fpl/price-predictions/) |

**Price-change modelling.** FPL prices behave like a **stock market**: a player rises/falls when *net transfers in/out* crosses an ownership-scaled, undisclosed threshold (changes run ~02:30 UK). Modelling it = estimating that threshold from transfer velocity — a nice time-series / hidden-parameter exercise, and it protects team value, not profit.

**Effective ownership (EO)** is the fantasy analogue of **market-implied probability**: `EO = %owned + %captaining` (× chips). Your rank moves with `(your holding − EO) × player points`, so template-vs-differential decisions are a variance/expected-value trade-off just like staking in betting — see [Bet Selection & Staking](./Bet_Selection_Staking_and_High_Odds_Analysis.md) for the EV/variance intuition.

---

## 7) 2025/26 rules that change the model (stay current)

Scoring/chip changes alter the xP function every season — hard-code nothing. Key **2025/26** changes ([Premier League](https://www.premierleague.com/en/news/4373187/whats-new-for-202526-changes-in-fantasy-premier-league) · [RotoWire](https://www.rotowire.com/soccer/article/fpl-rule-changes-2025-26-season-assists-defensive-contributions-fantasy-premier-league-double-chips-94629)):

- **Defensive Contribution points** are now a full feature: **DEF** get +2 for **10 CBIT** (clearances/blocks/interceptions/tackles); **MID/FWD** get +2 for **12 CBIRT** (adds ball **recoveries**). Rewards low-xG, high-work players — a new modelling target.
- **Two half-season chip sets** (8 total): Wildcard, Free Hit, Triple Captain, Bench Boost in *each* half; the first set must be used before the **GW19** deadline and cannot be carried over.
- **Assistant Manager chip removed** (it existed only in 2024/25) — drop it from any historical feature set.

---

## 8) Other fantasy ecosystems (adjacent prediction problems)

| Platform | What it is | Data / API | Money? | Link |
|---|---|---|---|---|
| **UEFA Fantasy** (UCL/UEL/UECL) | Official free season game across UEFA club comps; 100m budget, matchday transfers | Community-scraped `gaming.uefa.com` JSON (no official docs) | No (free) | [gaming.uefa.com/en/uclfantasy](https://gaming.uefa.com/en/uclfantasy) |
| **DFS — DraftKings / FanDuel** | US **daily** fantasy soccer; $50k salary cap, 8-player rosters (2F/2M/2D/GK/UTIL); paid entry | Optimizer ecosystems (RotoWire, FantasyCruncher) | **Yes** | [draftkings.com/fantasy-soccer](https://www.draftkings.com/fantasy-soccer) · [RotoWire optimizer](https://www.rotowire.com/daily/soccer/optimizer.php) |
| **Dream11** 🇮🇳 | India's largest fantasy platform (cricket/football/…); skill-game legally, but… | Private (no public API) | **Was paid** | [dream11.com](https://www.dream11.com/) · [Wikipedia](https://en.wikipedia.org/wiki/Dream11) |
| **Sorare** | **Blockchain** fantasy (migrating Ethereum→Solana, 2025): own/trade licensed **NFT** player cards, 5-a-side lineups scored on real performance | Public GraphQL API | **Yes (crypto)** | [sorare.com](https://sorare.com/) |

> 🇮🇳 **India regulatory note (currency):** the **Promotion and Regulation of Online Gaming Act, 2025** banned real-money online games; **Dream11 paused all paid contests in Aug 2025**, moving to free-to-play with non-cash rewards. Fantasy is "skill" in most Indian states, but **paid formats are now restricted — check local law.** See the [Asia & India page](./Asia_and_India_Football_Data_and_Markets.md).

---

## 9) 🇧🇷 Cartola FC — fantasy do futebol brasileiro

**Cartola FC** (by Globo) is Brazil's dominant fantasy game for the **Brasileirão Série A**: pick 11 + reserves + *capitão* under a **cartoletas** budget; players score via **scouts** (gols, assistências, desarmes, defesas, etc.). It is **free** to play (paid "Cartola PRO" only adds features/insights). It exposes an **unofficial but community-documented REST API** (`api.cartola.globo.com` / `api.cartolafc.globo.com`) with endpoints for the market (`/atletas/mercado`), clubs (`/clubes`), rounds (`/partidas`), and market status (`/mercado/status`).

| Tool | Language | Purpose | Link |
|---|---|---|---|
| **vicenteneto/python-cartolafc** | Python | Wrapper for the Cartola FC REST API (market, athletes, leagues) | [github](https://github.com/vicenteneto/python-cartolafc) |
| **henriquepgomide/caRtola** | R / Python | The reference **dataset + predictive models** for Cartola (2014-23): scraping, EDA, xP-style modelling | [github](https://github.com/henriquepgomide/caRtola) |
| **rafaelpierre/cartolator** | Python | Simple scout/athlete puller → CSV | [github](https://github.com/rafaelpierre/cartolator) |
| **DanielSalesS/cartola_etl** · **cartola_data** | Python | ETL / data pipelines for Cartola API data | [etl](https://github.com/DanielSalesS/cartola_etl) · [data](https://github.com/DanielSalesS/cartola_data) |
| **topic hub** | — | Browse all Cartola OSS | [github.com/topics/cartolafc](https://github.com/topics/cartolafc) |

> For **Brasileirão** underlying stats (xG/scouts) to power a Cartola model, use FBref-via-`soccerdata`/`ScraperFC` from the [tools page](./Open_Source_Tools_and_Libraries.md) and results/odds from the Americas page. Cartola is a *free skill game* — a perfect, no-stake ML sandbox for Portuguese-speaking learners (**pesquisa e educação**).

---

## 10) Reality check & responsible-gaming note

- **FPL/UEFA/Cartola are free** — the "risk" is only your rank and time; that makes them the **ideal, low-harm ML playground**. Public projection services already rival open ML, so a home-grown model's edge is small — **do it to learn**, not to "win money."
- **Where money enters** (DFS, paid Dream11 historically, Sorare), the usual truths return: variance is high, **rake/fees replace the vig**, sharks use the same optimizers, and most entrants lose. Same responsible-gaming discipline as betting applies — set limits, never chase (nunca persiga o prejuízo). **This page is not investment or betting advice.**
- **Sorare adds crypto risk** (card price volatility + wallet/market risk) on top of fantasy variance.

| Region | Resource | Contact |
|---|---|---|
| 🌍 Global | [Gambling Therapy](https://www.gamblingtherapy.org/) | Online chat/forum, multilingual |
| 🇬🇧 UK | [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) | National Gambling Helpline **0808 8020 133** (24/7) |
| 🇧🇷 Brazil | [Jogo Responsável — SPA/Ministério da Fazenda](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) | Apoio emocional **CVV 188** |

---

## Related in AIForge
- [Open-Source Tools & Libraries](./Open_Source_Tools_and_Libraries.md) · [Global Football Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md) · [Innovative Models & Deep Learning](./Innovative_Models_and_Deep_Learning.md) · [Bet Selection & Staking](./Bet_Selection_Staking_and_High_Odds_Analysis.md) · [Asia & India](./Asia_and_India_Football_Data_and_Markets.md) · Americas · Parent vertical: [`../`](../) (Sports Analytics AI)

**Sources:** fantasy.premierleague.com/api (bootstrap-static/, fixtures/, element-summary/, event/{gw}/live/, entry/, leagues-classic/, dream-team/, event-status/, team/set-piece-notes/) · github.com/amosbastian/fpl · fpl.readthedocs.io · github.com/jeppe-smith/fpl-api · pypi.org/project/pandas-fpl · github.com/177arc/pandas-fpl · github.com/vaastav/Fantasy-Premier-League · github.com/olbauday/FPL-Core-Insights · fplreview.com · docs.fplreview.com · fantasyfootballhub.co.uk · fantasyfootballscout.co.uk · fantasyfootballfix.com · fplform.com · fplcopilot.com · fantasyfootballpundit.com · arxiv.org/abs/2508.09992 · github.com/daniegr/OpenFPL · github.com/saheedniyi02/fpl-ai · github.com/daniel-mehta/FPL-Expected-Points · github.com/ritviyer/FPL-Team-Prediction · github.com/solioanalytics/open-fpl-solver · github.com/ewenme/fplinear · fploptimized.com · livefpl.net · fplstatistics.com · github.com/amosbastian/FPLbot · premierleague.com (2025/26 rules) · rotowire.com · gaming.uefa.com · draftkings.com/fantasy-soccer · rotowire.com/daily/soccer · dream11.com · en.wikipedia.org/wiki/Dream11 · sorare.com · github.com/vicenteneto/python-cartolafc · github.com/henriquepgomide/caRtola · github.com/rafaelpierre/cartolator · github.com/DanielSalesS/cartola_etl · begambleaware.org · gamcare.org.uk · gamblingtherapy.org · gov.br/fazenda (SPA)

**Keywords:** fantasy premier league API, FPL bootstrap-static, expected points xP, FPL predicted points, FPL Review Massive Data, OpenFPL, FPL optimization MILP solver, open-fpl-solver HiGHS, vaastav Fantasy-Premier-League dataset, LiveFPL live rank, FPL price change predictor, effective ownership, captaincy EV, fixture difficulty rating FDR, defensive contribution 2025/26, UEFA Fantasy, Dream11, Sorare NFT fantasy, daily fantasy sports DFS, Cartola FC API, análise fantasy futebol, pontos esperados, previsão de escalação, otimização de time, jogo responsável, futebol fantasy brasileiro.
