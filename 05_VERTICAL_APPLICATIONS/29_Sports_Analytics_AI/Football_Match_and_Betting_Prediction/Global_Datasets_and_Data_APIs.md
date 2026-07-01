# Global Football Datasets & Data APIs (All Countries)

> Authoritative, worldwide index of **where to get football (soccer / futebol) data** for match & betting prediction — historical results, odds, event/tracking data, xG, live APIs, and reference/meta sources — with real URLs, free-vs-paid marks, and multi-region coverage (Europe, 🇧🇷 Brazil, South America, Africa, Asia, North America, international/World Cup). Built for **research & education (data science / ML)**, current 2024–2026.

> ⚠️ **Research & education only — not betting advice.** Betting markets are **highly efficient**: from a sample of **397,935** Pinnacle football games the closing line correlated with observed outcomes at **r² ≈ 0.997** ([Trademate Sports analysis](https://tradematesports.medium.com/closing-line-the-most-important-metric-in-sports-trading-58e56cdb4458)). Most bettors lose money over time; **beating the closing line consistently is extremely hard**. Nothing here is a tip or a system. If gambling is a problem, get help: [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) · [Gambling Therapy](https://www.gamblingtherapy.org/) · 🇧🇷 [Jogo Responsável (SPA/MF)](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) · 🇧🇷 CVV **188**.

**How to choose:** free CSV history (football-data.co.uk) for quick baselines → open event/tracking data (StatsBomb, Metrica, SkillCorner) for advanced features → live/reference APIs (API-Football, football-data.org) for pipelines → paid providers (Opta, Sportradar, StatsBomb, Wyscout) for production/coverage depth. Sibling page: [Kaggle Football Datasets](./Kaggle_Football_Datasets_and_Competitions.md).

---

## 1) Results + Odds (historical CSV / SQLite datasets)

The workhorses for match-outcome and value-betting research. Odds columns let you back-test against the market (and see how efficient it is).

| Source | Coverage | Odds? | Free/Paid | URL |
|---|---|---|---|---|
| **football-data.co.uk** | 22 EU divisions back to 1993/94 + "extra" leagues (🇦🇷 Argentina, 🇧🇷 Brazil, 🇨🇳 China, 🇯🇵 Japan, 🇲🇽 Mexico, 🇺🇸 USA/MLS, Russia, Nordics, etc.) | ✅ up to 10 bookmakers; **opening + closing** since 2019/20; O/U + Asian handicap | **Free** (CSV) | [data.php](https://www.football-data.co.uk/data.php) · [Europe](https://www.football-data.co.uk/downloadm.php) · [extra](https://www.football-data.co.uk/all_new_data.php) |
| **Kaggle: Club Football Match Data (2000–2025)** (adamgbor) | 27 countries / 42 leagues; merges football-data.co.uk results + **ClubElo** ratings + pre-match odds | ✅ pre-match | **Free** | [kaggle](https://www.kaggle.com/datasets/adamgbor/club-football-match-data-2000-2025) |
| **Kaggle: European Soccer Database** (hugomathien) | Canonical SQLite DB: 25k+ matches (2008–2016), 11 EU countries, team/player FIFA attributes | ✅ up to 10 providers | **Free** | [kaggle](https://www.kaggle.com/datasets/hugomathien/soccer) |
| **Beat The Bookie: Odds Series** (Kaunitz et al.) | Football odds time series from **32 bookmakers** across ~1,000 leagues, 2000–2015 | ✅ time series | **Free** | [kaggle](https://www.kaggle.com/datasets/austro/beat-the-bookie-worldwide-football-dataset) · [code](https://github.com/Lisandro79/BeatTheBookie) · [paper](https://arxiv.org/abs/1710.02824) |
| **OddsPortal** | Odds comparison + historical odds/movement, hundreds of bookmakers, worldwide leagues | ✅ opening/closing, many books | **Free site** (scrape ⚠️ ToS/JS-rendered) | [results](https://www.oddsportal.com/football/results/) · scraper [OddsHarvester](https://github.com/jordantete/OddsHarvester) |
| **datahub.io — Football** / footballcsv | Top-5 EU leagues + World Cup, auto-updated daily, CSV/JSON | ➖ | **Free** (public-domain-ish) | [datahub](https://datahub.io/collections/football) · [github](https://github.com/datasets/football-datasets) · [footballcsv](https://footballcsv.github.io/) |

---

## 2) Event & Tracking data (advanced — for xG, xT, pitch control, pressing)

Event data = every on-ball action with x/y; tracking data = all 22 players + ball at 10–25 Hz. The open sets below are the standard teaching corpora.

| Source | Type | Coverage | Free/Paid | URL |
|---|---|---|---|---|
| **StatsBomb Open Data** | Event + **360** (freeze-frames) | Men's/Women's World Cups, UEFA Euro, Champions League finals, La Liga (Messi years), FA WSL, Indian Super League, more | **Free** (attribution required) | [github](https://github.com/statsbomb/open-data) · py [statsbombpy](https://github.com/statsbomb/statsbombpy) · R [StatsBombR](https://github.com/statsbomb/StatsBombR) |
| **Metrica Sports Sample Data** | **Tracking + event** | 3 anonymized full matches (CSV + FIFA-EPTS + JSON) | **Free** | [github](https://github.com/metrica-sports/sample-data) |
| **SkillCorner Open Data** | Broadcast **tracking** (computer vision) | 10 matches (Australian A-League 2024/25) + season aggregates (physical, off-ball runs, passing) | **Free** (w/ PySport) | [github](https://github.com/SkillCorner/opendata) |
| **StatsBomb** (full) | Event + 360 | Global leagues, licensed | **Paid** (free tier via open-data) | [statsbomb.com](https://statsbomb.com/) |
| **Stats Perform / Opta** | Event (official) | Official data partner of Premier League, Bundesliga, Serie A, La Liga, MLS + 3,900+ competitions | **Paid / enterprise** | [Opta data](https://www.statsperform.com/products/opta-data/) |
| **Sportradar** | Live + event (XY coords) | 650+ competitions, tiered coverage, Extended API 100+ data points | **Paid** (trial) | [dev portal](https://developer.sportradar.com/soccer/reference/soccer-api-overview) |
| **Wyscout** (Hudl) | Event + video scouting | 1,000+ competitions, wide global coverage | **Paid** | [hudl.com/wyscout](https://www.hudl.com/products/wyscout) |

---

## 3) xG / advanced-stat sources (free, via scraping)

| Source | What | Coverage | Free/Paid | URL |
|---|---|---|---|---|
| **Understat** | Shot-level **xG / xA** | Top-5 EU leagues + Russian PL, since 2014/15 | **Free** (scrape) | [understat.com](https://understat.com/) · dataset [douglasbc](https://github.com/douglasbc/scraping-understat-dataset) · [understatapi](https://pypi.org/project/understatapi/) |
| **FBref** (StatsBomb-powered) | Team/player stats, xG, match logs | Global — 40+ leagues incl. Brasileirão, MLS, J-League, Liga MX | **Free** (scrape / via soccerdata) | [fbref.com](https://fbref.com/) |
| **FotMob / Sofascore** | Ratings, xG, momentum | Global | **Free site** (unofficial API/scrape) | via [soccerdata](https://github.com/probberechts/soccerdata) / [ScraperFC](https://github.com/oseymour/ScraperFC) |

---

## 4) Live / reference APIs (build pipelines)

| API | Coverage | Free tier | Paid from | URL |
|---|---|---|---|---|
| **API-Football** (api-sports.io) | **1,100+** leagues & cups: fixtures, standings, events, lineups, players, **pre-match odds**, stats | ✅ **100 req/day**, all endpoints | **$19/mo** (Pro, 7,500 req/day; up to 1.5M/day on top tier) | [api-football.com](https://www.api-football.com/) · [docs](https://www.api-football.com/documentation-v3) |
| **football-data.org** | 12 free comps incl. 🇧🇷 **Brasileirão Série A**, EPL, La Liga, UCL, World Cup, Euro | ✅ **10 calls/min**, fixtures/results/tables | from **€29/mo** (Deep Data) · €49/mo (Standard); stats/odds add-ons | [football-data.org](https://www.football-data.org/) · [coverage](https://www.football-data.org/coverage) · [pricing](https://www.football-data.org/pricing) |
| **SportMonks** | 2,500+ leagues; live, odds, stats; **Predictions add-on** (AI-powered fixture probabilities) | 🆓 trial (2 free leagues) | €29 / €99 / €249 | [sportmonks.com](https://www.sportmonks.com/football-api/) |
| **TheSportsDB** | Teams, events, players, scores; crowd-sourced | ✅ **Free** (public), premium for higher limits | Patreon premium (~$9/mo) | [thesportsdb.com](https://www.thesportsdb.com/free_sports_api) |
| **Sportradar** | 650+ competitions, official-grade, XY events | 🆓 trial keys | Enterprise | [developer.sportradar.com](https://developer.sportradar.com/soccer/reference/soccer-api-overview) |
| **Sofascore** | Global live/stats | ➖ (no official public API) | Unofficial scrape only | via [ScraperFC](https://github.com/oseymour/ScraperFC) |

---

## 5) Reference / meta (ratings, values, fixtures)

| Source | What | Coverage | Free/Paid | URL |
|---|---|---|---|---|
| **ClubElo** | Daily **club Elo** ratings + match win/draw/loss probabilities | European clubs, daily since 1939 | **Free API** (CSV) | [clubelo.com/API](http://clubelo.com/API) |
| **World Football Elo Ratings** | **National-team** Elo (margin, importance, home adv.) | 200+ national teams | **Free site** | [eloratings.net](https://eloratings.net/) · [Kaggle mirror](https://www.kaggle.com/datasets/saifalnimri/international-football-elo-ratings) |
| **FIFA / UEFA rankings** | Official world/continental rankings | National teams & clubs | **Free** | [FIFA ranking Kaggle](https://www.kaggle.com/datasets/tadhgfitzgerald/fifa-international-soccer-mens-ranking-1993now) |
| **Transfermarkt** (via dcaribou) | Squads, **market values**, transfers, appearances | 79k+ games, 37k+ players, weekly refresh | **Free** (scraped mirror) | [Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores) · [github](https://github.com/dcaribou/transfermarkt-datasets) |
| **OpenFootball** | Public-domain fixtures/results (Football.TXT + JSON) | Worldwide clubs & internationals, **no API key** | **Free (public domain)** | [org](https://github.com/openfootball) · [leagues](https://github.com/openfootball/leagues) · [football.json](https://github.com/openfootball/football.json) · [world](https://github.com/openfootball/world) |

---

## 6) Tooling — scrapers & standardizers (Python/R, all free/OSS)

| Tool | Lang | Sources it unifies | URL |
|---|---|---|---|
| **soccerdata** | Python | ClubElo, ESPN, FBref, Football-Data.co.uk, Sofascore, SoFIFA, Understat, WhoScored | [github](https://github.com/probberechts/soccerdata) |
| **worldfootballR** | R | FBref, Transfermarkt, Understat (fotmob dropped; **repo archived Sep 2025, read-only**) | [github](https://github.com/JaseZiv/worldfootballR) |
| **ScraperFC** | Python | Capology, ClubELO, FBref, Sofascore, Transfermarkt, Understat | [github](https://github.com/oseymour/ScraperFC) |
| **kloppy** (PySport) | Python | Standardizes event/tracking from StatsBomb, Metrica, Opta, Sportec, Wyscout, Tracab, etc. | [github](https://github.com/PySport/kloppy) |
| **transfermarkt-datasets** | Python/SQL | Full Transfermarkt ETL pipeline | [github](https://github.com/dcaribou/transfermarkt-datasets) |

---

## 7) Regional / country coverage

### 🇪🇺 Europe (top leagues + lower divisions)
| Data | Where | Free/Paid |
|---|---|---|
| EPL / La Liga / Serie A / Bundesliga / Ligue 1 + 2nd/3rd tiers | football-data.co.uk (22 divisions), Understat (xG top-5), FBref, OpenFootball | Free |
| Full history 2008–16 (11 countries) | European Soccer Database (hugomathien) | Free |
| Live/odds pipelines | API-Football, football-data.org, SportMonks | Freemium/Paid |

### 🇧🇷 Brazil (Brasileirão / futebol brasileiro)
| Data | Where | Free/Paid |
|---|---|---|
| Campeonato Brasileiro Série A (matches, goals, **cartões**) 2003–2024 | [adaoduque/campeonato-brasileiro](https://www.kaggle.com/datasets/adaoduque/campeonato-brasileiro-de-futebol) · [github](https://github.com/adaoduque/Brasileirao_Dataset) | Free |
| Brasileirão + Libertadores + Sudamericana + Copa do Brasil | [ricardomattos05](https://www.kaggle.com/datasets/ricardomattos05/jogos-do-campeonato-brasileiro) · [Brazilian-Soccer-Data](https://github.com/ricardo-mattoss/Brazilian-Soccer-Data) | Free |
| Brazil results **with odds** | football-data.co.uk (extra leagues) | Free |
| Brasileirão live (free tier) | [football-data.org](https://www.football-data.org/coverage) | Free tier ✅ |
| Full Série A/B live, odds, xG | API-Football, SportMonks, FBref | Freemium/Paid |

### 🌎 South America (CONMEBOL — Libertadores, Sudamericana, Copa América)
| Data | Where | Free/Paid |
|---|---|---|
| Copa Libertadores / Sudamericana fixtures & results | [OpenFootball](https://github.com/openfootball/leagues) · ricardomattos05 (BR clubs) | Free |
| Argentina/Brazil league results + odds | football-data.co.uk | Free |
| Copa América internationals | [martj42 (1872–2026)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | Free |

### 🌍 Africa (CAF — AFCON, CAF Champions League)
| Data | Where | Free/Paid |
|---|---|---|
| Africa Cup of Nations match stats | [mohammedessam97/africa-cup-of-nations](https://www.kaggle.com/datasets/mohammedessam97/africa-cup-of-nations) | Free |
| African national-team results 2010–2024 | [oussamalariouch](https://www.kaggle.com/datasets/oussamalariouch/african-national-football-from-2010-2024) | Free |
| AFCON/CAF via internationals + APIs | martj42, API-Football, SportMonks | Free/Paid |

### 🌏 Asia (AFC — J-League, K-League, Asian Cup, ACL)
| Data | Where | Free/Paid |
|---|---|---|
| 🇯🇵 J-League results **with odds** | football-data.co.uk (Japan) | Free |
| 🇰🇷 K-League, 🇨🇳 CSL, ACL, AFC Asian Cup | API-Football / SportMonks (1,100+ / 2,500+ leagues), FBref | Freemium/Paid |
| Asian internationals (Asian Cup) | martj42 international results | Free |

### 🇺🇸 North America (MLS, Liga MX, CONCACAF)
| Data | Where | Free/Paid |
|---|---|---|
| MLS matches/events/tables | [josephvm/major-league-soccer-dataset](https://www.kaggle.com/datasets/josephvm/major-league-soccer-dataset) | Free |
| USA/MLS + Mexico results **with odds** | football-data.co.uk (USA, Mexico) | Free |
| MLS/Liga MX live, xG | API-Football, FBref | Freemium/Free |

### 🏆 International / World Cup (1930–2026)
| Data | Where | Free/Paid |
|---|---|---|
| FIFA World Cup 1930–**2026** (actively maintained) | [piterfm/fifa-football-world-cup](https://www.kaggle.com/datasets/piterfm/fifa-football-world-cup) | Free |
| Comprehensive WC DB (men's 1930–2022 + women's 1991–2019) | [jfjelstul/worldcup](https://github.com/jfjelstul/worldcup) | Free |
| International results **1872–2026** (~48k matches) | [martj42](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | Free |
| World Cup 2026 fixtures (Football.TXT) | [OpenFootball world-cup](https://github.com/openfootball/worldcup) | Free |

---

## 8) Reality check — why "beating the market" is so hard (read before modeling odds)

- The **closing line** aggregates all sharp money and late news; it is the market's most efficient estimate. Kaunitz et al. (2017) found a strategy could beat *published* odds in back-test, **but bookmakers limited/closed winning accounts**, killing it in practice ([arXiv:1710.02824](https://arxiv.org/abs/1710.02824)).
- Sharp books (e.g. **Pinnacle**) price with low margin + high limits, so their closing lines are near-efficient (**r² ≈ 0.997** vs outcomes across 397,935 games). A realistic skill metric is **Closing Line Value (CLV)** — do your prices repeatedly beat the close? ([Pinnacle: CLV explained](https://www.pinnacle.com/betting-resources/en/educational/what-is-closing-line-value-clv-in-sports-betting)).
- The **bookmaker margin / overround (vig)** means a naive bettor loses on average even with "fair" skill. Treat all of this as **modeling practice**, not income.

**Foundational goal-modeling papers (exact citations):**
- Maher, M.J. (1982). *Modelling association football scores.* **Statistica Neerlandica** 36(3): 109–118. [DOI: 10.1111/j.1467-9574.1982.tb00782.x](https://doi.org/10.1111/j.1467-9574.1982.tb00782.x) — independent-Poisson attack/defence model (the origin of Poisson goal models).
- Dixon, M.J. & Coles, S.G. (1997). *Modelling Association Football Scores and Inefficiencies in the Football Betting Market.* **J. R. Stat. Soc. C (Applied Statistics)** 46(2): 265–280. [DOI: 10.1111/1467-9876.00065](https://doi.org/10.1111/1467-9876.00065) — low-score dependence correction + time-decay weighting; basis of most modern models.
- Kaunitz, L., Zhong, S. & Kreiner, J. (2017). *Beating the bookies with their own numbers — and how the online sports betting market is rigged.* [arXiv:1710.02824](https://arxiv.org/abs/1710.02824).

---

## 9) Responsible gambling (Jogo Responsável) — mandatory

Gambling can cause serious harm. This index exists for **data science / ML research and education only**. Set limits, never chase losses, and seek help if betting stops being fun.

| Region | Resource | Contact |
|---|---|---|
| 🌍 Global | [Gambling Therapy](https://www.gamblingtherapy.org/) | Online chat/forum, multilingual |
| 🇬🇧 UK | [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) | National Gambling Helpline **0808 8020 133** (24/7) |
| 🇧🇷 Brazil | [Jogo Responsável — SPA/Ministério da Fazenda](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) (Portaria SPA/MF nº 1.231/2024) | Apoio emocional **CVV 188** |

---

## Related in AIForge
- [Kaggle Football Datasets & Competitions](./Kaggle_Football_Datasets_and_Competitions.md) · Parent vertical: [`../`](../) (Sports Analytics AI)

**Sources:** football-data.co.uk · Kaggle (hugomathien, adamgbor, austro, martj42, piterfm, josephvm, adaoduque, ricardomattos05, mohammedessam97, oussamalariouch, davidcariboo, saifalnimri, tadhgfitzgerald) · github.com/statsbomb/open-data · github.com/metrica-sports/sample-data · github.com/SkillCorner/opendata · understat.com · fbref.com · github.com/probberechts/soccerdata · github.com/JaseZiv/worldfootballR · github.com/oseymour/ScraperFC · github.com/PySport/kloppy · github.com/dcaribou/transfermarkt-datasets · github.com/openfootball · api-football.com · football-data.org · sportmonks.com · thesportsdb.com · developer.sportradar.com · statsperform.com/products/opta-data · clubelo.com/API · eloratings.net · datahub.io/collections/football · oddsportal.com · arxiv.org/abs/1710.02824 · doi.org/10.1111/j.1467-9574.1982.tb00782.x (Maher 1982) · doi.org/10.1111/1467-9876.00065 (Dixon–Coles 1997) · tradematesports.medium.com · pinnacle.com · begambleaware.org · gamcare.org.uk · gamblingtherapy.org · gov.br/fazenda (SPA)

**Keywords:** football data API, soccer datasets, historical odds, betting market efficiency, closing line value, xG expected goals, StatsBomb open data, tracking data, API-Football, football-data.org, ClubElo, Brasileirão data, MLS dataset, World Cup dataset, Poisson goal model, Dixon-Coles, dados de futebol, previsão de partidas, odds de apostas, jogo responsável, mercado de apostas eficiente.
