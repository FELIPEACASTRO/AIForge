# Europe — Football Data, Leagues & Markets

> Authoritative, regional index of **where to get football (soccer / futebol) data** for match & betting-prediction research across **all of UEFA Europe** — the "Big-5" (England, Spain, Italy, Germany, France), the mid-tier leagues (Portugal, Netherlands, Belgium, Scotland, Turkey, Greece), the Nordics, Central/Eastern Europe, and UEFA club & national competitions — with real URLs, free-vs-paid marks, current 2024–2026. Built for **data science / ML research & education**.

> ⚠️ **Research & education only — NOT betting advice.** European football-betting markets are **highly efficient**: at sharp books like **Pinnacle** the closing line tracks realized outcome frequencies almost perfectly — in a sample of **397,935** Pinnacle football games the closing line correlated with observed outcomes at **r² ≈ 0.997** ([Trademate Sports analysis, Medium](https://tradematesports.medium.com/closing-line-the-most-important-metric-in-sports-trading-58e56cdb4458)). After the bookmaker margin (*overround / vigorish*), **most bettors lose money over time**, and a sustained, provable edge is rare and hard to build. Nothing here is a tip, a "system", or a guarantee. Help: [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) · [Gambling Therapy](https://www.gamblingtherapy.org/) · 🇧🇷 [Jogo Responsável (SPA/MF)](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) · 🇧🇷 CVV **188**.

**How to choose (Brazil-friendly path):** start with free CSV history WITH closing odds ([football-data.co.uk](https://www.football-data.co.uk/)) → add free xG ([Understat](https://understat.com/)) and free advanced stats ([FBref](https://fbref.com/)) → add [StatsBomb open-data](https://github.com/statsbomb/open-data) event data → wrap live/reference data with [API-Football](https://www.api-football.com/) / [football-data.org](https://www.football-data.org/) → escalate to paid providers (Opta / Stats Perform, Hudl StatsBomb, Wyscout) only for coverage depth. Sibling pages: [Global Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md) · [Africa, Middle East & Oceania Data](./Africa_MiddleEast_and_Oceania_Football_Data.md) · [Kaggle Football Datasets](./Kaggle_Football_Datasets_and_Competitions.md) · [Open-Source Tools](./Open_Source_Tools_and_Libraries.md) · [Innovative Models](./Innovative_Models_and_Deep_Learning.md).

---

## 1) European league landscape

The **Big-5** are the deepest-covered leagues (results, odds, xG, event data all available free). Mid-tier and lower divisions are well covered for results + odds via football-data.co.uk, and increasingly for xG/stats via FBref.

| Tier | Leagues (country) | Typical free coverage |
|---|---|---|
| **Big-5** | Premier League 🏴, La Liga 🇪🇸, Serie A 🇮🇹, Bundesliga 🇩🇪, Ligue 1 🇫🇷 | Results + odds (football-data.co.uk), xG (Understat), stats (FBref), event data (StatsBomb where licensed) |
| **Big-5 2nd tiers** | Championship, LaLiga 2, Serie B, 2. Bundesliga, Ligue 2 | Results + odds (football-data.co.uk), stats (FBref) |
| **Mid-tier** | Primeira Liga 🇵🇹, Eredivisie 🇳🇱, Pro League 🇧🇪, Premiership 🏴󠁧󠁢󠁳󠁣󠁴󠁿, Süper Lig 🇹🇷, Super League 🇬🇷 | Results + odds (football-data.co.uk), stats (FBref) |
| **Nordics** | Allsvenskan 🇸🇪, Eliteserien 🇳🇴, Superliga 🇩🇰, Veikkausliiga 🇫🇮 | Results + odds (football-data.co.uk "extra"), stats (FBref) |
| **Central/Eastern Europe** | Ekstraklasa 🇵🇱, Fortuna Liga 🇨🇿, Super League 🇨🇭, Bundesliga 🇦🇹, SuperLiga 🇷🇸, HNL 🇭🇷, Ukr. Premier League 🇺🇦 | Results (openfootball, FBref); odds partial (football-data.co.uk "extra") |

> **Russia note.** Since late February 2022, UEFA and FIFA have suspended all Russian clubs and national teams from their competitions ([UEFA statement, 28 Feb 2022](https://www.uefa.com/news-media/news/0272-148df1faf082-6e50b5ea1f84-1000--fifa-uefa-suspend-russian-clubs-and-national-teams-from-a/)); the suspension has been **renewed every season since — most recently for 2026/27** (a fifth consecutive campaign). The Russian Premier League still plays domestically, and historical/ongoing RPL results + odds remain in football-data.co.uk "extra" and xG in Understat.

---

## 2) football-data.co.uk — the canonical free EU results + odds archive

The workhorse for European match-outcome and value-betting research: free CSVs with results back to **1993/94**, match stats, and **multi-bookmaker odds** (odds from **2000/01**) including 1X2, Over/Under 2.5, Asian Handicap, and Pinnacle closing prices.

| Page | What | URL |
|---|---|---|
| Main data hub | League index, notes, column legend | [data.php](https://www.football-data.co.uk/data.php) |
| Europe (main leagues) | Big-5 + 2nd/3rd tiers, all seasons, CSV | [downloadm.php](https://www.football-data.co.uk/downloadm.php) |
| "Extra" leagues | Nordics, Poland, Russia, Switzerland, Greece, etc. (single multi-season files) | [all_new_data.php](https://www.football-data.co.uk/all_new_data.php) |

**Key column groups** (see [notes.txt](https://www.football-data.co.uk/notes.txt) for the full legend):

- **Result / stats:** `Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR` (full-time home/away goals + result), `HTHG/HTAG/HTR` (half-time), plus shots (`HS/AS`), shots on target (`HST/AST`), corners (`HC/AC`), fouls (`HF/AF`), cards (`HY/AY/HR/AR`), referee.
- **1X2 odds:** `B365H/D/A`, `PSH/PSD/PSA` (Pinnacle), `WHH/D/A`, `BWH/D/A`, `VCH/D/A`, plus market max/avg (`MaxH/D/A`, `AvgH/D/A`).
- **Closing odds** (since 2019/20): same books suffixed **`C`** — e.g. `PSCH/PSCD/PSCA` (Pinnacle closing), `B365CH…`, `MaxCH/AvgCH…`.
- **Over/Under 2.5:** `B365>2.5 / B365<2.5`, `P>2.5 / P<2.5` (Pinnacle), plus market avg/max and closing (`…C`).
- **Asian Handicap:** `AHh` (handicap line), `B365AHH/AHA`, `PAHH/PAHA` (Pinnacle), plus closing variants.

> Odds columns are what make this archive uniquely useful: they let you back-test a model against the market and measure **Closing Line Value (CLV)** — see §9. The `PSC*` (Pinnacle closing) columns are the sharpest reference available for free.

---

## 3) Advanced stats — xG, event & tracking data

| Source | What | Coverage | Free/Paid | URL |
|---|---|---|---|---|
| **FBref** (StatsBomb-powered) | Team/player stats, xG, match logs, shot-creating actions | 40+ leagues incl. all Big-5 + mid-tier EU | **Free** (scrape / via soccerdata) | [fbref.com](https://fbref.com/) |
| **Understat** | Shot-level **xG / xA** | Top-5 EU leagues + Russian Premier League, since 2014/15 | **Free** (scrape) | [understat.com](https://understat.com/) · py [understatapi](https://pypi.org/project/understatapi/) |
| **StatsBomb Open Data** | Event + **360** freeze-frames | Men's/Women's World Cups, UEFA Euro, UCL finals, La Liga (Messi era), FA WSL, more | **Free** (attribution: credit StatsBomb + logo) | [github](https://github.com/statsbomb/open-data) · py [statsbombpy](https://github.com/statsbomb/statsbombpy) · R [StatsBombR](https://github.com/statsbomb/StatsBombR) |
| **Opta / Stats Perform** | Official event data | Data partner of Premier League, Bundesliga, Serie A, La Liga + 3,900+ comps | **Paid / enterprise** | [Opta data](https://www.statsperform.com/products/opta-data/) |
| **StatsBomb** (full, now **Hudl**) | Event + 360 | Global licensed leagues (StatsBomb acquired by Hudl, Aug 2024) | **Paid** (free tier via open-data) | [statsbomb.com](https://statsbomb.com/) |
| **Wyscout** (Hudl) | Event + video scouting | 1,000+ competitions worldwide | **Paid** | [hudl.com/wyscout](https://www.hudl.com/products/wyscout) |
| **SkillCorner** | Broadcast **tracking** (computer vision) | Open-data sample + licensed physical/off-ball data | **Free sample / Paid** | [github](https://github.com/SkillCorner/opendata) · [skillcorner.com](https://skillcorner.com/) |
| **Impect** (Packing, now **Catapult**) | Packing® event data | 70+ competitions worldwide (acquired by Catapult, 2025) | **Paid** | [impect.com](https://www.impect.com/en/) |

---

## 4) Ratings & market values

| Source | What | Coverage | Free/Paid | URL |
|---|---|---|---|---|
| **ClubElo** | Daily **club Elo** ratings + win/draw/loss probabilities | European clubs, daily since 1939 | **Free API** (CSV) | [clubelo.com/API](http://clubelo.com/API) |
| **Transfermarkt** (via **dcaribou**) | Squads, **market values**, transfers, appearances | 79k+ games, 37k+ players, weekly refresh | **Free** (scraped mirror) | [Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores) · [github](https://github.com/dcaribou/transfermarkt-datasets) |
| **SofaScore** | Live scores, ratings, xG, momentum | Global | **Free site** (no official public API — scrape) | via [ScraperFC](https://github.com/oseymour/ScraperFC) / [soccerdata](https://github.com/probberechts/soccerdata) |
| **WhoScored** (Opta-powered) | Ratings, detailed match stats | Global top leagues | **Free site** (scrape ⚠️ ToS/JS-rendered) | via [soccerdata](https://github.com/probberechts/soccerdata) |

---

## 5) Live / reference APIs

| API | Coverage | Free tier | Paid from | URL |
|---|---|---|---|---|
| **API-Football** (api-sports.io) | 1,100+ leagues & cups: fixtures, standings, events, lineups, players, **pre-match odds**, stats | ✅ **100 req/day**, all endpoints | **$19/mo** (Pro tier) | [api-football.com](https://www.api-football.com/) · [docs](https://www.api-football.com/documentation-v3) |
| **football-data.org** | 12+ free comps incl. EPL, La Liga, Serie A, Bundesliga, Ligue 1, UCL, Euro | ✅ **10 calls/min**, fixtures/results/tables | from **€29/mo** | [football-data.org](https://www.football-data.org/) · [coverage](https://www.football-data.org/coverage) · [pricing](https://www.football-data.org/pricing) |
| **SportMonks** | 2,500+ leagues; live, odds, stats; Predictions add-on | 🆓 trial (2 free leagues) | €29 / €99 / €249 | [sportmonks.com](https://www.sportmonks.com/football-api/) |
| **The Odds API** | Live/upcoming odds from 40+ books (US/UK/EU/AU) — EU list includes **Pinnacle** & Betfair Exchange; 1X2, totals, spreads | ✅ **500 free credits/mo** (credit-based) | paid credit tiers | [the-odds-api.com](https://the-odds-api.com/) |

> ⚠️ **The Odds API** now lists Pinnacle and Betfair Exchange (EU), but its Pinnacle feed is scraped from the public site ("may incur a delay") and its free tier serves current/upcoming odds, not deep historical archives. For back-testing and **CLV** research on past seasons, the free `PSC*` (Pinnacle closing) columns in football-data.co.uk (§2) remain the better free reference.

---

## 6) Open datasets (download once, model offline)

| Dataset | What | Coverage | URL |
|---|---|---|---|
| **European Soccer Database** (Hugo Mathien) | Canonical SQLite DB: 25k+ matches, team/player FIFA attributes, odds | 11 EU countries, 2008–2016 | [Kaggle: hugomathien/soccer](https://www.kaggle.com/datasets/hugomathien/soccer) |
| **Club Football Match Data (2000–2025)** (Adam Gábor) | Merges football-data.co.uk results + **ClubElo** ratings + pre-match odds | 27 countries / 42 leagues | [Kaggle: adamgbor/club-football-match-data-2000-2025](https://www.kaggle.com/datasets/adamgbor/club-football-match-data-2000-2025) |
| **openfootball** | Public-domain fixtures/results (Football.TXT + JSON), no API key | Worldwide leagues, Europe, cups | [org](https://github.com/openfootball) · [europe](https://github.com/openfootball/europe) · [leagues](https://github.com/openfootball/leagues) · [football.json](https://github.com/openfootball/football.json) |
| **footballcsv** | Top leagues + cups as tidy CSV, auto-updated | Big-5 + more | [footballcsv.github.io](https://footballcsv.github.io/) · [github](https://github.com/footballcsv) |
| **martj42 — international results** | 49k+ men's internationals (1872→2024), shootouts, goalscorers | All national teams incl. UEFA | [github](https://github.com/martj42/international_results) · [Kaggle](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) |

---

## 7) Python / R tooling (all free / OSS)

| Tool | Lang | Sources it unifies | URL |
|---|---|---|---|
| **soccerdata** | Python | ClubElo, ESPN, FBref, Football-Data.co.uk, Sofascore, SoFIFA, Understat, WhoScored | [github](https://github.com/probberechts/soccerdata) |
| **ScraperFC** | Python | Capology, ClubElo, FBref, Sofascore, Transfermarkt, Understat | [github](https://github.com/oseymour/ScraperFC) |
| **understatapi** | Python | Understat (shot-level xG/xA) | [PyPI](https://pypi.org/project/understatapi/) |
| **worldfootballR** | R | FBref, Transfermarkt, Understat (**repo archived Sep 2025, read-only**) | [github](https://github.com/JaseZiv/worldfootballR) |

---

## 8) UEFA competitions data (UCL / UEL / UECL / Nations League / Euro)

Note the 2024/25 rebrand: the third-tier club competition dropped "Europa" and is now the **UEFA Conference League** — the abbreviation **UECL** is retained (to avoid clashing with UCL). See [UEFA announcement](https://www.uefa.com/uefaconferenceleague/news/0282-185d6b03cbe4-4763e115dd7f-1000--uefa-europa-conference-league-to-be-renamed-uefa-conferenc/).

| Competition | Free data | Paid / live |
|---|---|---|
| **UEFA Champions League (UCL)** | football-data.org (free tier), FBref, [openfootball/champions-league](https://github.com/openfootball/champions-league), StatsBomb open-data (finals) | API-Football, SportMonks |
| **UEFA Europa League (UEL)** | FBref, [openfootball/champions-league](https://github.com/openfootball/champions-league) | API-Football, SportMonks |
| **UEFA Conference League (UECL)** | FBref, openfootball | API-Football, SportMonks |
| **UEFA Nations League** | FBref, martj42 internationals, openfootball | API-Football |
| **UEFA Euro (2020/2024/2028)** | [openfootball/euro](https://github.com/openfootball/euro), StatsBomb open-data (Euro), football-data.org, martj42 | API-Football, SportMonks |

---

## 9) Modeling & market-efficiency literature (exact citations)

**Foundational goal models**

- **Maher, M.J. (1982).** *Modelling association football scores.* **Statistica Neerlandica** 36(3): 109–118. [DOI: 10.1111/j.1467-9574.1982.tb00782.x](https://doi.org/10.1111/j.1467-9574.1982.tb00782.x) — independent-Poisson attack/defence model (origin of Poisson goal models).
- **Dixon, M.J. & Coles, S.G. (1997).** *Modelling Association Football Scores and Inefficiencies in the Football Betting Market.* **J. R. Stat. Soc. C (Applied Statistics)** 46(2): 265–280. [DOI: 10.1111/1467-9876.00065](https://doi.org/10.1111/1467-9876.00065) — low-score dependence correction + time-decay weighting; basis of most modern models.

**Dynamic ratings & ML benchmarks**

- **Constantinou, A.C. & Fenton, N.E. (2013).** *Determining the level of ability of football teams by dynamic ratings based on the relative discrepancies in scores between adversaries* (**pi-ratings**). **J. Quantitative Analysis in Sports** 9(1): 37–50. [DOI: 10.1515/jqas-2012-0036](https://doi.org/10.1515/jqas-2012-0036) · [PDF](http://www.constantinou.info/downloads/papers/pi-ratings.pdf) · R pkg [piratings](https://github.com/larsvancutsem/piratings).
- **Constantinou, A.C. (2019).** *Dolores: a model that predicts football match outcomes from all over the world.* **Machine Learning** 108: 49–75. [DOI: 10.1007/s10994-018-5703-7](https://doi.org/10.1007/s10994-018-5703-7) — dynamic ratings + Hybrid Bayesian Networks trained across 52 leagues.
- **Berrar, D., Lopes, P. et al. (2024).** *Evaluating soccer match prediction models: a deep learning approach and feature optimization for gradient-boosted trees* (2024 Soccer Prediction Challenge). **Machine Learning.** [DOI: 10.1007/s10994-024-06608-w](https://doi.org/10.1007/s10994-024-06608-w) · [arXiv:2309.14807](https://arxiv.org/abs/2309.14807).

**Market efficiency — read before modeling odds**

- The **closing line** aggregates all sharp money and late news; it is the market's most efficient estimate. At **Pinnacle** the closing line correlates with realized outcomes at **r² ≈ 0.997** across **397,935** games ([Trademate Sports, Medium](https://tradematesports.medium.com/closing-line-the-most-important-metric-in-sports-trading-58e56cdb4458)). A realistic skill metric is **Closing Line Value (CLV)** — do your prices repeatedly beat the close? ([Pinnacle: What is CLV](https://www.pinnacle.com/betting-resources/en/educational/what-is-closing-line-value-clv-in-sports-betting)).
- The **bookmaker margin / overround (vig)** means a naive bettor loses on average even with "fair" skill. Treat all of this as **modeling practice**, not income.
- **Honest note on high odds / accumulators:** an accumulator multiplies the bookmaker's per-leg margin — a 4-fold at ~5% vig each carries roughly a compounded ~19% expected loss, *worse* EV than the singles. High-odds "big win" bets are **not** a shortcut to profit; they are lower-probability, higher-variance, and higher-margin. This page is an **honest debunk**, not promotion.

---

## 10) Responsible gambling (Jogo Responsável) — mandatory

Gambling can cause serious harm. This index exists for **data science / ML research and education only** — it is **not** advice, a tip service, or a "system". Set limits, never chase losses, and seek help if betting stops being fun.

| Region | Resource | Contact |
|---|---|---|
| 🌍 Global | [Gambling Therapy](https://www.gamblingtherapy.org/) | Online chat/forum, multilingual |
| 🇬🇧 UK | [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) | National Gambling Helpline **0808 8020 133** (24/7) |
| 🇧🇷 Brasil | [Jogo Responsável — SPA/Ministério da Fazenda](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) | Apoio emocional **CVV 188** |

**PT-BR:** Este índice é apenas para **pesquisa e educação** em ciência de dados / ML. **Não é dica, aposta garantida nem "esquema".** Os mercados europeus são altamente eficientes; após a margem da casa (*overround*), a maioria dos apostadores perde dinheiro ao longo do tempo. Aposte com responsabilidade, defina limites e, se precisar de ajuda, procure o **CVV 188** ou o [Jogo Responsável](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas).

---

## Related in AIForge
- [Global Football Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md) · [Africa, Middle East & Oceania Data](./Africa_MiddleEast_and_Oceania_Football_Data.md) · [Kaggle Football Datasets & Competitions](./Kaggle_Football_Datasets_and_Competitions.md) · [Open-Source Tools & Libraries](./Open_Source_Tools_and_Libraries.md) · [Innovative Models & Deep Learning](./Innovative_Models_and_Deep_Learning.md) · Parent vertical: [`../`](../) (Sports Analytics AI)

**Sources:** football-data.co.uk (data.php, downloadm.php, all_new_data.php, notes.txt) · fbref.com · understat.com · github.com/statsbomb/open-data · statsperform.com/products/opta-data · statsbomb.com · hudl.com/products/wyscout · github.com/SkillCorner/opendata · skillcorner.com · impect.com · clubelo.com/API · Kaggle (davidcariboo/player-scores, hugomathien/soccer, adamgbor/club-football-match-data-2000-2025, martj42) · github.com/dcaribou/transfermarkt-datasets · github.com/probberechts/soccerdata · github.com/oseymour/ScraperFC · pypi.org/project/understatapi · github.com/JaseZiv/worldfootballR · github.com/openfootball (europe, champions-league, euro, leagues, football.json) · footballcsv.github.io · github.com/martj42/international_results · api-football.com · football-data.org · sportmonks.com · the-odds-api.com · uefa.com · doi.org/10.1111/j.1467-9574.1982.tb00782.x (Maher 1982) · doi.org/10.1111/1467-9876.00065 (Dixon–Coles 1997) · doi.org/10.1515/jqas-2012-0036 (pi-ratings 2013) · doi.org/10.1007/s10994-018-5703-7 (Dolores 2019) · doi.org/10.1007/s10994-024-06608-w & arxiv.org/abs/2309.14807 (2024 Soccer Prediction Challenge) · tradematesports.medium.com · pinnacle.com · begambleaware.org · gamcare.org.uk · gamblingtherapy.org · gov.br/fazenda (SPA)

**Keywords:** European football data, soccer datasets Europe, Big-5 leagues data, Premier League La Liga Serie A Bundesliga Ligue 1 data, football-data.co.uk closing odds, Pinnacle closing line, Understat xG, FBref stats, StatsBomb open data, ClubElo ratings, Transfermarkt market values, API-Football, football-data.org, UEFA Champions League data, Euro 2024 dataset, Dixon-Coles Poisson model, pi-ratings, closing line value, market efficiency, dados de futebol europeu, ligas europeias, odds de fechamento, previsão de partidas, jogo responsável, mercado de apostas eficiente.
