# Americas — Football Data, Leagues & Markets

> Authoritative, regional index of **where to get football (soccer / futebol) data** for match & betting-prediction research across the **Americas** — South America (**CONMEBOL**: 🇧🇷 Brazil, 🇦🇷 Argentina, 🇺🇾 Uruguay, 🇨🇴 Colombia, 🇨🇱 Chile, 🇵🇪 Peru, 🇪🇨 Ecuador, 🇵🇾 Paraguay) and North/Central America & Caribbean (**CONCACAF**: 🇺🇸🇨🇦 MLS, 🇲🇽 Liga MX, USL, Central America) — plus continental cups (Libertadores, Sudamericana, CONCACAF Champions Cup, Leagues Cup) and the 🇧🇷 **Brazilian betting-regulation** context. Real URLs, free-vs-paid marks, current 2024–2026. Built for **data science / ML research & education**.

> ⚠️ **Research & education only — not betting advice (isto não é dica de aposta).** Betting markets are **highly efficient**; the bookmaker's overround (*margem / vig*) means a bettor must win **≥ ~52.4%** of even-money bets just to break even, and **most bettors lose money over time**. Sustained edges are **rare and hard** — honest skill is measured by **Closing Line Value (CLV)**, not by lucky wins. Nothing here is a tip or a system. If gambling is a problem, get help: [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) · [Gambling Therapy](https://www.gamblingtherapy.org/) (multilingual, PT/ES) · 🇧🇷 [Jogo Responsável — SPA/MF](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel) · 🇧🇷 CVV **188** · 🇺🇸 1-800-GAMBLER.

**How to choose:** free CSV history with odds ([football-data.co.uk](https://www.football-data.co.uk/brazil.php) — Brazil/Argentina/Mexico/USA) for quick baselines → free advanced data ([American Soccer Analysis](https://www.americansocceranalysis.com/) xG + Goals Added for MLS/USL/NWSL; [FBref](https://fbref.com/en/comps/) results) → **live/reference APIs** ([API-Football](https://www.api-football.com/), [SportMonks](https://www.sportmonks.com/), [football-data.org](https://www.football-data.org/)) for pipelines → scrape ([soccerdata](https://github.com/probberechts/soccerdata), [ScraperFC](https://github.com/oseymour/ScraperFC), [worldfootballR](https://github.com/JaseZiv/worldfootballR)) for [SofaScore](https://www.sofascore.com/)/[Transfermarkt](https://www.transfermarkt.com/) → paid providers (Opta/Stats Perform, Sportradar, Wyscout) for depth. Sibling pages: [Global Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md) · [Africa, Middle East & Oceania](./Africa_MiddleEast_and_Oceania_Football_Data.md) · [Kaggle Football Datasets](./Kaggle_Football_Datasets_and_Competitions.md).

---

## 1) The Americas league landscape (what to model)

Two confederations. **CONMEBOL** (10 nations, South America) is the smallest-membership but historically strongest confederation; **CONCACAF** (41 members, North/Central America & Caribbean) is dominated by MLS and Liga MX. Data availability is uneven: 🇧🇷 Brazil, MLS and 🇲🇽 Liga MX are well-covered; Andean and Central-American leagues are thinner.

| Country | Top league (local name) | Confed. | Continental cups |
|---|---|---|---|
| 🇧🇷 Brazil | **Brasileirão Série A** (Campeonato Brasileiro); Série B/C/D + state leagues (*estaduais*) | CONMEBOL | Libertadores · Sudamericana |
| 🇦🇷 Argentina | **Liga Profesional de Fútbol** (Primera División) + Copa de la Liga | CONMEBOL | Libertadores · Sudamericana |
| 🇺🇾 Uruguay | **Primera División** (Campeonato Uruguayo) | CONMEBOL | Libertadores · Sudamericana |
| 🇨🇴 Colombia | **Categoría Primera A** (Liga BetPlay Dimayor) | CONMEBOL | Libertadores · Sudamericana |
| 🇨🇱 Chile | **Primera División** (Liga de Primera / ANFP) | CONMEBOL | Libertadores · Sudamericana |
| 🇵🇪 Peru | **Liga 1** (Primera División) | CONMEBOL | Libertadores · Sudamericana |
| 🇪🇨 Ecuador | **LigaPro Serie A** | CONMEBOL | Libertadores · Sudamericana |
| 🇵🇾 Paraguay | **Primera División** (APF) | CONMEBOL | Libertadores · Sudamericana |
| 🇺🇸🇨🇦 USA/Canada | **Major League Soccer (MLS)**; USL Championship, USL League One, MLS Next Pro | CONCACAF | Champions Cup · Leagues Cup |
| 🇲🇽 Mexico | **Liga MX** (Apertura/Clausura); Liga de Expansión MX | CONCACAF | Champions Cup · Leagues Cup |
| 🇨🇷 Costa Rica | **Liga Promerica** (Primera División) | CONCACAF | Champions Cup · Central American Cup |
| 🌎 Central America | Guatemala Liga Nacional, Honduras Liga Nacional, Panamá LPF, El Salvador | CONCACAF | Champions Cup · Central American Cup |

> Format notes (2024–2026): Brasileirão Série A = **20 clubs, double round-robin (*pontos corridos*)**, promotion/relegation with Série B. Argentina & Liga MX use **Apertura/Clausura** split seasons with playoffs (*liguilla*). MLS = **conference + Audi MLS Cup Playoffs**; the season calendar and possible shift toward the international (autumn–spring) calendar is a live 2025–2027 governance topic.

---

## 2) South America (CONMEBOL) — data sources

| Country / competition | Source | What you get | Free/Paid | URL |
|---|---|---|---|---|
| 🇧🇷 Brasileirão + Libertadores/Sudamericana/Copa do Brasil | **Kaggle — ricardomattos05** (Brazilian Soccer Database) | Historical + current matches for all major comps Brazilian clubs play; maintained | **Free** | [kaggle](https://www.kaggle.com/datasets/ricardomattos05/jogos-do-campeonato-brasileiro) · [GitHub src](https://github.com/ricardo-mattoss/Brazilian-Soccer-Data) |
| 🇧🇷 Brasileirão Série A | **Kaggle — adaoduque** (Campeonato Brasileiro de Futebol) | Match-level results since 2003, popular teaching set | **Free** | [kaggle](https://www.kaggle.com/datasets/adaoduque/campeonato-brasileiro-de-futebol) |
| 🇧🇷 + South America | **Kaggle — felipesembay** (Sofascore & Transfermarkt) | Scraped SofaScore + Transfermarkt data for Brazil/South America | **Free** | [kaggle](https://www.kaggle.com/datasets/felipesembay/sofascore-and-transfermarkt-football-data) |
| 🇧🇷 Série A | **football-data.co.uk (Brazil)** | Results + **match-odds** columns (incl. Bet365/Pinnacle where available), CSV since 2012 | **Free** | [brazil.php](https://www.football-data.co.uk/brazil.php) |
| 🇦🇷 Argentina Primera | **football-data.co.uk (Argentina)** | Results + odds, CSV (Extra/New-Leagues section) | **Free** | [data hub](https://www.football-data.co.uk/data.php) |
| 🌎 Brazil, Argentina, Colombia + | **openfootball / south-america** | Public-domain (CC0) schedules & results, JSON/CSV/TXT | **Free (OSS)** | [github](https://github.com/openfootball/south-america) |
| 🌎 All CONMEBOL leagues + cups | **FBref** (Sports Reference) | Results, schedules, standings, squad/player season stats — Série A `comps/24`, Copa Lib. `comps/14`, Copa de la Liga `comps/905` | **Free** (view) | [fbref comps](https://fbref.com/en/comps/) |
| 🌎 All CONMEBOL leagues + cups | **SofaScore** | Live scores, ratings, xG, momentum, lineups — deep South-America coverage | **Free** (view; no public API) | [sofascore.com](https://www.sofascore.com/) |
| 🌎 All CONMEBOL | **Transfermarkt** | Squads, **market values** (*valor de mercado*), transfers, injuries | **Free** (view) | [transfermarkt.com](https://www.transfermarkt.com/) · [dcaribou datasets](https://github.com/dcaribou/transfermarkt-datasets) |
| 🇧🇷 Official records | **CBF** (Confederação Brasileira de Futebol) | Fixtures, *súmulas* (match reports), BID, tables — reference truth for Brazil | **Free** (view) | [cbf.com.br](https://www.cbf.com.br/) |

> ⚠️ [**Understat**](https://understat.com/) does **not** cover any Americas league (top-5 European leagues + Russian RPL only) — do **not** expect Understat xG for Brasileirão, MLS or Liga MX.

---

## 3) North/Central America (CONCACAF) — data sources

The **best free advanced data in the Americas** is [**American Soccer Analysis (ASA)**](https://www.americansocceranalysis.com/): open **xG** and the proprietary **Goals Added (g+)** metric, with a public API and R/Python clients.

| Competition | Source | What you get | Free/Paid | URL |
|---|---|---|---|---|
| 🇺🇸 MLS, NWSL, USL Champ. (2017–), USL League One (2019–) | **American Soccer Analysis** (data explorer + public API) | Team/player/keeper **xG**, **Goals Added (g+)**, game & shot data; no key needed | **Free** | [app](https://www.americansocceranalysis.com/) · [API docs](https://app.americansocceranalysis.com/api/v1/__docs__/) |
| ↳ ASA programmatic access | **itscalledsoccer** (Python) · **itscalledsoccer-r** (R, on CRAN) | Official wrappers → tidy DataFrames of games/players/teams/xG/g+ | **Free (OSS)** | [PyPI](https://pypi.org/project/itscalledsoccer/) · [GitHub py](https://github.com/American-Soccer-Analysis/itscalledsoccer) · [GitHub R](https://github.com/American-Soccer-Analysis/itscalledsoccer-r) |
| 🇺🇸 MLS official | **MLSsoccer.com — Stats** | Official standings, schedules, box-score stats | **Free** (view) | [mlssoccer.com/stats](https://www.mlssoccer.com/stats/) |
| 🇺🇸 MLS | **Kaggle — josephvm** (MLS Dataset) | 6,000+ matches, ~420k events, player/game/event/table tables | **Free** | [kaggle](https://www.kaggle.com/datasets/josephvm/major-league-soccer-dataset) |
| 🌎 MLS, Liga MX, USL, CONCACAF cups | **FBref** | Results, schedules, season stats — MLS `comps/22`, Liga MX `comps/31` | **Free** (view) | [fbref comps](https://fbref.com/en/comps/) |
| 🇲🇽 Liga MX + North America | **SofaScore / Transfermarkt** | Live/ratings/xG (SofaScore); market values (Transfermarkt) | **Free** (view) | [sofascore](https://www.sofascore.com/) · [transfermarkt](https://www.transfermarkt.com/) |
| 🇺🇸 MLS/NWSL selected | **StatsBomb Open Data** | Full **event data** (JSON) for released comps (incl. MLS, NWSL, Copa América, Argentina Liga Profesional) — check `competitions.json` | **Free** (attribution) | [github](https://github.com/statsbomb/open-data) |

> ⚠️ ASA covers **North American leagues only** — it does **not** include Liga MX. For Liga MX advanced numbers, use SofaScore/API-Football or paid Opta/Sportradar.

---

## 4) Cross-region live/reference APIs & scraper toolkits (cover the Americas)

For leagues without a tidy CSV (Andean, Central-American, Liga MX granular), pull from these.

| Source | Americas coverage | Free tier | Paid from | URL |
|---|---|---|---|---|
| **API-Football** (api-sports.io) | 1,200+ leagues incl. Brasileirão A/B, Liga Profesional AR, Liga MX, MLS, USL, Colombia, Chile, Peru, Ecuador + **Libertadores, Sudamericana, CONCACAF cups, Leagues Cup**; fixtures, events, stats, **pre-match odds** | ✅ **100 req/day**, all endpoints | **$19/mo** (Pro) | [api-football.com](https://www.api-football.com/) · [coverage](https://www.api-football.com/coverage) |
| **SportMonks** | Dedicated **MLS** & **Brasileirão** endpoints; South/North America leagues, odds, Predictions API | 🆓 14-day trial | €29/mo (Starter) → Enterprise | [MLS API](https://www.sportmonks.com/football-api/mls-api/) · [Brasileirão API](https://www.sportmonks.com/football-api/brasileirao-api/) |
| **football-data.org** | Free tier includes **Brazil Série A (BSA)** only (12 comps total); **Copa Libertadores (CLI)**, Série B, Copa do Brasil, Liga MX, MLS, Argentina/Chile/Colombia/Peru/Uruguay are **paid** | ✅ 10 calls/min | paid tiers | [coverage](https://www.football-data.org/coverage) · [pricing](https://www.football-data.org/pricing) |
| **The Odds API** | Bookmaker odds incl. some Americas competitions (h2h/spreads/totals) | ✅ 500 req/mo | Paid tiers | [the-odds-api.com](https://the-odds-api.com/sports-odds-data/football-odds.html) |
| **Sportradar / Opta (Stats Perform) / Wyscout (Hudl)** | Official/enterprise event & tracking for most Americas top leagues | 🆓 trial keys | Enterprise | [Sportradar](https://developer.sportradar.com/soccer/reference/soccer-api-overview) · [Opta](https://www.statsperform.com/opta-data/) |

**Scraper toolkits** (Python/R, free/OSS): [`soccerdata`](https://github.com/probberechts/soccerdata) (FBref, SofaScore, ClubElo, Football-Data.co.uk…), [`ScraperFC`](https://github.com/oseymour/ScraperFC) (FBref, SofaScore, Transfermarkt, Understat…), [`worldfootballR`](https://github.com/JaseZiv/worldfootballR) (FBref, Transfermarkt, Understat — repo **archived read-only since Sept 2025** but still installable), [`itscalledsoccer`](https://github.com/American-Soccer-Analysis/itscalledsoccer) (ASA), [`statsbombpy`](https://github.com/statsbomb/statsbombpy) + [`mplsoccer`](https://github.com/andrewRowlinson/mplsoccer) for event/pitch viz.

> ⚠️ **FBref caveat (critical, 2026):** On **2026-01-20**, Sports Reference announced that Stats Perform (Opta) terminated its feed deal and **all advanced stats (xG, xA, progressive passes, SCA, etc.) were removed** from FBref — about a week after Stats Perform was named FIFA's exclusive 2026 World Cup betting-data & streaming distributor ([Heredia write-up](https://ricardoheredia.substack.com/p/farewell-fbref-advanced-stats-when) · [The IX Sports](https://www.theixsports.com/the-ix-soccer/fbrefs-loss-advanced-stats-womens-soccer-data-accessibility/) · [Awful Announcing](https://awfulannouncing.com/soccer/sports-reference-pulls-advanced-data-agreement-violation-dispute.html)). **Basic results/schedules/standings remain free**, but any Americas pipeline that depended on FBref xG is now **broken** — pivot to ASA (North America), SofaScore, or paid Opta/Sportradar.

---

## 5) 🇧🇷 Brazil deep-dive — data maturity, regulation & analytics community

**Data maturity.** Brasileirão is the **most data-mature league in the Americas after MLS**: broad SofaScore/Transfermarkt coverage, Opta/Stats Perform tracking behind broadcast, and the community-standard **Ricardo Mattos dataset** ([Kaggle](https://www.kaggle.com/datasets/ricardomattos05/jogos-do-campeonato-brasileiro)) covering Brasileirão + Libertadores + Sudamericana + Copa do Brasil. Official reference truth is **CBF** ([cbf.com.br](https://www.cbf.com.br/) — fixtures, *súmulas*, BID). Free advanced xG for Brazil is now **thin** post-FBref removal — SofaScore (view/scrape) or paid providers are the practical options.

**Betting regulation — Lei das Bets (Lei nº 14.790/2023).** Brazil legalized and regulated fixed-odds betting (*apostas de quota fixa*) and online casino via **Law 14.790/2023** (of 29 Dec 2023), in force **1 January 2025** ([Planalto — full text](https://www.planalto.gov.br/ccivil_03/_ato2023-2026/2023/lei/l14790.htm)). Key points, all **research context, not advice**:

| Item | Detail |
|---|---|
| Regulator | **SPA/MF** — Secretaria de Prêmios e Apostas, Ministério da Fazenda |
| Effective | Only SPA-authorised operators may operate/advertise from **01 Jan 2025** |
| Licence | fixed fee up to **R$30 million** (covers up to 3 brands); anti-fraud & AML obligations |
| Tax | **12% on GGR** (*receita bruta*) + bettor-side income tax on net winnings |
| Player safeguards | **CPF + facial recognition** ID; **ban on credit betting**; **no sign-up bonuses**; behavioural monitoring |
| Self-exclusion | Statutory; **centralized self-exclusion platform** live **10 Dec 2025** → [gov.br/…/autoexclusao](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/autoexclusao) (block 1–12 months or indefinite; >600k requests by mid-2026) |
| Responsible gaming | **Portaria SPA/MF nº 1.231/2024** (31 Jul 2024) sets *jogo responsável* rules → [SPA jogo responsável](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel) |

**Sponsorship context (why this matters for data culture).** Betting brands flooded Brazilian football: for 2024, **~85% of Série A + B clubs (34/40)** carried a betting front-of-shirt sponsor, and **Betano** became the league's first betting **title sponsor** (Brasileirão Betano, ~US$50m/3-year deal, Apr 2024); record deals included **Pixbet–Flamengo** and the (later cancelled) **VaideBet–Corinthians**. After the 2025 regulated-tax regime, operators **pulled back** — betting front-of-shirt sponsors in Série A fell from **18 (2025) to 13 (2026) clubs (~28%)** ([SportQuake](https://www.sportquake.com/news-insights/the-rise-of-sports-betting-sponsorship-in-brazilian-football/) · [iGaming Business](https://igamingbusiness.com/finance/brazil-betting-sponsorship-in-decline/)). This betting money is a major (and volatile) revenue line for clubs — relevant when reasoning about squad quality and market context.

**Analytics community.** Brazil has a growing *futebol* analytics scene — public Kaggle datasets (ricardomattos05, adaoduque, felipesembay), Python/R users of `soccerdata`/`worldfootballR`, and academic/independent work on xG and Poisson/Dixon-Coles models for Brasileirão. Treat community scrapes as **useful but unofficial** — validate against CBF for canonical results.

---

## 6) Continental competitions (Americas) — data & fixtures

| Competition | Format (2024–2026) | Data source | Free/Paid |
|---|---|---|---|
| **Copa Libertadores** (CONMEBOL) | Premier South-American club cup; group stage → knockout, single-leg final | API-Football, SofaScore, FBref (`comps/14`), Kaggle ricardomattos05 | Free/Paid |
| **Copa Sudamericana** (CONMEBOL) | Second-tier continental club cup | API-Football, SofaScore, Kaggle ricardomattos05 | Free/Paid |
| **CONCACAF Champions Cup** | Rebranded 2024; **27 clubs**, five knockout rounds; Pachuca 2024, Toluca 2026 — region's top club title | API-Football, SofaScore, [Concacaf.com](https://www.concacaf.com/competitions/champions-cup/) | Free/Paid |
| **Leagues Cup** | MLS × Liga MX; 2025 = **36 clubs** (all 18 Liga MX + top 18 MLS), CCC-qualification berths | API-Football, SofaScore, MLSsoccer | Free/Paid |
| **Copa do Brasil** | 🇧🇷 National knockout cup | API-Football, Kaggle ricardomattos05, CBF | Free/Paid |
| **Copa América** (CONMEBOL nat. teams) | 2024 hosted in USA; next 2028 | API-Football, [eloratings.net](https://eloratings.net/), StatsBomb (released editions) | Free/Paid |

---

## 7) Reality check — why beating Americas markets is hard

- **The market is the benchmark.** Bookmaker/exchange **closing lines** aggregate sharp money and team news; classic efficiency studies (Pope & Peel 1989; Kuypers 2000) find little exploitable bias in win-draw-loss odds, and even a back-test "winner" gets **limited/closed** in practice (Kaunitz et al. 2017, [arXiv:1710.02824](https://arxiv.org/abs/1710.02824)).
- **The vig is a headwind.** At typical −110 pricing you need **≥ 52.4%** win rate just to break even (110 ÷ 210 = 0.524); overround on 1X2 markets works the same way. Skill is judged by **CLV** ([Pinnacle: what is CLV](https://www.pinnacle.com/betting-resources/en/educational/what-is-closing-line-value-clv-in-sports-betting)), not by short-run profit.
- **ML is not a money printer.** Surveys/reviews find modest, fragile gains — Bunker, Yeung & Fujii, *Machine Learning for Soccer Match Result Prediction* ([arXiv:2403.07669](https://arxiv.org/abs/2403.07669)); *A Systematic Review of Machine Learning in Sports Betting* ([arXiv:2410.21484](https://arxiv.org/abs/2410.21484)); and "can simple models beat the odds?" studies whose backtests show, at best, fragile calibration-limited signals — e.g. [Wilkens (2026)](https://journals.sagepub.com/doi/10.1177/22150218261416681) finds a simple xG→Skellam model with ~10% *simulated* Bundesliga ROI yet notes bookmaker odds stay **better-calibrated**, and paper-ROI like this rarely survives real-world limits, margins and CLV erosion.
- **Foundations to learn (not to profit):** Maher (1982) and **Dixon–Coles (1997)** Poisson goal models; modern xG- and distribution-based forecasters (e.g. Mendes-Neves et al., *Forecasting Soccer Matches through Distributions*, [arXiv:2501.05873](https://arxiv.org/abs/2501.05873)). Build these to **understand football**, and treat any "edge" with deep skepticism.

---

## 8) Responsible gambling (Jogo Responsável) — mandatory

Gambling can cause serious harm. This index exists for **data science / ML research and education only** — it is **not** betting advice and **not** a tip service. Set limits, never chase losses (*nunca persiga o prejuízo*), and seek help if betting stops being fun.

| Region | Resource | Contact |
|---|---|---|
| 🌍 Global | [Gambling Therapy](https://www.gamblingtherapy.org/) | Online chat/forum, **multilingual (PT/ES)** |
| 🇧🇷 Brazil | [Jogo Responsável — SPA/MF](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel) · [Autoexclusão](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/autoexclusao) | Bloqueio de CPF em todas as *bets*; apoio emocional **CVV 188** |
| 🇺🇸 USA | [NCPG — National Council on Problem Gambling](https://www.ncpgambling.org/) | **1-800-GAMBLER** (1-800-426-2537), 24/7 |
| 🇬🇧 UK | [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) | National Gambling Helpline **0808 8020 133** (24/7) |
| 🌎 Latin America | [Gambling Therapy (Español/Português)](https://www.gamblingtherapy.org/) | Chat/foro multilingüe |

---

## Related in AIForge
- [Global Football Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md) · [Africa, Middle East & Oceania — Football Data](./Africa_MiddleEast_and_Oceania_Football_Data.md) · [Kaggle Football Datasets & Competitions](./Kaggle_Football_Datasets_and_Competitions.md) · [Innovative Models & Deep Learning](./Innovative_Models_and_Deep_Learning.md) · [Open-Source Tools & Libraries](./Open_Source_Tools_and_Libraries.md) · Parent vertical: [`../`](../) (Sports Analytics AI)

**Sources:** americansocceranalysis.com · app.americansocceranalysis.com/api/v1/__docs__ · github.com/American-Soccer-Analysis/itscalledsoccer(-r) · pypi.org/project/itscalledsoccer · mlssoccer.com/stats · kaggle.com/datasets/ricardomattos05 · github.com/ricardo-mattoss/Brazilian-Soccer-Data · kaggle.com/datasets/adaoduque/campeonato-brasileiro-de-futebol · kaggle.com/datasets/felipesembay/sofascore-and-transfermarkt-football-data · kaggle.com/datasets/josephvm/major-league-soccer-dataset · football-data.co.uk/brazil.php · football-data.co.uk/data.php · github.com/openfootball/south-america · fbref.com/en/comps (24/22/31/14/905) · sofascore.com · transfermarkt.com · github.com/dcaribou/transfermarkt-datasets · cbf.com.br · api-football.com/coverage · sportmonks.com/football-api (mls-api, brasileirao-api) · football-data.org/coverage · the-odds-api.com · developer.sportradar.com · statsperform.com/opta-data · github.com/probberechts/soccerdata · github.com/oseymour/ScraperFC · github.com/JaseZiv/worldfootballR · github.com/statsbomb/open-data · understat.com · concacaf.com · eloratings.net · planalto.gov.br (Lei 14.790/2023) · gov.br/fazenda/SPA (jogo-responsavel, autoexclusão) · sportquake.com · igamingbusiness.com · ricardoheredia.substack.com · theixsports.com · awfulannouncing.com · pinnacle.com · arxiv.org/abs/2403.07669 · arxiv.org/abs/2410.21484 · arxiv.org/abs/1710.02824 · arxiv.org/abs/2501.05873 · journals.sagepub.com/doi/10.1177/22150218261416681 · begambleaware.org · gamcare.org.uk · gamblingtherapy.org · ncpgambling.org

**Keywords:** Americas football data, CONMEBOL dataset, CONCACAF data, Brasileirão data, Campeonato Brasileiro Série A dataset, dados de futebol brasileiro, Liga Profesional Argentina data, Liga MX dataset, MLS data, American Soccer Analysis xG, Goals Added g+, itscalledsoccer, Copa Libertadores data, Copa Sudamericana, CONCACAF Champions Cup, Leagues Cup, Copa do Brasil, football-data.co.uk Brazil, ricardomattos05 Kaggle, openfootball south-america, API-Football, SportMonks Brasileirão API, SofaScore, Transfermarkt, FBref advanced stats removed, Lei das Bets 14.790/2023, apostas esportivas Brasil, SPA Ministério da Fazenda, jogo responsável, autoexclusão apostas, closing line value, betting market efficiency, previsão de partidas de futebol, mercado de apostas eficiente, isto não é dica de aposta.
