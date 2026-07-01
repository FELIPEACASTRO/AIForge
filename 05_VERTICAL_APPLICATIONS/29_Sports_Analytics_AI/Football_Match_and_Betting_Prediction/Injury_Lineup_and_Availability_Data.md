# Injury, Lineup & Player-Availability Data

> Focused, worldwide index of **team-news / player-availability data** for football (soccer) prediction — **injury tables (tabelas de lesões)**, **predicted & confirmed line-ups (escalações prováveis e confirmadas)**, suspensions (suspensões), rotation and international-duty fatigue — and how to model them **without leaking future information**. Availability is one of the few *genuinely predictive, under-priced* signal families, distinct from generic match features: a starting XI is public only shortly before kickoff, and star-player absences move goal and 1X2 markets. Every site, API, dataset and paper below was checked for existence/status via live search or fetch. Built for **research & education (pesquisa e educação — data science / ML)**, current 2024–2026.

> ⚠️ **Markets are largely efficient and most bettors lose.** Bookmakers and exchange traders adjust within seconds of team news, and the classic-line vs closing-line edge is thin. This page is **research & education only — NOT betting or investment advice (não é dica de aposta).** Nothing here predicts an individual result. If gambling stops being fun, get help: [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) · [Gambling Therapy](https://www.gamblingtherapy.org/) · 🇧🇷 **Jogo Responsável** — [SPA/Ministério da Fazenda](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) · apoio emocional **CVV 188**.

**Where this fits:** for outcome models (Poisson/Dixon-Coles/Elo/xG/ML/DL), value/CLV, generic features, datasets/APIs and the classic papers, see the sibling pages → [Match Prediction Models & Techniques](./Match_Prediction_Models_and_Techniques.md) · [Global Football Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md) · [Bet Selection & Staking](./Bet_Selection_Staking_and_High_Odds_Analysis.md) · [Betting Exchanges & Trading](./Betting_Exchanges_Trading_and_Microstructure.md). This page is the **availability / team-news layer** feeding all of them.

---

## 0) Why team-news is a distinct edge (and its limits)

Most public features (form, xG, Elo, home advantage) are already priced. **Availability is different for three reasons:** (1) it is *late-arriving* — an official XI is published only ~75 minutes before kickoff (see §7); (2) it is *high-leverage* — a benched or absent key player changes team λ (expected goals) more than a season of small form swings; and (3) it is *messy* — pre-match "predicted" line-ups carry real uncertainty (returning-from-injury doubt, manager mind-games). The injury literature underlines the scale: **4,123 injuries were recorded across the top-five European leagues in 2023/24, for a combined ≈€732m cost** ([Howden 2023/24 Men's European Football Injury Index, via Sport Resolutions](https://www.sportresolutions.com/news/significant-increase-in-injuries-costs-top-leagues-732-million-euros)). Absence is therefore a large, recurring perturbation — but because traders watch the same feeds, the edge lives in **speed, correct sizing of the effect, and avoiding leakage**, not in the raw fact that "player X is out."

---

## 1) Predicted & confirmed line-ups (escalações)

Aggregators publish **probable XIs** days ahead and switch to the **confirmed XI** at the official release. Treat "predicted" as a distribution, not a fact.

| Source | Data | Free/Paid | Link |
|---|---|---|---|
| **Sofascore** | Predicted + confirmed line-ups, formations, avg positions; 500+ competitions | Free | [sofascore.com](https://www.sofascore.com/) |
| **FotMob** | Probable XIs, injury/fitness status + expected return dates, alerts; Lineup Builder tool | Free | [fotmob.com](https://www.fotmob.com/) · [lineup-builder](https://www.fotmob.com/lineup-builder) |
| **WhoScored** | Match previews with predicted line-ups + injuries/suspensions, top-5 leagues & tournaments | Free | [whoscored.com/previews](https://www.whoscored.com/previews) |
| **RotoWire Soccer** | Predicted **and** confirmed starting XIs, formations, team news; PL + World Cup + more | Freemium | [rotowire.com/soccer/lineups.php](https://www.rotowire.com/soccer/lineups.php) |
| **SportsGambler** | Predicted **and** confirmed line-ups across many worldwide leagues; separate injuries/suspensions hub | Free | [lineups](https://www.sportsgambler.com/lineups/football/) · [injuries](https://www.sportsgambler.com/injuries/football/england-premier-league/) |
| **Fantasy Football Scout** | Predicted line-ups & team news (Lineup11 graphics); FPL-oriented but general | Freemium | [fantasyfootballscout.co.uk/team-news](https://www.fantasyfootballscout.co.uk/team-news) |

---

## 2) Injury tables & availability trackers (tabelas de lesões)

Structured "who's out, why, expected return" lists — the backbone of an availability feature set.

| Source | Data | Free/Paid | Link |
|---|---|---|---|
| **PhysioRoom** | Long-running PL injury table (injured players + injury type), per-club injury counts & season injury-type stats | Free | [physioroom.com/…/premier-league-injury-table](https://www.physioroom.com/advice/premier-league-injury-table/) |
| **Premier Injuries** (Ben Dinnery) | EPL injury/suspension DB, per-club pages, expert newsroom; email updates | Freemium | [injury-table](https://www.premierinjuries.com/injury-table.php) · [newsroom](https://www.premierinjuries.com/newsroom) · [Ben Dinnery](https://www.premierinjuries.com/expert/ben-dinnery) |
| **Transfermarkt** | League "injured players" (`/verletztespieler/`) + per-club "suspensions & injuries" (`/sperrenundverletzungen/`), type + games/expected time missed; global coverage | Free | [LaLiga injured](https://www.transfermarkt.com/laliga/verletztespieler/wettbewerb/ES1) · [Arsenal susp.+inj.](https://www.transfermarkt.com/fc-arsenal/sperrenundverletzungen/verein/11) |
| **Premier League (official)** | Club-by-club latest player injuries | Free | [premierleague.com/en/latest-player-injuries](https://www.premierleague.com/en/latest-player-injuries) |
| **Sky Sports** | PL injury table + **suspension tracker** + FPL notes | Free | [skysports.com injuries](https://www.skysports.com/football/news/11661/13415725/) |
| **NBC Sports** | PL 2025/26 club-by-club injuries, suspensions, status | Free | [nbcsports.com injury table](https://www.nbcsports.com/soccer/news/premier-league-injury-table-2025-26-season-club-by-club-injuries-suspensions-latest-updates) |

> **Coverage caveat:** the richest structured data is **Premier League-centric** (PhysioRoom, Premier Injuries, Sky, NBC). For non-English leagues, **Transfermarkt** (per-league/club) and **Sofascore/FotMob** are the most consistent free sources — swap the competition/club slug in the URL.

---

## 3) Programmatic availability (APIs)

For back-testable pipelines you need machine-readable injuries + line-ups keyed to a fixture.

| API | Availability data | Free/Paid | Link |
|---|---|---|---|
| **API-Football** (API-Sports) | `injuries` endpoint (player, team, fixture, type/reason, date) + `fixtures/lineups` (starting XI, formation, grid positions, bench, coach); coverage flags per league | Freemium (free ~100 req/day) | [documentation-v3](https://www.api-football.com/documentation-v3) · [injuries endpoint](https://www.api-football.com/news/post/new-endpoint-injuries) · [api-sports.io docs](https://api-sports.io/documentation/football/v3) |
| **Sportmonks** | `sidelined` include (injuries **&** suspensions: type, start/end, games missed, `completed`) on teams/players/fixtures; **predicted** line-ups (include) + **Premium Expected Line-ups** endpoint | Paid (has free plan) | [injuries & suspensions](https://www.sportmonks.com/glossary/injuries-and-suspensions/) · [expected line-ups](https://docs.sportmonks.com/v3/endpoints-and-entities/endpoints/premium-expected-lineups) · [predicted line-ups](https://docs.sportmonks.com/v3/tutorials-and-guides/tutorials/includes/predicted-lineups) |

> **Timing of API line-ups (leakage-critical):** API-Football exposes confirmed line-ups only **~20–40 minutes before kickoff** (when the competition is covered). Sportmonks separates **predicted** (pre-match forecast) from confirmed. Log the *retrieval timestamp*, not just kickoff, so back-tests never use a line-up you couldn't have had (see §7).

---

## 4) Datasets for back-testing availability

Historical injury/availability data is far scarcer than results data — these are the practical open options (verify freshness before use).

| Dataset / repo | Content | Free/Paid | Link |
|---|---|---|---|
| **Player Injuries & Team Performance** (Kaggle, amritbiswas007) | EPL injuries joined to team performance | Free | [kaggle](https://www.kaggle.com/datasets/amritbiswas007/player-injuries-and-team-performance-dataset) |
| **soccer-players-injuries** (Kaggle, eliesemmel) | Per-player injury history, European leagues | Free | [kaggle](https://www.kaggle.com/datasets/eliesemmel/soccerplayersinjuries) |
| **Football Injuries for Performance Optimization** (Kaggle, lakshmimukundan93) | League injury records | Free | [kaggle](https://www.kaggle.com/datasets/lakshmimukundan93/football-injuries-for-performance-optimization) |
| **Football Player Injury Data** (Kaggle, kolambekalpesh) | Player injury records | Free | [kaggle](https://www.kaggle.com/datasets/kolambekalpesh/football-player-injury-data) |
| **yasirbm/soccer_player_injury_model** | ML pipeline predicting *injury risk* (preprocessing, features, training) — adjacent, useful template | Free (OSS) | [github](https://github.com/yasirbm/soccer_player_injury_model) |

> **What stats providers do NOT give you:** FBref/Understat/StatsBomb (see [tools page](./Open_Source_Tools_and_Libraries.md)) provide *historical minutes played* — excellent for **expected-minutes / rotation** models — but **no forward-looking availability**. You must join them to an injury/line-up feed above.

---

## 5) Modeling player availability

The practical goal is a per-team, per-match **availability-adjusted strength** that shifts your λ / rating *before* odds settle.

- **Key-player absence impact (ausência de jogador-chave).** Estimate each player's marginal contribution (e.g. team xG-for / xG-against with vs without, or a plus-minus / VAEP-style value from the [models page](./Innovative_Models_and_Deep_Learning.md)) and subtract it when out. A missing prolific striker hits **Over/Under and BTTS (ambos marcam)** as much as 1X2. Shrink small samples — most "with/without" splits are noisy.
- **Goalkeeper changes (troca de goleiro).** A backup GK is a discrete, often under-priced shift in expected goals conceded; treat GK as its own availability feature, not one of eleven.
- **Expected minutes / rotation (minutagem esperada).** Model P(start) and P(60′+) per player from line-up history and manager tendency; aggregate to a probable-XI strength rather than assuming the nominal best XI.
- **Rotation in congestion (rodízio em calendário apertado).** Managers rest players with <4 days' recovery and in dead-rubber or cup ties; congestion both *raises injury risk* and *predicts rotation* ([fixture-congestion systematic review, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9758680/) · [ML for injuries, Sports Med Open](https://link.springer.com/article/10.1186/s40798-022-00465-4)).
- **Returning-from-injury uncertainty (incerteza de retorno).** "Back in the squad" ≠ 90 minutes; weight expected minutes down for first games back and model the fitness ramp.

---

## 6) Suspensions, international duty & fatigue

- **Suspensions (suspensões).** Straight red = automatic ban; **accumulated yellow cards** trigger bans at league-specific thresholds (e.g. the Premier League's 5-yellow cut-off). These are *fully deterministic and knowable in advance* — the cleanest availability signal. Track via **Transfermarkt** (`/sperrenundverletzungen/`) and **Sky Sports'** suspension tracker (§2).
- **International-duty fatigue (fadiga de data FIFA).** Post-international windows raise injury/rotation risk, amplified by long-haul travel and unaccustomed load ([match workload + international travel, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11360389/) · [International Match Calendar expert statement, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12591028/)). Flag matches immediately after breaks and for players who travelled farthest.
- **Congestion windows.** Midweek European/cup ties before a weekend league game are the highest-rotation, highest-risk spots — a strong prior even before team news drops.

---

## 7) Timing & leakage — only use pre-kickoff info

This is where most amateur "edges" are actually **look-ahead bias (vazamento / data leakage)**.

- **Official release is late.** From the **2024/25 season the Premier League publishes confirmed starting XIs ~75 minutes before kickoff** on its app and website — earlier than in previous seasons ([Premier League, Aug 2024](https://www.premierleague.com/en/news/4081650)). A model trained on the *confirmed* XI can only be *deployed* after that release — otherwise you are cheating your back-test.
- **Golden rule:** every availability feature must carry a **timestamp**; at prediction time use only what was known then (predicted XI + injury table as of T−minutes), never the confirmed XI unless you are pricing *inside* the confirmation window.
- **Confirmation as a live edge.** The **line-up drop is itself a discrete news event**: markets reprice within seconds. Empirically, football betting markets are broadly efficient yet show pockets of mispricing around news and in-play ([Angelini & De Angelis 2019, *Int. J. Forecasting*](https://www.sciencedirect.com/science/article/abs/pii/S0169207018301134) · [in-play informational efficiency, Singleton et al.](https://www.carlsingletoneconomics.com/uploads/4/2/3/0/42306545/information_efficiency_angelini_de_angelis_singleton.pdf)). Any realistic edge requires being *faster or better-calibrated than the closing line* — an execution/latency problem, not just a modelling one. See [Exchanges & Trading](./Betting_Exchanges_Trading_and_Microstructure.md).

---

## 8) Reality check & responsible-gambling note

- **The data is the same for everyone.** Injury tables, predicted XIs and confirmation feeds are watched by every trader; treating them as a secret money-printer is the fastest way to lose. Any residual edge is small, perishable, and eaten by margin/commission.
- **Uncertainty dominates.** Predicted line-ups are wrong often; "doubtful" players start, "certain" starters are late scratches. Model this uncertainty — do **not** bet as if a predicted XI were confirmed.
- **This page builds features and studies markets — it does not tell you to bet.** Research & education only (**não é dica de aposta**). Markets are efficient; **most people lose money gambling.**

| Region | Resource | Contact |
|---|---|---|
| 🌍 Global | [Gambling Therapy](https://www.gamblingtherapy.org/) | Online chat/forum, multilingual |
| 🇬🇧 UK | [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) | National Gambling Helpline **0808 8020 133** (24/7) |
| 🇧🇷 Brazil | [Jogo Responsável — SPA/Ministério da Fazenda](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) | Apoio emocional **CVV 188** |

---

## Related in AIForge
- [Match Prediction Models & Techniques](./Match_Prediction_Models_and_Techniques.md) · [Innovative Models & Deep Learning](./Innovative_Models_and_Deep_Learning.md) · [Global Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md) · [Open-Source Tools & Libraries](./Open_Source_Tools_and_Libraries.md) · [Bet Selection & Staking](./Bet_Selection_Staking_and_High_Odds_Analysis.md) · [Betting Exchanges & Trading](./Betting_Exchanges_Trading_and_Microstructure.md) · [Fantasy Football & FPL](./Fantasy_Football_and_FPL_Analytics.md) · Parent vertical: [`../`](../) (Sports Analytics AI)

**Sources:** sofascore.com · fotmob.com (+ lineup-builder) · whoscored.com/previews · rotowire.com/soccer/lineups.php · sportsgambler.com (lineups + injuries) · fantasyfootballscout.co.uk/team-news · physioroom.com/advice/premier-league-injury-table · premierinjuries.com (injury-table.php, newsroom, expert/ben-dinnery) · transfermarkt.com (verletztespieler, sperrenundverletzungen) · premierleague.com/en/latest-player-injuries + /news/4081650 · skysports.com PL injuries/suspension tracker · nbcsports.com PL injury table 2025-26 · api-football.com (documentation-v3, injuries endpoint) + api-sports.io/documentation/football/v3 · sportmonks.com (injuries-and-suspensions glossary; docs.sportmonks.com premium/predicted line-ups) · kaggle.com (amritbiswas007 player-injuries-and-team-performance; eliesemmel soccerplayersinjuries; lakshmimukundan93 football-injuries-for-performance-optimization; kolambekalpesh football-player-injury-data) · github.com/yasirbm/soccer_player_injury_model · sciencedirect.com S0169207018301134 (Angelini & De Angelis 2019, Int. J. Forecasting) · carlsingletoneconomics.com in-play efficiency (Angelini, De Angelis & Singleton) · pmc.ncbi.nlm.nih.gov PMC9758680 / PMC11360389 / PMC12591028 · link.springer.com 10.1186/s40798-022-00465-4 (ML for injuries, Sports Med Open 2022) · sportresolutions.com / Howden 2023-24 Men's European Football Injury Index (4,123 injuries, €732m) · begambleaware.org · gamcare.org.uk · gamblingtherapy.org · gov.br/fazenda (SPA)

**Keywords:** football injury data, predicted lineups, confirmed lineup, team news betting, player availability, escalação provável, tabela de lesões, suspensões, PhysioRoom, Premier Injuries Ben Dinnery, Transfermarkt injuries, API-Football injuries endpoint, Sportmonks sidelined, Sofascore FotMob lineups, expected minutes rotation, key player absence xG, goalkeeper change, international duty fatigue, fixture congestion, data leakage look-ahead, 75-minute lineup rule, closing line efficiency, minutagem esperada, rodízio de elenco, fadiga data FIFA, previsão de escalação, jogo responsável, não é dica de aposta.
