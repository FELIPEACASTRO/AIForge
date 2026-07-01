# Asia & India — Football Data, Leagues & Markets

> Authoritative, regional index of **where to get football (soccer / futebol) data** for match & betting-prediction research across the **AFC region** — East Asia (🇯🇵 Japan, 🇰🇷 Korea, 🇨🇳 China), West Asia / Gulf (🇸🇦 Saudi, 🇶🇦 Qatar, 🇦🇪 UAE, 🇮🇷 Iran), South Asia (🇮🇳 India), Southeast Asia (🇹🇭 Thailand, 🇮🇩 Indonesia, 🇻🇳 Vietnam) and 🇦🇺 Australia — plus the **Asian Handicap** market (its Asian origin, mechanics, efficiency). Real URLs, free-vs-paid marks, current 2024–2026. Built for **data science / ML research & education**.

> ⚠️ **Research & education only — not betting advice.** Betting markets are **highly efficient**, and the **Asian Handicap (AH)** market in particular is priced by sharp, high-limit Asian books that are very hard to beat. The first published study of AH efficiency (Constantinou, *J. Sports Analytics* 2022, [arXiv:2003.09384](https://arxiv.org/abs/2003.09384)) found only marginal, fragile edges over 13 EPL seasons. **Most bettors lose money over time.** Nothing here is a tip or a system. If gambling is a problem, get help: [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) · [Gambling Therapy](https://www.gamblingtherapy.org/) (multilingual) · 🇧🇷 [Jogo Responsável](https://www.gov.br/fazenda/pt-br/assuntos/loterias/jogo-responsavel) · 🇧🇷 CVV **188**.

**How to choose:** free CSV history with odds ([football-data.co.uk](https://www.football-data.co.uk/data.php) — Japan & China) for quick baselines → free open **event/tracking** data ([StatsBomb](https://github.com/statsbomb/open-data) ISL, [SkillCorner](https://github.com/SkillCorner/opendata) A-League) for advanced features → **official league portals** (J.League, K League) for clean reference data → **live/reference APIs** ([API-Football](https://www.api-football.com/), [SportMonks](https://www.sportmonks.com/), Asia-focused [iSports](https://www.isportsapi.com/)) for pipelines → paid providers (Opta/Stats Perform, Sportradar, Wyscout) for coverage depth. Sibling pages: [Global Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md) · [Kaggle Football Datasets](./Kaggle_Football_Datasets_and_Competitions.md).

---

## 1) The AFC league landscape (what to model)

The Asian Football Confederation (AFC) has 47 member associations. Data availability is **very uneven**: East-Asian and Gulf leagues are well-covered; Southeast-Asian and Central-Asian leagues are thin. Odds liquidity, however, is deep almost everywhere because Asia is the world's largest football-betting market.

| Country | Top league (local name) | Tier / notes | Continental slots |
|---|---|---|---|
| 🇯🇵 Japan | **J1 League** (Ｊ１リーグ) — J-League; J2, J3 below | Best-run data ecosystem in Asia; official Data Site | ACL Elite / ACL Two |
| 🇰🇷 South Korea | **K League 1** (K리그1); K League 2 | Official open data portal; strong analytics | ACL Elite / ACL Two |
| 🇨🇳 China | **Chinese Super League (CSL)** (中超) | 2025: operations moved to new **CFL** operator; still "CSL" | ACL Elite / ACL Two |
| 🇸🇦 Saudi Arabia | **Saudi Pro League (SPL / Roshn)** (دوري روشن) | PIF-funded; Ronaldo/Benzema/Neymar era; hosts ACL Elite finals | ACL Elite (up to 3–4 clubs) |
| 🇶🇦 Qatar | **Qatar Stars League (QSL)** (دوري نجوم قطر) | World Cup 2022 legacy infra | ACL Elite / ACL Two |
| 🇦🇪 UAE | **UAE Pro League (ADNOC)** | Strong Gulf book/odds interest | ACL Elite / ACL Two |
| 🇮🇷 Iran | **Persian Gulf Pro League** (لیگ برتر خلیج فارس) | Large, competitive; sparse official digital data | ACL Elite / ACL Two |
| 🇮🇳 India | **Indian Super League (ISL)** + **I-League** | ISL = top tier (AIFF); free StatsBomb season exists | ACL Two / AFC playoffs |
| 🇹🇭 Thailand | **Thai League 1 (T1)** | Best-covered SE-Asian league | ACL Two |
| 🇮🇩 Indonesia | **Liga 1** | Huge fanbase; often cited as the origin of the Asian Handicap (claim is weakly sourced) | ACL Two |
| 🇻🇳 Vietnam | **V.League 1** | Growing; limited structured data | ACL Two |
| 🇦🇺 Australia | **A-League Men** (AFC member since 2006) | Open SkillCorner tracking; Opta-covered | ACL Elite / ACL Two |

> Note: **I-League** is now India's second tier below ISL (the two swapped status under AIFF/FSDL restructuring; promotion/relegation between them is a live governance topic in 2024–2026).

---

## 2) Free datasets & official portals (per country)

Prefer these for reproducible research — they are free and stable. **Odds** columns exist only for football-data.co.uk (Japan, China).

| Country / League | Source | What you get | Free/Paid | URL |
|---|---|---|---|---|
| 🇯🇵 Japan J1/J2 | **football-data.co.uk (Japan)** | Full-time results + **pre-match odds** (market best/avg + **Pinnacle**; incl. Asian-handicap & O/U columns), recent seasons (from ~2012), CSV | **Free** | [japan.php](https://www.football-data.co.uk/japan.php) |
| 🇨🇳 China CSL | **football-data.co.uk (China)** | Results + **pre-match odds** (market best/avg + Pinnacle; incl. AH & O/U columns), recent seasons, CSV | **Free** | [china.php](https://www.football-data.co.uk/china.php) |
| 🇯🇵 Japan J1/J2/J3 | **J.League Data Site** (official) | Schedules, results, standings, appearance/goal records; EN + JP | **Free** (view; redistribution restricted) | [data.j-league.or.jp](https://data.j-league.or.jp/SFTP01/) · [Digital Data Book](https://ddb.j-league.or.jp/) |
| 🇰🇷 Korea K1/K2 | **K League Data Portal** (official) | Official records, match/club/player stats | **Free** (view; ©KPFL, no redistribution) | [data.kleague.com](https://data.kleague.com/) |
| 🇰🇷 Korea | **kmangyo/Kleague_Data** (GitHub) | Korean league data-analysis repo (scraped) | **Free (OSS)** | [github](https://github.com/kmangyo/Kleague_Data) |
| 🇸🇦 Saudi SPL | **alioh/Saudi-Professional-League-Datasets** | CSV match results 2000/01–2017/18 (FlashScore + Slstat) | **Free (OSS)** | [github](https://github.com/alioh/Saudi-Professional-League-Datasets) |
| 🇸🇦 Saudi SPL | **Kaggle: SPL Stats — Last 4 Seasons** (saudidata2030) | Recent SPL player/team stats | **Free** | [kaggle](https://www.kaggle.com/datasets/saudidata2030/spl-stats-last-4-years) |
| 🇮🇳 India ISL | **StatsBomb Open Data** — Indian Super League **2021/22** | Full **event data** (every on-ball action, x/y) as JSON | **Free** (attribution) | [github](https://github.com/statsbomb/open-data) · [announce](https://blogarchive.statsbomb.com/news/statsbomb-announce-the-release-of-free-indian-super-league-data/) |
| 🇮🇳 India ISL | **Kaggle ISL** (amaanafif) · (sahilmaheshwari) | Match/season results, standings, tabular | **Free** | [amaanafif](https://www.kaggle.com/datasets/amaanafif/indian-super-league) · [sahilmaheshwari](https://www.kaggle.com/datasets/sahilmaheshwari/indian-super-league-dataset) |
| 🇦🇺 Australia A-League | **SkillCorner Open Data** | Broadcast **tracking** (CV/ML) — 10 matches 2024/25 + season aggregates (physical, off-ball runs, passing) | **Free** (w/ PySport) | [github](https://github.com/SkillCorner/opendata) |
| 🌏 Asian internationals | **martj42 — International Results 1872→now** | Every men's international incl. **AFC Asian Cup** & WC qualifiers | **Free** | [kaggle](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) |
| 🌏 Multi-league | **adamgbor — Club Football Match Data 2000–2025** | 27 countries incl. some Asian leagues; results + ClubElo + pre-match odds | **Free** | [kaggle](https://www.kaggle.com/datasets/adamgbor/club-football-match-data-2000-2025) |

> ⚠️ Official portals (J.League, K League) permit **viewing**, not bulk redistribution/commercial reuse — scrape responsibly and cache locally; check each site's terms.

---

## 3) Cross-region live/reference APIs & scrapers (cover Asia)

For leagues without a tidy CSV (Korea, Gulf, SE-Asia), pull from these. Asia-focused providers (**iSports**, **BetsAPI**) natively expose **Asian Handicap** odds.

| Source | Asian coverage | Free tier | Paid from | URL |
|---|---|---|---|---|
| **API-Football** (api-sports.io) | 1,200+ leagues/cups incl. J1, K1/K2, CSL, SPL, QSL, UAE, ISL, A-League; fixtures, lineups, events, stats, **pre-match odds** | ✅ **100 req/day**, all endpoints | **$19/mo** | [api-football.com](https://www.api-football.com/) · [coverage](https://www.api-football.com/coverage) |
| **SportMonks** | 2,300+ leagues; explicit **APAC** (J.League, K League, ISL, A-League); live, stats, odds, Predictions API | 🆓 trial (2 leagues) | €29 → Enterprise (all leagues) | [sportmonks.com](https://www.sportmonks.com/football-api/) · [coverage](https://www.sportmonks.com/football-api/coverage/) |
| **iSports API** | **Asia-specialist**: J-League, K-League, CSL + Asian books; **Asian Handicap / O-U** odds feeds | 🆓 trial | Paid tiers | [isportsapi.com](https://www.isportsapi.com/) |
| **BetsAPI** | Results + odds incl. Asian bookmakers (e.g. Saudi Pro League listing); AH markets | 🆓 limited | Paid | [betsapi.com](https://betsapi.com/) |
| **The Odds API** | Bookmaker odds incl. some Asian competitions; h2h/spreads/totals | ✅ 500 req/mo | Paid tiers | [the-odds-api.com](https://the-odds-api.com/sports-odds-data/football-odds.html) |
| **football-data.org** | Free tier is Europe/WC-centric (limited Asia); good for structure/learning | ✅ 10 calls/min | €49+ | [football-data.org/coverage](https://www.football-data.org/coverage) |
| **Sofascore** | Global incl. all Asian leagues (QSL, UAE, Liga 1, V.League, Thai L1, Persian Gulf) — ratings, xG, momentum | ➖ no official public API | scrape only | via [ScraperFC](https://github.com/oseymour/ScraperFC) / [soccerdata](https://github.com/probberechts/soccerdata) |
| **Transfermarkt** | Squads & **market values** — Saudi (`/wettbewerb/SA1`), J1, K1, ISL, etc. | scrape | via mirror | [SPL market values](https://www.transfermarkt.com/saudi-pro-league/marktwerte/wettbewerb/SA1) · [dcaribou datasets](https://github.com/dcaribou/transfermarkt-datasets) |
| **Sportradar / Opta (Stats Perform) / Wyscout (Hudl)** | Official/enterprise event & tracking for most Asian top leagues | 🆓 trial keys | Enterprise | [Sportradar](https://developer.sportradar.com/soccer/reference/soccer-api-overview) · [Opta](https://www.statsperform.com/opta-data/) |

**Scraper toolkits** (Python/R, free/OSS): [`soccerdata`](https://github.com/probberechts/soccerdata) (FBref, Sofascore, ClubElo, Football-Data.co.uk…), [`ScraperFC`](https://github.com/oseymour/ScraperFC), [`worldfootballR`](https://github.com/JaseZiv/worldfootballR), [`statsbombpy`](https://github.com/statsbomb/statsbombpy) + [`mplsoccer`](https://github.com/andrewRowlinson/mplsoccer) for the ISL open data.

> ⚠️ **FBref caveat (important, 2026):** FBref historically carried Asian leagues (J1 `comps/25`, K League 1 `comps/55`, Saudi PL `comps/70`, Persian Gulf `comps/64`, A-League `comps/65`, ISL `comps/82`). On **2026-01-20**, Sports Reference announced **advanced stats (xG, etc.) were removed** after its Opta/Stats Perform feed deal ended. Basic results/schedules remain; treat FBref advanced-metric pipelines for Asia as **at-risk** and verify freshness. [Understat](https://understat.com/) does **not** cover any Asian league (top-5 EU + Russian PL only).

---

## 4) Continental competitions (AFC) — data & fixtures

| Competition | Format (2024–2026) | Data source | Free/Paid |
|---|---|---|---|
| **AFC Champions League Elite (ACL Elite)** | Rebranded 2024/25; 24 clubs split **East/West**, 8-game league phase; centralized single-venue finals **in Saudi Arabia** (Jeddah) 2024/25 & 2025/26 | API-Football, SportMonks, Sofascore; [Wikipedia 2025–26](https://en.wikipedia.org/wiki/2025%E2%80%9326_AFC_Champions_League_Elite) | Free/Paid |
| **AFC Champions League Two (ACL Two)** | Second-tier continental club cup | API-Football, Sofascore | Free/Paid |
| **AFC Asian Cup** | National-team championship; **2027 hosted by 🇸🇦 Saudi Arabia** | martj42 internationals; [eloratings.net](https://eloratings.net/) | Free |

---

## 5) The **Asian Handicap** (handicap asiático) — origin, mechanics, why it dominates

The Asian Handicap is the defining football-betting market of the region and increasingly worldwide. **Educational explainer only.**

**Origin (Asia).** AH is commonly traced to **Indonesia**, where bookmakers ran a line-handicap format known as **"hang cheng" betting** — though the specific Indonesia-origin claim is only weakly sourced. The English term **"Asian handicap"** was coined by UK journalist **Joe Saumarez Smith in November 1998** at the request of Indonesian bookmaker **Joe Phan**, who asked for a translation of hang-cheng betting ([Wikipedia: Asian handicap](https://en.wikipedia.org/wiki/Asian_handicap)). It then spread across Asia and into Europe.

**Mechanics.** A virtual goal head-start/deficit is applied to one team so the market is (near) a 50/50 two-way price; **the draw is removed** as a distinct outcome. Handicaps come in **whole, half, and quarter** (split) lines:

| Line (fav) | You back favourite −X | Result if match is… |
|---|---|---|
| **−0.5** | wins by ≥1 | win / lose (no push) |
| **−1.0** | wins by ≥2 | win; **exactly 1 → push** (stake returned); else lose |
| **−0.25** (−0 & −0.5) | half stake on 0, half on −0.5 | draw → **half-loss**; win → win |
| **−0.75** (−0.5 & −1.0) | split | win by 1 → **half-win**; win by 2+ → win; draw/loss → lose |

There is also the **Asian total (Over/Under goal line)** using the same quarter-line, half-stake logic.

**Why it dominates Asian books.** (1) Two-way, near-even pricing lets books run **very low margins and very high limits** — attractive to large/sharp bettors; (2) quarter lines finely tune probability and reduce pushes; (3) it suits in-play and high-turnover markets. The flip side for a modeller: because these lines aggregate enormous sharp volume, they are **extremely efficient** — Constantinou (2022, [arXiv:2003.09384](https://arxiv.org/abs/2003.09384)) built the first published Bayesian-network model to assess AH efficiency and found **beating it consistently is very hard**; realistic skill is measured by **Closing Line Value (CLV)**, not by isolated wins ([Pinnacle: CLV](https://www.pinnacle.com/betting-resources/en/educational/what-is-closing-line-value-clv-in-sports-betting)).

> ⚠️ **Books/exchanges context (educational, not endorsement):** Asian markets historically centre on sharp books (e.g. Pinnacle) and Asian-facing operators/aggregators; many jurisdictions restrict or ban these. Availability and legality vary by country — **know your local law**. This page does **not** direct anyone to any operator.

---

## 6) India deep-dive (🇮🇳 futebol na Índia)

**ISL data & analytics.** The **Indian Super League** (top tier, run under AIFF; commercial history with FSDL, and 2025 talks with **Genius Sports** over data-driven commercial rights) is the best-documented Indian competition. The standout free asset is **StatsBomb's open ISL event data (2021/22 season)** — a rare, full event corpus for a South-Asian league, ideal for xG/xT teaching ([StatsBomb open-data](https://github.com/statsbomb/open-data)). Tabular results/standings live on Kaggle (amaanafif, sahilmaheshwari) and community sites like [Grey Area Analytics](https://www.greyareaanalytics.com/). Official federation: [AIFF](https://www.the-aiff.com/).

**Research signal.** Nanavati & Rangaswamy, *"Bridging Data Gaps and Building Knowledge Networks in Indian Football Analytics"* (arXiv, Apr 2025, [2504.16572](https://arxiv.org/abs/2504.16572)), documents how **institutional resistance, infrastructure limits, and fragmented governance** constrain Indian football analytics, and how informal amateur/"data-sleuth" communities fill the gap — essential context before you trust Indian football data coverage.

**Cricket vs football data maturity.** India's analytics depth is heavily skewed to **cricket** (IPL): ball-by-ball public data, mature fantasy/broadcast analytics, and a decade-plus of stat culture. Football lags on structured, granular data — so expect **richer cricket datasets** on Kaggle and **thinner, more fragmented** football data. This asymmetry is the single most important caveat for Indian football modelling.

**Fantasy (Dream11) context.** [Dream11](https://medium.com/dream11-tech-blog) is India's largest fantasy-sports platform (cricket, football, kabaddi) operating at very large scale (its engineering blog describes serving **100M+ users** and tens of millions of requests/minute at peak). Fantasy platforms are legally treated in India as **games of skill** (distinct from betting) and are a major driver of sports-data demand — but Dream11 exposes **no public data API**; researchers rely on public match feeds, not platform internals.

**Betting-regulation note (educational, not advice).** Betting/gambling is a **State subject** under the Seventh Schedule of India's Constitution — historically governed by the colonial-era **Public Gambling Act, 1867** plus divergent state laws (a few states permit specific forms; most restrict real-money betting). In **2025**, Parliament passed the **Promotion and Regulation of Online Gaming Act, 2025** (Presidential assent 22 Aug 2025), which **prohibits online real-money games involving betting/wagering** while promoting e-sports and social games (administration/rules phasing in through 2026) — see [Wikipedia](https://en.wikipedia.org/wiki/Promotion_and_Regulation_of_Online_Gaming_Act,_2025) and the [official bill PDF (PIB)](https://static.pib.gov.in/WriteReadData/specificdocs/documents/2025/aug/doc2025821618101.pdf). **Implication for this index:** treat Indian football-betting data as **research-only**; real-money online sports betting faces tightening national prohibition.

---

## 7) Reality check — why "beating the Asian market" is so hard

- Asian books price with **low margin + high limits**, so their **closing lines** are near-efficient estimates of true probability; late sharp money and team news are already baked in.
- AH efficiency has been formally studied: Constantinou (2022) found any edge is **small and fragile**, and Kaunitz et al. (2017, [arXiv:1710.02824](https://arxiv.org/abs/1710.02824)) showed even a back-test "winning" strategy gets **limited/closed** by books in practice.
- The honest skill metric is **Closing Line Value (CLV)** — do your prices *repeatedly* beat the close? — not a lucky winning streak.
- The **overround / vig** (margem) means a naive bettor loses on average even with fair-looking skill. **Treat all of this as modelling practice, not income.**

---

## 8) Responsible gambling (Jogo Responsável) — mandatory

Gambling can cause serious harm. This index exists for **data science / ML research and education only**. Set limits, never chase losses, and seek help if betting stops being fun.

| Region | Resource | Contact |
|---|---|---|
| 🌍 Global | [Gambling Therapy](https://www.gamblingtherapy.org/) | Online chat/forum, **multilingual (incl. Asian languages)** |
| 🇬🇧 UK | [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) | National Gambling Helpline **0808 8020 133** (24/7) |
| 🇮🇳 India | Mandatory operator safeguards under the Online Gaming Act 2025 (self-exclusion, deposit/time limits, age checks) | State helplines vary; [iCall psychosocial helpline](https://icallhelpline.org/) |
| 🇧🇷 Brazil | [Jogo Responsável — Ministério da Fazenda / SPA](https://www.gov.br/fazenda/pt-br/assuntos/loterias/jogo-responsavel) | Apoio emocional **CVV 188** |

---

## Related in AIForge
- [Global Football Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md) · [Kaggle Football Datasets & Competitions](./Kaggle_Football_Datasets_and_Competitions.md) · [Innovative Models & Deep Learning](./Innovative_Models_and_Deep_Learning.md) · [Open-Source Tools & Libraries](./Open_Source_Tools_and_Libraries.md) · Parent vertical: [`../`](../) (Sports Analytics AI)

**Sources:** football-data.co.uk (japan.php, china.php) · data.j-league.or.jp · ddb.j-league.or.jp · data.kleague.com · github.com/kmangyo/Kleague_Data · github.com/alioh/Saudi-Professional-League-Datasets · github.com/statsbomb/open-data · blogarchive.statsbomb.com · github.com/SkillCorner/opendata · Kaggle (saudidata2030, amaanafif, sahilmaheshwari, martj42, adamgbor) · api-football.com/coverage · sportmonks.com/football-api/coverage · isportsapi.com · betsapi.com · the-odds-api.com · football-data.org/coverage · transfermarkt.com (SA1) · github.com/dcaribou/transfermarkt-datasets · github.com/probberechts/soccerdata · github.com/oseymour/ScraperFC · github.com/JaseZiv/worldfootballR · fbref.com (comps 25/55/64/65/70/82) · understat.com · en.wikipedia.org/wiki/Asian_handicap · en.wikipedia.org/wiki/2025–26_AFC_Champions_League_Elite · en.wikipedia.org/wiki/Promotion_and_Regulation_of_Online_Gaming_Act,_2025 · pib.gov.in · the-aiff.com · greyareaanalytics.com · medium.com/dream11-tech-blog · arxiv.org/abs/2504.16572 · arxiv.org/abs/2003.09384 · arxiv.org/abs/1710.02824 · pinnacle.com · begambleaware.org · gamcare.org.uk · gamblingtherapy.org · icallhelpline.org · gov.br/fazenda

**Keywords:** Asian football data, AFC leagues dataset, J-League data, J.League Data Site, K League data portal, Chinese Super League data, Saudi Pro League dataset, Indian Super League ISL data, StatsBomb ISL open data, Persian Gulf Pro League, Qatar Stars League, UAE Pro League, A-League tracking data, AFC Champions League Elite, AFC Asian Cup, Asian Handicap, handicap asiático, hang cheng, quarter goal line, Dream11 fantasy, India online gaming act, closing line value, betting market efficiency, jogo responsável, dados de futebol asiático, previsão de partidas, mercado de apostas eficiente.
